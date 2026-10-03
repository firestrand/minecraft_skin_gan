"""Generative autoencoder with the original dense architecture and training objectives."""

import logging
import os
import re
from pathlib import Path
from typing import Any

# Respect an explicitly selected supported Keras backend.
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from keras.initializers import RandomNormal  # noqa: E402
from keras.layers import Dense, Flatten, Input, Reshape  # noqa: E402
from keras.models import Model, Sequential  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from numpy.typing import ArrayLike, NDArray  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.neighbors import KernelDensity  # noqa: E402

logger = logging.getLogger(__name__)
type Array = NDArray[Any]


def approximateLogLiklihood(
    x_generated: ArrayLike, x_test_input: ArrayLike, search_space: ArrayLike | None = None
) -> float:
    """Return the mean KDE log density using the historical bandwidth search."""
    generated = np.asarray(x_generated)
    test = np.asarray(x_test_input)
    generated = generated.reshape((len(generated), -1))
    test = test.reshape((len(test), -1))
    params = {"bandwidth": np.logspace(-4, 0, 5) if search_space is None else search_space}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=4)
    grid.fit(generated)
    logger.info("KDE bandwidth: %s", grid.best_params_)
    return float(np.mean(grid.best_estimator_.score_samples(test)))


def findNearest(x_train: Array, x_test: Array) -> Array:
    """Return the first training image with the lowest squared pixel distance."""
    diff = np.square(x_train - x_test)
    mse = [np.sum(x) for x in diff]
    return x_train[np.argmin(mse)]


def _resume_weights(model: Any, name: str) -> int:
    """Load the highest numbered current or legacy checkpoint; return completed epochs."""
    checkpoints = []
    for prefix in (f"weights_{name}", f"weights_mnist_{name}"):
        for path in Path("models").glob(f"{prefix}.*"):
            match = re.fullmatch(rf"{prefix}\.(\d+)\.(?:weights\.h5|hdf5|h5)", path.name)
            if match:
                checkpoints.append((int(match[1]), path))
    if not checkpoints:
        return 0
    epoch, path = max(checkpoints, key=lambda entry: entry[0])
    model.load_weights(path)
    return epoch


def _callbacks(name: str) -> list[Any]:
    return [
        keras.callbacks.ModelCheckpoint(
            f"models/weights_{name}.{{epoch:02d}}.weights.h5",
            save_weights_only=True,
            save_freq="epoch",
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, min_delta=1e-4, restore_best_weights=True
        ),
    ]


class GAE:
    def __init__(self, img_shape: tuple[int, ...] = (28, 28), encoded_dim: int = 2) -> None:
        self.img_shape = img_shape
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.001)
        self.optimizer_discriminator = Adam(0.00001)
        # Keras 3 optimizers own a fixed set of variables. Do not share the
        # autoencoder optimizer with the independent discriminator.
        self._discriminator_optimizer = Adam(0.001)
        self.discriminator = self.get_discriminator_model(img_shape)
        self.decoder = self.get_decoder_model(encoded_dim, img_shape)
        self.encoder = self.get_encoder_model(img_shape, encoded_dim)
        img = Input(shape=img_shape)
        self.autoencoder = Model(img, self.decoder(self.encoder(img)))
        latent = Input(shape=(encoded_dim,))
        self.decoder_discriminator = Model(latent, self.discriminator(self.decoder(latent)))
        self.initialize_full_model(encoded_dim)

    def initialize_full_model(self, encoded_dim: int) -> None:
        self.autoencoder.compile(optimizer=self.optimizer, loss="mse")
        self.discriminator.trainable = True
        self.discriminator.compile(
            optimizer=self._discriminator_optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.discriminator.trainable = False
        self.decoder_discriminator.compile(
            optimizer=self.optimizer_discriminator, loss="binary_crossentropy", metrics=["accuracy"]
        )

    @staticmethod
    def get_encoder_model(img_shape: tuple[int, ...], encoded_dim: int) -> Any:
        return Sequential(
            [
                Input(shape=img_shape),
                Flatten(),
                Dense(1000, activation="relu"),
                Dense(1000, activation="relu"),
                Dense(encoded_dim),
            ]
        )

    @staticmethod
    def get_decoder_model(encoded_dim: int, img_shape: tuple[int, ...]) -> Any:
        return Sequential(
            [
                Input(shape=(encoded_dim,)),
                Dense(1000, activation="relu"),
                Dense(1000, activation="relu"),
                Dense(int(np.prod(img_shape)), activation="sigmoid"),
                Reshape(img_shape),
            ]
        )

    @staticmethod
    def get_discriminator_model(img_shape: tuple[int, ...]) -> Any:
        # Keras 3 repeats draws for a reused integer/None seed. A per-model
        # SeedGenerator advances across layers and honors set_random_seed.
        initializer = RandomNormal(mean=0.0, stddev=0.01, seed=keras.random.SeedGenerator())
        return Sequential(
            [
                Input(shape=img_shape),
                Flatten(),
                Dense(
                    1000,
                    activation="relu",
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                ),
                Dense(
                    1000,
                    activation="relu",
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                ),
                Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                ),
            ]
        )

    def imagegrid(self, epochnumber: int) -> None:
        fig = plt.figure(figsize=[20, 20])
        try:
            for i in range(-5, 5):
                for j in range(-5, 5):
                    img = self.decoder.predict(np.array([[i * 0.5, j * 0.5]]), verbose=0)
                    ax = fig.add_subplot(10, 10, (i + 5) * 10 + j + 5 + 1)
                    ax.set_axis_off()
                    ax.imshow(img.reshape(self.img_shape))
            fig.savefig(f"{epochnumber}.png")
            plt.show()
        finally:
            plt.close(fig)

    def train(self, x_train_input: Array, batch_size: int = 128, epochs: int = 5) -> None:
        Path("models").mkdir(parents=True, exist_ok=True)
        saved_epoch = _resume_weights(self.autoencoder, "autoencoder")
        if saved_epoch < epochs:
            self.autoencoder.fit(
                x_train_input,
                x_train_input,
                batch_size=batch_size,
                epochs=epochs,
                initial_epoch=saved_epoch,
                callbacks=_callbacks("autoencoder"),
            )
        logger.info("Training KDE")
        codes = self.encoder.predict(x_train_input, verbose=0)
        self.kde = KernelDensity(kernel="gaussian", bandwidth=3.16).fit(codes)
        logger.info("Initial training of discriminator")
        saved_epoch = _resume_weights(self.discriminator, "discriminator")
        train_count = len(x_train_input)
        if saved_epoch < epochs:
            images = np.vstack([x_train_input, self.generate(n=train_count)])
            labels = np.vstack([np.ones((train_count, 1)), np.zeros((train_count, 1))])
            self.discriminator.trainable = True
            try:
                self.discriminator.fit(
                    images,
                    labels,
                    epochs=epochs,
                    initial_epoch=saved_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=_callbacks("discriminator"),
                )
            finally:
                self.discriminator.trainable = False
        logger.info("Training GAN")
        self.generateAndPlot(x_train_input, fileName="before_gan.png")
        self.trainGAN(x_train_input, epochs=int(train_count / batch_size), batch_size=batch_size)
        self.generateAndPlot(x_train_input, fileName="after_gan.png")

    def trainGAN(self, x_train_input: Array, epochs: int = 1000, batch_size: int = 128) -> None:
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            idx = np.random.randint(0, x_train_input.shape[0], half_batch)
            imgs_fake = self.generate(n=half_batch)
            self.discriminator.trainable = True
            try:
                d_loss_real = self.discriminator.train_on_batch(
                    x_train_input[idx], np.ones((half_batch, 1))
                )
                d_loss_fake = self.discriminator.train_on_batch(
                    imgs_fake, np.zeros((half_batch, 1))
                )
            finally:
                self.discriminator.trainable = False
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            codes = self.kde.sample(batch_size)
            g_similarity = self.decoder_discriminator.train_on_batch(
                codes, np.ones((batch_size, 1))
            )
            if epoch % 50 == 0:
                logger.info(
                    "epoch %d [D accuracy: %.2f] [G accuracy: %.2f]",
                    epoch,
                    d_loss[1],
                    g_similarity[1],
                )

    def generate(self, n: int = 10000) -> Array:
        return self.decoder.predict(self.kde.sample(n), verbose=0)

    def generateAndPlot(
        self, x_train_input: Array, n: int = 10, fileName: str = "generated.png"
    ) -> None:
        fig = plt.figure(figsize=[20, 20])
        try:
            images = self.generate(n * n)
            index = 1
            for image in images:
                image = image.reshape(self.img_shape)
                ax = fig.add_subplot(n, n + 1, index)
                index += 1
                ax.set_axis_off()
                ax.imshow(image)
                if index % (n + 1) == 0:
                    ax = fig.add_subplot(n, n + 1, index)
                    index += 1
                    ax.imshow(findNearest(x_train_input, image))
            fig.savefig(fileName)
            plt.show()
        finally:
            plt.close(fig)

    @staticmethod
    def mean_log_likelihood(x_test_input: Array) -> None:
        # Historical public method fits a KDE but returns no score.
        KernelDensity(kernel="gaussian", bandwidth=0.2).fit(x_test_input)


def main() -> None:
    with np.load("images/train_test.npz") as data:
        x_train = data["arr_0"].astype(np.float32) / 255.0
        x_test = data["arr_1"].astype(np.float32) / 255.0
    ann = GAE(img_shape=(64, 64, 4), encoded_dim=128)
    ann.train(x_train, epochs=50)
    encoded_imgs = ann.encoder.predict(x_test, verbose=0)
    decoded_imgs = ann.autoencoder.predict(x_test, verbose=0)
    Path("models").mkdir(parents=True, exist_ok=True)
    ann.autoencoder.save("models/autoencoder.keras")
    ann.decoder.save("models/decoder.keras")
    n, m = 10, 3
    fig = plt.figure(figsize=(n, m + 0.5))
    try:
        for i in range(1, n + 1):
            for row, image, shape in (
                (0, x_test[i], (64, 64, 4)),
                (1, encoded_imgs[i], (8, 4, 4)),
                (2, decoded_imgs[i], (64, 64, 4)),
            ):
                ax = plt.subplot(m, n, row * n + i)
                plt.imshow(image.reshape(shape))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.suptitle("Minecraft Skin GAE")
        Path("images/results").mkdir(parents=True, exist_ok=True)
        plt.savefig("images/results/mnist_gae_00.jpg", bbox_inches="tight")
        plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
