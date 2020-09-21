import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import glob
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV


def approximateLogLiklihood(x_generated, x_test_input, search_space=np.logspace(-4, 0, 5)):
    x_generated = np.array(x_generated).reshape((len(x_generated), -1))
    x_test_input = np.array(x_test_input).reshape((len(x_test_input), -1))
    # use grid search cross-validation to optimize the bandwidth
    print("new")
    params = {'bandwidth': search_space}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=4)
    grid.fit(x_generated)
    print(grid.best_params_)
    kde = grid.best_estimator_
    scores = kde.score_samples(x_test_input)
    return np.sum(scores) / len(scores)


def findNearest(x_train, x_test):
    diff = np.square(x_train - x_test)
    mse = [np.sum(x) for x in diff]
    return x_train[np.argmin(mse)]


initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)


class GAE:
    def __init__(self, img_shape=(28, 28), encoded_dim=2):
        self.img_shape = img_shape
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.001)
        self.optimizer_discriminator = Adam(0.00001)
        self.discriminator = self.get_discriminator_model(img_shape)
        self.decoder = self.get_decoder_model(encoded_dim, img_shape)
        self.encoder = self.get_encoder_model(img_shape, encoded_dim)
        # Initialize Autoencoder
        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        # Initialize Discriminator
        latent = Input(shape=(encoded_dim,))
        gen_image_from_latent = self.decoder(latent)
        is_real = self.discriminator(gen_image_from_latent)
        self.decoder_discriminator = Model(latent, is_real)
        # Finally compile models
        self.initialize_full_model(encoded_dim)

    def initialize_full_model(self, encoded_dim):
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        self.discriminator.compile(optimizer=self.optimizer,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        # Default start discriminator is not trainable
        for layer in self.discriminator.layers:
            layer.trainable = False

        self.decoder_discriminator.compile(optimizer=self.optimizer_discriminator,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])

    @staticmethod
    def get_encoder_model(img_shape, encoded_dim):
        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(encoded_dim))
        encoder.summary()
        return encoder

    @staticmethod
    def get_decoder_model(encoded_dim, img_shape):
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder

    @staticmethod
    def get_discriminator_model(img_shape):
        discriminator = Sequential()
        discriminator.add(Flatten(input_shape=img_shape))
        discriminator.add(Dense(1000, activation='relu',
                                kernel_initializer=initializer,
                                bias_initializer=initializer))
        discriminator.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                                bias_initializer=initializer))
        discriminator.add(Dense(1, activation='sigmoid', kernel_initializer=initializer,
                                bias_initializer=initializer))
        discriminator.summary()
        return discriminator

    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        for i in range(-5, 5):
            for j in range(-5, 5):
                topred = np.array((i * 0.5, j * 0.5))
                topred = topred.reshape((1, 2))
                img = self.decoder.predict(topred)
                img = img.reshape(self.img_shape)
                ax = fig.add_subplot(10, 10, (i + 5) * 10 + j + 5 + 1)
                ax.set_axis_off()
                ax.imshow(img)
        fig.savefig(str(epochnumber) + ".png")
        plt.show()
        plt.close(fig)

    def train(self, x_train_input, batch_size=128, epochs=5):
        fileNames = glob.glob('models/weights_mnist_autoencoder.*')
        fileNames.sort()
        if len(fileNames) != 0:
            saved_epoch = int(fileNames[-1].split('.')[1])
            self.autoencoder.load_weights(fileNames[-1])
        else:
            saved_epoch = -1
        if saved_epoch < epochs - 1:
            self.autoencoder.fit(x_train_input, x_train_input, batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=[
                                     keras.callbacks.ModelCheckpoint('models/weights_autoencoder.{epoch:02d}.hdf5',
                                                                     verbose=0,
                                                                     save_best_only=False,
                                                                     save_weights_only=False,
                                                                     mode='auto',
                                                                     period=1),
                                     keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=1e-4,
                                                                   restore_best_weights=True)])
        print("Training KDE")
        codes = self.encoder.predict(x_train_input)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=3.16).fit(codes)
        print("Initial Training of discriminator")
        fileNames = glob.glob('models/weights_mnist_discriminator.*')
        fileNames.sort()
        if len(fileNames) != 0:
            saved_epoch = int(fileNames[-1].split('.')[1])
            self.discriminator.load_weights(fileNames[-1])
        else:
            saved_epoch = -1

        train_count = len(x_train_input)
        if saved_epoch < epochs - 1:
            # Combine real and fake images for discriminator training
            imgs_fake = self.generate(n=train_count)
            valid = np.ones((train_count, 1))  # result for training images
            fake = np.zeros((train_count, 1))  # result for generated fakes
            labels = np.vstack([valid, fake])  # combine together
            images = np.vstack([x_train_input, imgs_fake])
            # Train the discriminator
            self.discriminator.fit(images, labels, epochs=epochs, batch_size=batch_size, shuffle=True,
                                   callbacks=[
                                       keras.callbacks.ModelCheckpoint(
                                           'models/weights_discriminator.{epoch:02d}.hdf5',
                                           verbose=0,
                                           save_best_only=False,
                                           save_weights_only=False,
                                           mode='auto',
                                           period=1),
                                       keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=1e-4,
                                                                     restore_best_weights=True)])

        print("Training GAN")
        self.generateAndPlot(x_train_input, fileName="before_gan.png")
        self.trainGAN(x_train_input, epochs=int(train_count / batch_size), batch_size=batch_size)
        self.generateAndPlot(x_train_input, fileName="after_gan.png")

    def trainGAN(self, x_train_input, epochs=1000, batch_size=128):
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            # ---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train_input.shape[0], half_batch)
            imgs_real = x_train_input[idx]
            # Generate a half batch of new images
            imgs_fake = self.generate(n=half_batch)
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            codes = self.kde.sample(batch_size)
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))
            # Train generator
            g_logg_similarity = self.decoder_discriminator.train_on_batch(codes, valid_y)
            # Plot the progress
            if epoch % 50 == 0:
                print("epoch %d [D accuracy: %.2f] [G accuracy: %.2f]" % (epoch, d_loss[1], g_logg_similarity[1]))

    def generate(self, n=10000):
        codes = self.kde.sample(n)
        images = self.decoder.predict(codes)
        return images

    def generateAndPlot(self, x_train_input, n=10, fileName="generated.png"):
        fig = plt.figure(figsize=[20, 20])
        images = self.generate(n * n)
        index = 1
        for image in images:
            image = image.reshape(self.img_shape)
            ax = fig.add_subplot(n, n + 1, index)
            index = index + 1
            ax.set_axis_off()
            ax.imshow(image)
            if index % (n + 1) == 0:
                nearest = findNearest(x_train_input, image)
                ax = fig.add_subplot(n, n + 1, index)
                index = index + 1
                ax.imshow(nearest)
        fig.savefig(fileName)
        plt.show()

    @staticmethod
    def mean_log_likelihood(x_test_input):
        KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_test_input)


if __name__ == '__main__':
    # Load minecraft skins
    data = np.load('images/train_test.npz')
    x_train, x_test = data['arr_0'], data['arr_1']

    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    ann = GAE(img_shape=(64, 64, 4), encoded_dim=128)
    ann.train(x_train, epochs=50)
    # ann.generateAndPlot(x_train)

    encoded_imgs = ann.encoder.predict(x_test)
    decoded_imgs = ann.autoencoder.predict(x_test)

    # TODO: Refactor to use KDE for generation
    # Save the models for later generation
    ann.autoencoder.save('models/autoencoder.mdl')
    ann.decoder.save('models/decoder.mdl')

    n = 10
    m = 3
    plt.figure(figsize=(n, m + .5))
    for i in range(1, n + 1):
        # display original images
        ax = plt.subplot(m, n, i)
        plt.imshow(x_test[i].reshape(64, 64, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display encoded images
        ax = plt.subplot(m, n, n + i)
        plt.imshow(encoded_imgs[i].reshape(8, 4, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(m, n, 2 * n + i)
        plt.imshow(decoded_imgs[i].reshape(64, 64, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    version = "00"
    plt.suptitle('Minecraft Skin GAE')
    plt.savefig('images/results/mnist_gae_{}.jpg'.format(version), bbox_inches='tight')
    plt.show()
