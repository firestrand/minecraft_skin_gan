"""Optional spatial models and reconstruction objectives for controlled ablations.

The convolutional variant treats the PNG atlas as a plane: adjacent pixels across
body-part boundaries are not guaranteed to represent adjacent surface geometry.
It is a comparison candidate, not a demonstrated improvement over the dense GAE.
"""

from dataclasses import dataclass
from typing import Any

from generate_skin import keras


@dataclass
class GenerativeModels:
    encoder: Any
    decoder: Any
    discriminator: Any
    autoencoder: Any
    decoder_discriminator: Any


@keras.saving.register_keras_serializable(package="minecraft_skin_gan")
def visible_rgba_loss(y_true: Any, y_pred: Any) -> Any:
    """Target-alpha-weighted visible RGB MSE plus independent alpha MSE.

    RGB uses sum(alpha * squared RGB error) / (3 * sum(alpha)); zero visible
    pixels contribute zero RGB error. Alpha MSE has equal scalar weight. This
    deliberately differs from legacy all-channel MSE and is opt-in.
    """
    ops = keras.ops
    alpha = y_true[..., 3:4]
    rgb_error = ops.square(y_true[..., :3] - y_pred[..., :3])
    rgb_loss = ops.sum(alpha * rgb_error) / ops.maximum(3 * ops.sum(alpha), 1e-7)
    alpha_loss = ops.mean(ops.square(alpha - y_pred[..., 3:4]))
    return rgb_loss + alpha_loss


def create_conv_models(*, encoded_dim: int = 128) -> GenerativeModels:
    """Build a compact 64x64 RGBA autoencoder and discriminator from native layers."""
    if type(encoded_dim) is not int or encoded_dim <= 0:
        raise ValueError("Encoded dimension must be a positive integer")
    layers = keras.layers
    inputs = keras.Input(shape=(64, 64, 4))
    features = inputs
    for filters in (16, 32, 64):
        features = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(features)
    codes = layers.Dense(encoded_dim)(layers.Flatten()(features))
    encoder = keras.Model(inputs, codes, name="conv_encoder_v1")

    latent = keras.Input(shape=(encoded_dim,))
    decoded = layers.Dense(8 * 8 * 64, activation="relu")(latent)
    decoded = layers.Reshape((8, 8, 64))(decoded)
    for filters in (64, 32, 16):
        decoded = layers.UpSampling2D(size=2, interpolation="nearest")(decoded)
        decoded = layers.Conv2D(filters, 3, padding="same", activation="relu")(decoded)
    pixels = layers.Conv2D(4, 1, activation="sigmoid")(decoded)
    decoder = keras.Model(latent, pixels, name="conv_decoder_v1")

    images = keras.Input(shape=(64, 64, 4))
    features = images
    for filters in (16, 32, 64):
        features = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(features)
    features = layers.GlobalAveragePooling2D()(features)
    score = layers.Dense(1, activation="sigmoid")(features)
    discriminator = keras.Model(images, score, name="conv_discriminator_v1")
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)), name="conv_autoencoder_v1")
    adversary = keras.Model(latent, discriminator(decoder(latent)), name="conv_adversary_v1")
    return GenerativeModels(encoder, decoder, discriminator, autoencoder, adversary)
