import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Model
import keras
import numpy as np
from PIL import Image

# mask_image = Image.open('images/masks/v18_standard_mask.png').convert('RGBA')
# mask_array = np.asarray(mask_image)
#
# images = np.load('images/results/gen_00.npy')
# index = 1
# for image in images:
#     image = np.clip(image.reshape((64,64,4)) ,0., 1.) * 255
#     image_int = image.round().astype(np.uint8)
#     gen_image = Image.fromarray(image_int)
#     gen_image.save("images/results/{}.png".format(index))
#     index += 1

# TODO: Refactor to use KDE for generation

# load the models for later generation
decoder = keras.models.load_model('models/decoder.mdl')

rand_arr = np.random.rand(100,128)
images = decoder.predict(rand_arr)

index = 0
for image in images:
    image = np.clip(image.reshape((64,64,4)) ,0., 1.) * 255
    image_int = image.round().astype(np.uint8)
    gen_image = Image.fromarray(image_int)
    gen_image.save("images/results/gae_{}.png".format(index))
    index += 1