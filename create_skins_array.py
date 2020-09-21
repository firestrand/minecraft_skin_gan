import glob

from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from progress.bar import Bar

file_list = glob.glob("images/skins/*.png")

skin_arrays = []
with Bar('Processing', max=len(file_list)) as bar:
    for file in file_list:
        skin_image = Image.open(file).convert('RGBA')
        skin_arrays.append(np.asarray(skin_image))
        bar.next()

stacked = np.stack(skin_arrays)
# Load minecraft skins
x_train, x_test = train_test_split(stacked, test_size=0.2, random_state=1976)

np.savez('images/train_test.npz', x_train, x_test)