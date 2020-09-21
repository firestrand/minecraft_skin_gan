"""
The Purpose of this script is to filter the png files in the source directory into the project image directories.
Files that do not appear to match the v18 Standard or Slim format are moved to other.
"""
import glob
import os

from PIL import Image
import numpy as np

def main():
    file_list = glob.glob("images/skins/*.png")
    for file in file_list:
        skin_image = Image.open(file).convert('RGBA')
        skin_array = np.asarray(skin_image)
        head_array = skin_array[8:16, 0:32, :]

        std_count = 0
        if np.std(head_array[:, :, 0:1]) < 10:
            std_count += 1
        if np.std(head_array[:, :, 1:2]) < 10:
            std_count += 1
        if np.std(head_array[:, :, 2:3]) < 10:
            std_count += 1

        if std_count > 1:
            os.rename(file, 'images/other/' + file.split('/')[-1])

    print('done')

if __name__ == "__main__":
    main()
