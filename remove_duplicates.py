from PIL import Image
from progress.bar import Bar
import os
import glob
import imagehash

def main():
    file_list = glob.glob("images/*.png")
    image_hashes = set()
    with Bar('Processing', max=len(file_list)) as bar:
        for file in file_list:
            image_hash = str(imagehash.phash(Image.open(file).convert('RGBA')))
            if image_hash in image_hashes:
                #print('Duplicate Image Found', file)
                os.remove(file)
            else:
                image_hashes.add(image_hash)
            bar.next()



if __name__ == "__main__":
    main()