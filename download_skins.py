"""
The Purpose of this script is to download skins into the project image directory.
Further processing will be necessary to determine format of the skin. Ex. v1.8 Standard vs v1.8 Slim vs Legacy version
"""
import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

import requests
from progress.bar import Bar


def download_skin(skin_url, skin_id):
    response = requests.get(skin_url.format(skin_id), stream=True)
    # Check if the image was retrieved successfully
    if response.status_code == 200:
        # Open a local file with wb ( write binary ) permission.
        with open('images/skins/{}.png'.format(skin_id), 'wb') as f:
            f.write(response.content)
            return True
    return False


def main():
    skin_url = "https://www.minecraftskins.com/skin/download/{}"  # [15000000:15206628]
    start = 15148205
    stop = 15206629
    with Bar('Processing', max=stop - start) as bar:
        with ThreadPoolExecutor(max_workers=20) as executor:
            for skin_start in range(start, stop, 100):
                future_skins = {executor.submit(download_skin, skin_url, skin_id): skin_id for skin_id in
                                range(skin_start, skin_start + 100)}
                for future in concurrent.futures.as_completed(future_skins):
                    if future.result():
                        bar.next()

            # for skin_id in range(start, stop):
            #     download_skin(skin_url, skin_id)
            #     bar.next()


if __name__ == "__main__":
    main()
