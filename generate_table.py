import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_IMAGE_FOLDER = 'images'

def main():
    print('Enter folder number:')
    folder_name = input()
    folder_path = os.path.join(OUT_IMAGE_FOLDER, folder_name)

    if not os.path.exists(folder_path):
        print('Can\'t find folder')

    filenames = os.listdir(folder_path)

    images = []
    for fn in filenames:
        img = cv2.imread(os.path.join(folder_path, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    print(images[0].shape)

    fig, ax = plt.subplots(4, 4)
    for i in range(len(images)):
        ax[i % 4][i // 4].imshow(images[i].astype(np.float32) / 255)
        ax[i % 4][i // 4].axis('off')

    plt.savefig('table.png')

    plt.show()

if __name__ == '__main__':
    main()