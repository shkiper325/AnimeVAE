import numpy as np
import imageio
import cv2
import n_sphere
import os

import torch

import nets

MODEL_PATH = 'model.img'
GIF_PATH = 'out.gif'
IMG_COUNT = 100
HIDDEN_SIZE = 128
PI = 3.1415

def main():
    print('Loading model')

    to_load = torch.load(MODEL_PATH)
    B = nets.B_net(128)
    B.load_state_dict(to_load['B'])

    gif_frames = []

    for i in range(IMG_COUNT):
        print('Processing image number ', i)

        phi = None
        if i <= IMG_COUNT / 2:
            phi = PI * i * 2 / IMG_COUNT
        else:
            phi = PI - (PI * i * 2 / IMG_COUNT - PI)
        phi_n_minus_1 = PI * i / IMG_COUNT

        N = np.array([phi for i in range(HIDDEN_SIZE)]) + 1e-10
        N[0] = 1
        N[-1] *= phi_n_minus_1

        N = n_sphere.convert_rectangular(N)

        N = np.expand_dims(N, 0)
        N = nets.FloatTensor(N)
        image = B(N).detach().cpu().numpy()[0]

        image = np.clip(image + 1, 0, 2-0.001) / 2 * 256.0
        image = image.astype(np.uint8)
        image = np.moveaxis(image, 0, 2)

        gif_frames.append(image)

        # cv2.imwrite(os.path.join('gif_source', str(i) + '.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # For debug

    imageio.mimwrite(GIF_PATH, gif_frames)

    print('Done!')

if __name__ == '__main__':
    main()