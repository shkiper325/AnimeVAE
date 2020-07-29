import os
import random
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nets

IMAGE_SIDE = 60
DATA_PATH = 'data'
OUT_IMAGE_FOLDER = 'images'

def data_iter(path, batch_size):
    filenames = os.listdir(DATA_PATH)

    ret = np.empty((batch_size, 3, IMAGE_SIDE, IMAGE_SIDE))
    ret_iter = 0    
    while True:
        random.shuffle(filenames)
        filenames_iter = 0

        while True:
            if filenames_iter >= len(filenames):
                break

            if ret_iter == batch_size:
                yield ret
                ret_iter = 0

            img = cv2.imread(os.path.join(DATA_PATH, filenames[filenames_iter]))
            if img is None:
                continue
            filenames_iter += 1

            img = cv2.resize(img, dsize=(IMAGE_SIDE, IMAGE_SIDE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = np.moveaxis(img, 2, 0)
            img = img.astype(np.float32)
            
            img = ((img / 255.) - 0.5) * 2

            ret[ret_iter] = img
            ret_iter += 1

def generate_standart_normal(n):
    I = np.identity(n)
    b = np.zeros((n,))
    return np.random.multivariate_normal(b, I)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--out-model-path', type=str, default='model.img')
    args = parser.parse_args()

    EPOCH_COUNT = 500
    
    BATCH_SIZE = 16
    EPOCH_LEN = len(os.listdir(DATA_PATH)) // BATCH_SIZE

    LR = 0.001
    
    HIDDEN_SIZE = 128

    SIGMA_SQ = 0.0001

    batch_flow = data_iter(DATA_PATH, BATCH_SIZE)

    A = nets.A_net(128)
    B = nets.B_net(128)

    nets.init_weights(A)
    nets.init_weights(B)

    A_optim = optim.Adam(A.parameters(), lr=LR)
    B_optim = optim.Adam(B.parameters(), lr=LR)

    kl_losses = []
    main_losses = []

    start_epoch = 0
    
    if args.load_model:
        print('Loading model')

        to_load = torch.load(args.load_model)

        A.load_state_dict(to_load['A'])
        B.load_state_dict(to_load['B'])
        A_optim.load_state_dict(to_load['A_optim'])
        B_optim.load_state_dict(to_load['B_optim'])

        kl_losses = to_load['kl_losses']
        main_losses = to_load['main_losses']
        start_epoch = to_load['epoch_num']

    PLOT_FREQ = 10

    IMG_GEN_FREQ = 50
    IMG_GEN_COUNT = 16

    plot_iter = 0
    img_gen_iter = 0    
    for epoch_num in range(start_epoch, EPOCH_COUNT):
        print('Epoch ', epoch_num, ' started!')

        print('Saving model')
        to_save = {
            'A' : A.state_dict(),
            'B' : B.state_dict(),
            'A_optim' : A_optim.state_dict(),
            'B_optim' : B_optim.state_dict(),
            'kl_losses' : kl_losses,
            'main_losses' : main_losses,
            'epoch_num' : epoch_num
        }
        torch.save(to_save, args.out_model_path)
        
        print('Learning!')
        
        batch_number = 0
        for batch in batch_flow:
            A_optim.zero_grad()
            B_optim.zero_grad()

            plot_iter += 1
            img_gen_iter += 1

            A_in = nets.FloatTensor(batch)
            X, mu = A(A_in)

            kl_loss_a = torch.sum(X, dim=1)
            kl_loss_b = torch.sum(mu * mu, dim=1)
            kl_loss_c = -HIDDEN_SIZE
            kl_loss_d = -(torch.sum(torch.log(X + 1e-10)))

            kl_loss = 0.5 * (kl_loss_a + kl_loss_b + kl_loss_c + kl_loss_d)
            
            N = np.array([generate_standart_normal(HIDDEN_SIZE) for i in range(BATCH_SIZE)])
            N = nets.FloatTensor(N)

            B_in = mu + X * N
            out_image = B(B_in)

            main_loss = torch.sum((nets.FloatTensor(batch) - out_image) ** 2, dim=(1, 2, 3))
            main_loss = main_loss / SIGMA_SQ / (IMAGE_SIDE * IMAGE_SIDE)

            full_loss = torch.sum(main_loss + kl_loss) / BATCH_SIZE
            full_loss.backward()
            A_optim.step()
            B_optim.step()

            if plot_iter % PLOT_FREQ == 0:
                print('Plotting...')

                np_kl_loss = np.mean(kl_loss.detach().cpu().numpy())
                np_main_loss = np.mean(main_loss.detach().cpu().numpy())
                kl_losses.append(np_kl_loss)
                main_losses.append(np_main_loss)

                plt.clf()
                plt.plot(kl_losses)
                plt.savefig('kl_losses.png')

                plt.clf()
                plt.plot(main_losses)
                plt.savefig('main_losses.png')

                plt.clf()
                plt.plot(np.array(kl_losses) + np.array(main_losses))
                plt.savefig('full_losses.png')

                print('Last KL loss: ', np_kl_loss)
                print('Last main loss: ', np_main_loss)

            if img_gen_iter % IMG_GEN_FREQ == 0:
                print('Writing ', img_gen_iter // IMG_GEN_FREQ, ' image set')

                N = np.array([generate_standart_normal(HIDDEN_SIZE) for i in range(IMG_GEN_COUNT)])
                N = nets.FloatTensor(N)
                images = out_image = B(N).detach().cpu().numpy() #???

                path = os.path.join(OUT_IMAGE_FOLDER, str(img_gen_iter // IMG_GEN_FREQ))

                for i in range(IMG_GEN_COUNT):
                    if not os.path.exists(path):
                        os.mkdir(path)

                    img = np.clip(images[i] + 1, 0, 2-0.001) / 2 * 256.0
                    img = img.astype(np.uint8)
                    img = np.moveaxis(img, 0, 2)

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(path, str(i) + '.png'), img)


            batch_number += 1
            if batch_number == EPOCH_LEN:
                batch_number = 0
                break
            
        


if __name__ == '__main__':
    main()