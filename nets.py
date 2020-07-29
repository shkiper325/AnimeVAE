import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

INIT_STD = 0.02

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

if USE_CUDA:
    print('Using CUDA')
else:
    print('Using CPU')

class A_net(nn.Module):
    def __init__(self, hidden_size=128):
        super(A_net, self).__init__()

        self.hidden_size = hidden_size
        
        self.main = nn.Sequential(
            #3 x 60 x 60
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            #32 x 16 x 16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #64 x 16 x 16
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            #96 x 8 x 8
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #128 x 4 x 4
            nn.Flatten()
        )
        
        self.X_layer = nn.Linear(in_features = 128*4*4, out_features=hidden_size)
        self.mu_layer = nn.Linear(in_features = 128*4*4, out_features=hidden_size)
        
        if USE_CUDA:
            self.cuda()
        
    def forward(self, x):
        x = self.main(x)
        
        X = F.relu(self.X_layer(x))
        mu = self.mu_layer(x)
        
        return X, mu


class B_net(nn.Module):
    def __init__(self, hidden_size=128):
        super(B_net, self).__init__()

        self.hidden_size = hidden_size
        
        self.main = nn.Sequential(
            #128 x 1 x 1
            nn.ConvTranspose2d(in_channels=hidden_size, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            #512 x 4 x 4
            nn.Upsample(size=(8, 8), mode='nearest'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #256 x 8 x 8
            nn.Upsample(size=(16, 16), mode='nearest'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #128 x 16 x 16
            nn.Upsample(size=(32, 32), mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #64 x 32 x 32
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #64 x 32 x 32
            nn.Upsample(size=(60, 60), mode='nearest'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #64 x 60 x 60
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
            #3 x 60 x 60
        )
        
        if USE_CUDA:
            self.cuda()
            
    def forward(self, x):
        x = torch.reshape(x, (-1, self.hidden_size, 1, 1))
        
        x = self.main(x)
        
        return x

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.normal_(m.weight.data, mean=0, std=INIT_STD)
        torch.nn.init.normal_(m.bias.data, mean=0, std=INIT_STD)
    elif type(m) in [nn.BatchNorm2d, nn.LeakyReLU, nn.ReLU, nn.Sequential, nn.Tanh]:
        return
    else:
        print('Couldn\'t init wieghts of layer with type:', type(m))