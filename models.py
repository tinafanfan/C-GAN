import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Number of channels in the training images. For color images this is 3
nc = 3
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self,latent_dim, image_size_1 = 64, image_size_2 = 64, channels = 3, num_classes = 24):
        super(Generator, self).__init__()
        self.image_size_1 = image_size_1
        self.image_size_2 = image_size_2
        self.channels = channels
        self.num_classes = num_classes
        ngf = 64
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution [batch_size, 100+num_classes, 1, 1]
            nn.ConvTranspose2d(100 + num_classes, ngf * 8,  4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.m = nn.Linear(num_classes, num_classes) 
#         self.m1 = nn.ConvTranspose2d(num_classes,  ngf * 8, 4, 1, 0)
#         self.m2 = nn.BatchNorm2d(ngf * 8)


    def forward(self, input: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        conditional = labels.float()
        
        conditional = self.m(conditional)
        conditional = torch.reshape(conditional, (input.shape[0], self.num_classes, 1, 1)) 
#         conditional = F.relu(self.m2(self.m1(conditional)))

        conditional_input = torch.cat((input, conditional), 1) # [batch_size, 124, 1, 1]
        out = self.main(conditional_input)

        return out



class Discriminator(nn.Module):
    def __init__(self, image_size_1 = 64, image_size_2 = 64, channels = 3, num_classes = 24):
        """
        Args:
            image_size_1 (int): The size of the image. x-axis
            image_size_2 (int): The size of the image. y-axis
            channels (int): The channels of the image.
            num_classes (int): Number of classes for dataset.
        """
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.image_size_2 = image_size_2
        self.image_size_1 = image_size_1
        # -----------------
        # DCGAN
        # -----------------
        # ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        ndf = 64 # Size of feature maps in discriminator
        self.main = nn.Sequential(
            
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(channels+1, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)            
            # nn.Sigmoid() # WGAN 1: remove sigmoid
        )
        self.m = nn.Linear(self.num_classes, self.image_size_1*self.image_size_2) # torch.Size([3, 24]) -> torch.Size([3, 76800])
#         self.m1 = nn.ConvTranspose2d(num_classes,  ngf * 8, 4, 1, 0)
#         self.m2 = nn.BatchNorm2d(ngf * 8)


    def forward(self,input: torch.Tensor, labels = None):
        
        input = np.squeeze(input) # torch.Size([3, 1, 3, 240, 320]) -> torch.Size([3, 3, 240, 320])
        conditional = labels.float()
        
        conditional = self.m(conditional)
        conditional = torch.reshape(conditional, (input.shape[0], 1, self.image_size_1, self.image_size_2)) # torch.Size([3, 1, 240, 320])
#         conditional = F.relu(self.m2(self.m1(conditional)))

        conditional_input = torch.cat((input, conditional), 1) # torch.Size([3, 4, 240, 320])
        out = self.main(conditional_input)
        return out
    
    
    
    
# class Generator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(Generator, self).__init__()
#         ngf = 64 # Size of feature maps in discriminator
#         channels = 3
#         num_classes = 24        
        
#         self.deconv1_1 = nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0)
#         self.deconv1_1_bn = nn.BatchNorm2d(ngf * 8)
#         self.deconv1_2 = nn.ConvTranspose2d(num_classes, ngf * 8, 4, 1, 0)
#         self.deconv1_2_bn = nn.BatchNorm2d(ngf * 8)
        
#         self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
#         self.deconv2_bn = nn.BatchNorm2d(ngf * 4)
#         self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
#         self.deconv3_bn = nn.BatchNorm2d(ngf * 2)
#         self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
#         self.deconv4_bn = nn.BatchNorm2d(ngf)
#         self.deconv5 = nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False)
        
        

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input, label):
#         x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
#         y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
#         x = torch.cat([x, y], 1)
#         x = F.relu(self.deconv2_bn(self.deconv2(x)))
#         x = F.relu(self.deconv3_bn(self.deconv3(x)))
#         x = F.relu(self.deconv4_bn(self.deconv4(x)))
#         x = F.tanh(self.deconv5(x))

#         return x

# class Discriminator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(Discriminator, self).__init__()
#         ndf = 64 # Size of feature maps in discriminator
#         channels = 3
#         num_classes = 24
#         self.conv1_1 = nn.Conv2d(channels, ndf, 4, 2, 1, bias=False)
#         self.conv1_2 = nn.Conv2d(num_classes, ndf, 4, 2, 1)
        
#         self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.conv2_bn = nn.BatchNorm2d(ndf * 2)
#         self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.conv3_bn = nn.BatchNorm2d(ndf * 4)
#         self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
#         self.conv4_bn = nn.BatchNorm2d(ndf * 8)
#         self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input, label):
#         label = fill[label]
#         x = F.leaky_relu(self.conv1_1(input), 0.2)
#         y = F.leaky_relu(self.conv1_2(label), 0.2)
#         x = torch.cat([x, y], 1)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
#         # x = F.sigmoid(self.conv5(x))

#         return x