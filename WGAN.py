import argparse
import os
import numpy as np
import math
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

from models import Generator, Discriminator
from utils import ICLEVRLoader
from utils import get_iCLEVR_data
from evaluator import evaluation_model

import datetime
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=4000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=27, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    args = parser.parse_args()

    tz = datetime.timezone(datetime.timedelta(hours=+8))

    folder_name = 'output'
    if os.path.isdir('output'):
        print('output 和 output_train 資料夾已存在')
    else:
        os.makedirs('output') # 開資料夾
        os.makedirs("output_train", exist_ok=True)
        print('建立 output 和 output_train 資料夾')
    
    
    img_shape = (args.channels, args.img_size, args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Configure data loader
    dataset = ICLEVRLoader('')
    
    # test data
    test_z = torch.randn(32, args.latent_dim, 1, 1).to(device)
    _,test_labels = get_iCLEVR_data(root_folder = '', mode = 'test')
    test_labels = torch.FloatTensor(test_labels).to(device)
    
    eval_model = evaluation_model()

                           
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    # ----------
    #  load or not
    # ----------
    
    
    if os.path.exists('g_weight.pth'):
        checkpoint = torch.load('g_weight.pth')
        generator.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        
        checkpoint = torch.load('d_weight.pth')
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        start_epoch = checkpoint['epoch']
        print('加載 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('無保存模型，將開始訓練！')
    
    
    # ----------
    #  Training
    # ----------

    batches_done = 0
    
    best_score = 0
    time_start = datetime.datetime.now(tz)
    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        print(epoch)
        generator.train()
        for i, (real_imgs, real_labels) in enumerate(dataloader):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(real_imgs.shape[0], args.latent_dim, 1, 1).to(device)
            fake_labels = real_labels

            # Generate a batch of images
            fake_imgs = generator(z, real_labels).detach()
            
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs, real_labels)) + torch.mean(discriminator(fake_imgs, fake_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
#             for p in discriminator.parameters():
#                 p.data.clamp_(-args.clip_value, args.clip_value)

            # Train the generator every n_critic iterations
            if i % args.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z, real_labels)
                
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs, fake_labels))

                loss_G.backward()
                optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        save_image(gen_imgs.data[:25], "output_train/%d.png" % epoch, nrow=5, normalize=True)
        save_image(real_imgs.data[:25], "output_train/real_%d.png" % epoch, nrow=5, normalize=True)

        generator.eval()
        with torch.no_grad():
            test_imgs = generator(test_z, test_labels)

        score = eval_model.eval(test_imgs, test_labels)
        print(f'\nScore: {score:.2f}')
        if score > best_score:
#             torch.save(generator.state_dict(), 'g_weight.pth')
#             torch.save(discriminator.state_dict(), 'd_weight.pth')
            
            state_g = {'generator': generator.state_dict(), 'optimizer_G': optimizer_G.state_dict(), 'epoch': epoch}
            state_d = {'discriminator': discriminator.state_dict(), 'optimizer_D': optimizer_D.state_dict(), 'epoch': epoch}
            torch.save(state_g, 'g_weight.pth')
            torch.save(state_d, 'd_weight.pth')
            
            grid = make_grid(test_imgs.cpu(), normalize=True).permute(1,2,0).numpy()
            plt.imshow(grid)
            name = 'generation_' + str(epoch) + '_' + str(round(score,3)) + '.png'
            plt.savefig(os.path.join(folder_name, name), dpi=100, bbox_inches='tight')
            plt.close()
            
            best_score = score

    time_difference = datetime.datetime.now(tz) - time_start
    print(time_difference)
if __name__ == "__main__":
    main()
