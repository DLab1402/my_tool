import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F


os.makedirs("images", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

class MLP(nn.Module):
    vis = []
    para = {
            "dim":               [],
            "Activate function": []
        }
    def __init__(self,para):
        super(MLP, self).__init__()
        self.para = para
        self.layers = nn.ModuleList()
        self.structure_calculate()

    def structure_calculate(self):
        for dim in self.para["dim"]:
            self.layers.append(nn.Linear(dim[0],dim[1]))
            if self.para["Activate function"] == "ReLU":
                self.layers.append(nn.ReLU())
            elif self.para["Activate function"] == "LeakyReLU":
                self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, noise):
        self.vis.clear()
        out = noise
        for layer in self.layers:
            out = layer(out)
            self.vis.append(out)
        return out


def trainer():
    generator = MLP(para={
            "dim":               [[100,256],[256,512],[512,1024],[1024,784]],
            "Activate function": "ReLU"
        })
    
    discriminator = MLP(para={
            "dim":               [[784,1024],[1024,512],[512,256],[256,1]],
            "Activate function": "LeakyReLU"
        })

    if cuda:
        generator.cuda()
        discriminator.cuda()

    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    for epoch in range(100):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # ----------------
            # Update weights
            # ----------------

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).data[0]

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)