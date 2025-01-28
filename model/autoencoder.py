import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):

    hidden_layer = None

    def __init__(self, layer_dims = None):
        super(Autoencoder, self).__init__()
        if layer_dims == None:
            layer_dims = [200,200,200,100,100,100,50]
        # Encoder
        encoder_layers = []
        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(layer_dims) - 2, 0, -1):
            decoder_layers.append(nn.Linear(layer_dims[i + 1], layer_dims[i]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(layer_dims[1], layer_dims[0]))
        decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        self.hidden_layer = x
        x = self.decoder(x)
        return x