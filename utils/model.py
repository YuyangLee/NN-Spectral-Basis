'''
Author: Aiden Li
Date: 2022-06-25 15:21:40
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 16:17:05
Description: Low-dim input MLP w\ or w\o positional encoding
'''
import torch.nn as nn
from utils.pos_enc import get_embedder


class LowDimMLP(nn.Module):
    def __init__(self, layers=[2, 32, 32, 32, 1], pe=False, multires=10):
        super(LowDimMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.embedder, out_dim = get_embedder(multires)
        self.pe = pe
        layers[1] = out_dim
        if pe:
            del layers[0]
        
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        if self.pe:
            x = self.embedder(x)
        for l in self.layers:
            x = l(x)
        return x.squeeze()

    