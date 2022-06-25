'''
Author: Aiden Li
Date: 2022-06-25 15:21:40
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 16:17:05
Description: Low-dim input MLP w\ or w\o positional encoding
'''
import torch.nn as nn

class LowDimMLP(nn.Module):
    def __init__(self, layers=[2, 32, 32, 32, 1], pe=False):
        super(LowDimMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x.squeeze()