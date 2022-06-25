'''
Author: Aiden Li
Date: 2022-06-25 15:20:42
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 21:41:51
Description: Spectral analysis utilizations
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
import seaborn as sns
sns.set()
density = 512

def fft2(zz):
    freq = torch.fft.fft2(zz)
    amplitude = torch.abs(freq)
    phrase = torch.angle(freq)
    return freq, amplitude, phrase
    
if __name__ == '__main__':
    from viz import viz_3d_surf
    xx = torch.linspace(-1. , 1., density)
    yy = torch.linspace(-1. , 1., density)
    xx, yy = torch.meshgrid(xx, yy)
    
    fn = lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 8 * torch.pi)
    zz = fn(xx, yy)
    
    freq, amplitude, phrase = fft2(zz)
    
    freq_view = torch.log(1 + torch.abs(freq))
    freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 1.0
    freq_view = torch.fft.fftshift(freq_view)
    freq_view = freq_view.detach().cpu().numpy()
    
    viz_3d_surf(freq_view, "freq", "export/debug")
    plt.imshow(freq_view)
    