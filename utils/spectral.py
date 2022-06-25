'''
Author: Aiden Li
Date: 2022-06-25 15:20:42
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 20:55:33
Description: Spectral analysis utilizations
'''
import numpy as np
import torch
import torch.fft
from viz import viz_3d
import seaborn as sns
sns.set()
density = 1024

def fft2(xx, yy, zz):
    freq = torch.fft.fft2(zz)
    amplitude = torch.abs(freq)
    phrase = torch.angle(freq)
    
    return freq, amplitude, phrase
    
if __name__ == '__main__':
    xx = torch.linspace(-1. , 1., density)
    yy = torch.linspace(-1. , 1., density)
    xx, yy = torch.meshgrid(xx, yy)
    
    fn = lambda x, y: 10 * torch.sinc(torch.sqrt(x**2 + y**2) / 2)
    zz = fn(xx, yy)
    
    freq, amplitude, phrase = fft2(None, None, zz)
    
    freq_view = torch.log(1 + torch.abs(freq))
    freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 1.0
    freq_view = torch.fft.fftshift(freq_view)
    freq_view = freq_view.detach().cpu().numpy()
    
    viz_3d(xx, yy, freq_view, "freq", "export/debug")
    