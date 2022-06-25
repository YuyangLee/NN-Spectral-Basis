'''
Author: Aiden Li
Date: 2022-06-25 15:20:42
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 15:44:21
Description: Spectral analysis utilizations
'''
import cv2
import numpy as np
import torch

density = 32

def fft2(xx, yy, zz):
    freq = torch.fft.fft2(zz)
    amplitude = torch.abs(freq)
    phrase = torch.angle(freq)
    
    return freq, amplitude, phrase
    
if __name__ == '__main__':
    xx = torch.linspace(-1. , 1., density)
    yy = torch.linspace(-1. , 1., density)
    xx, yy = torch.meshgrid(xx, yy)
    
    fn = lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) / 2)
    zz = fn(xx, yy)
    
    freq, amplitude, phrase = fft2(None, None, zz)
    
    freq_view = torch.log(1 + torch.abs(freq))
    freq_view = (freq - freq.min(dim=-3)[0]) / (freq.max(dim=-3)[0] - freq.min(dim=-3)[0]) * 255
    
    freq_view = freq_view.detach().cpu().numpy().astype('uint8').copy()
    
    cv2.imwrite("freq_view.jpg", freq_view)
