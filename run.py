'''
Author: Aiden Li
Date: 2022-06-25 14:32:26
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 22:13:16
Description: Fit a 2D function
'''
import os
from sched import scheduler
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model import LowDimMLP
from utils.spectral import fft2

from utils.viz import viz_seq_3d, viz_3d_surf, viz_spectrum, viz_spectrum_seq
    
    
def init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--density", default=256, type=int)
    parser.add_argument("--init_lr", default=1e-3, type=float)
    parser.add_argument("--viz_intv", default=10, type=int)
    parser.add_argument("--fn", default="block", type=str)
    parser.add_argument("--pe", default=False, type=bool)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    args =  parser.parse_args()
    
    sns.set()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    return args

    
def fit(args, model, func, x_range=[-1., 1.], y_range=[-1., 1.]):
    x_len = x_range[1] - x_range[0]
    y_len = y_range[1] - y_range[0]
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.init_lr)
    
    # To log the process of fitting
    with torch.no_grad():
        _xx = torch.linspace(x_range[0], x_range[1], args.density, device=args.device)
        _yy = torch.linspace(x_range[0], x_range[1], args.density, device=args.device)
        _xx, _yy = torch.meshgrid(_xx, _yy)
        _xy = torch.stack([_xx, _yy], dim=-1)
        _zz = torch.zeros([args.epochs, args.density, args.density], device=args.device)
        gt_zz = func(_xx, _yy)
        
    for epoch in trange(args.epochs):
        with torch.no_grad():
            _zz[epoch] = model(_xy).detach().clone()
            
        xx = torch.rand(args.density, device=args.device) * x_len + x_range[0]
        yy = torch.rand(args.density, device=args.device) * y_len + y_range[0]
        
        xx, yy = torch.meshgrid(xx, yy)
        
        zz_gt = func(xx, yy)
        zz_pred = model(torch.stack([xx, yy], dim=-1))
        
        optimizer.zero_grad()
        loss = F.mse_loss(zz_gt, zz_pred)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            tqdm.write(f"Epoch { epoch } MSE Loss { loss }")
                    
    return [ _xx, _yy, _zz, gt_zz ]

def to_spectral(xx, yy, zz):
    """Convert a spatial signal to it's corresponding spectral signal with 2-D Fourier Transform.

    Args:
        xx: N_x  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        yy: N_y  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        zz: t x N_x x N_y
    """
    pass

if __name__ == '__main__':
    args = init()
    
    basedir = os.path.join("export", "debug", args.fn if not args.pe else f"{args.fn}_pe")
    os.makedirs(basedir + '/img', exist_ok=True)
    os.makedirs(basedir + '/pdf', exist_ok=True)
    
    model = LowDimMLP(layers=[2, 64, 64, 64, 1], pe=args.pe).to(args.device)
    
    fn = {
        "sin_euclid_lo": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 2 * torch.pi),
        "sin_euclid_mi": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 4 * torch.pi),
        "sin_euclid_hi": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 8 * torch.pi),
        "sin_euclid_lo_biased": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 2 * torch.pi) + 0.5,
        "sin_euclid_mi_biased": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 4 * torch.pi) + 0.5,
        "sin_euclid_hi_biased": lambda x, y: torch.sin(torch.sqrt(x**2 + y**2) * 8 * torch.pi) + 0.5,
        "std_normal": lambda x, y: torch.exp(-(x**2 + y**2) / 2) / (2 * torch.pi),
        "half_unit_sphere": lambda x, y: torch.sqrt(1 - x**2 - y**2),
        "sinc": lambda x, y: 0.5 * (torch.sinc((x**2 + y**2) * 4 * torch.pi)), 
        "sc_lo": lambda x, y: 0.5 * (torch.sin(x * 2 * torch.pi) + torch.cos(y * 2 * torch.pi)), 
        "sc_mi": lambda x, y: 0.5 * (torch.sin(x * 2 * torch.pi) + torch.cos(y * 4 * torch.pi)), 
        "sc_hi": lambda x, y: 0.5 * (torch.sin(x * 2 * torch.pi) + torch.cos(y * 8 * torch.pi)), 
        "block": lambda x, y: (torch.heaviside(x - 0.5, torch.ones_like(x) * 0.5) - torch.heaviside(x + 0.5, torch.ones_like(x) * 0.5)) \
                            * (torch.heaviside(y - 0.5, torch.ones_like(x) * 0.5) - torch.heaviside(y + 0.5, torch.ones_like(x) * 0.5))
    }[args.fn]
    
    xx, yy, zz, gt_zz = fit(args, model, fn, x_range=[-1., 1.], y_range=[-1., 1.])
    
    with torch.no_grad():
        freq, amplitude, phrase = fft2(zz)
        freq_gt, amplitude_gt, phrase_gt = fft2(gt_zz)
        
        xx = xx.detach().cpu().numpy()
        yy = yy.detach().cpu().numpy()
        
        freq = freq[0::args.viz_intv]
        zz = zz.detach().cpu().numpy()[0::args.viz_intv]
        gt_zz = gt_zz.detach().cpu().numpy()
        
        ts = np.arange(0, zz.shape[0]) * args.viz_intv
        titles = [ f"Iter { str(t).zfill(4) }" for t in ts]
        
        viz_seq_3d(xx, yy, zz, titles, [ f"spatial_{ str(t).zfill(4) }" for t in ts], path=basedir)
        viz_3d_surf(xx, yy, gt_zz, "", f"gt_{args.fn}", basedir, pdf=True)
        
        viz_spectrum_seq(freq, titles, [ f"spectrum_{ str(t).zfill(4) }" for t in ts], path=basedir)
        viz_spectrum(freq_gt, "Amplitude", f"gt_spectrum_{args.fn}", basedir, pdf=True)
        