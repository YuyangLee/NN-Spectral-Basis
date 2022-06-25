'''
Author: Yu Liu, Aiden Li
Date: 2022-06-25 14:28:43
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 22:11:06
Description: Visualization
'''
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import seaborn as sns
sns.set()

figsize = (16, 16)

def viz_spectrum(freq, title, label, path, pdf=False):
    freq_view = torch.log(1 + torch.abs(freq))
    freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 1.0
    freq_view = torch.fft.fftshift(freq_view)
    freq_view = freq_view.detach().cpu().numpy()
    
    # viz_3d_surf(np.arange(freq_view.shape[0]), np.arange(freq_view.shape[0]), freq_view, title, f"{label}_3d_", path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('auto')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(freq_view)
    plt.title(title, fontsize=32)
    path = os.path.join(path, f"viz_{label}")
    plt.savefig(f'{path}.png')
    if pdf:
        plt.savefig(f'{path}.pdf')
    
def viz_spectrum_seq(freqs, titles, labels, path, pdf=False):
    for t in range(freqs.shape[0]):
        viz_spectrum(freqs[t], titles[t], labels[t], path, pdf)

def viz_3d_surf(xx, yy, zz, title, label, path, pdf=False):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-2.0, 2.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    path = os.path.join(path, f"viz_{label}")
    if pdf:
        plt.savefig(f'{path}.pdf')
    plt.title(title, fontsize=32)
    plt.savefig(f'{path}.png')
    # plt.show()


def viz_seq_3d(xx, yy, zz, titles, labels, path):
    """Visualize a 3D Spatial sequence to video

    Args:
        xx: N_x  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        yy: N_y  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        zz: t x N_x x N_y
        label: t
    """
    for t in range(zz.shape[0]):
        viz_3d_surf(xx, yy, zz[t], titles[t], labels[t], path, pdf = t % 25 == 0)
