'''
Author: Yu Liu, Aiden Li
Date: 2022-06-25 14:28:43
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-06-25 20:56:25
Description: Visualization
'''
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
# from .spectral import fft2

import seaborn as sns
sns.set()

figsize = (16, 16)

def viz_2d(xx, yy, zz, label, path):
    pass

def viz_seq_2d(xx, yy, zz, labels, path):
    pass


def viz_3d(xx, yy, zz, label, path, pdf=False):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    path = os.path.join(path, f"viz_spatial_seq{str(label).zfill(4)}")
    plt.title(label, fontsize=20)
    plt.savefig(f'{path}.png')
    if pdf:
        plt.savefig(f'{path}.pdf')
    plt.show()


def viz_seq_3d(xx, yy, zz, labels, path):
    """Visualize a 3D Spatial sequence to video

    Args:
        xx: N_x  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        yy: N_y  e.g. [-0.9, -0.8, ..., 0.9, 1.0]
        zz: t x N_x x N_y
        label: t
    """
    T = zz.shape[0]
    video_path = os.path.join(path, 'viz_spatial_seq.mp4')
    # writer = cv2.VideoWriter(video_path, -1, T, size)
    for t in range(T):
        viz_3d(xx, yy, zz[t], labels[t], path, pdf = t % 25 == 0)

        # path = f"viz_spatial_seq{str(labels[t]).zfill(4)}"
        # img = cv2.imread(path + '.png')
        # writer.write(img)
    # writer.release()
    