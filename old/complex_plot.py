import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def complex_region(center, r, quality):
    re, im = np.meshgrid(np.linspace(center.real-r, center.real+r, quality),
                         np.linspace(center.imag+r, center.imag-r, quality)*1j)
    return np.add(re, im)

def complex_plot(region, filename, cmap):
    plt.figure(figsize = (8, 8))
    plt.imshow(region, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename,
                dpi=region.shape[0]/6.15882, # This number works, don't change it
                bbox_inches='tight',
                pad_inches=0)

def complex_plot2(region, filename, cmap):
    # Set up colorbars
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cax2 = divider.append_axes("right", size="5%", pad=0.6)

    bar_data = ax.imshow(np.array([[(0, 0, 0)]]), vmin=0, vmax=255, cmap='gray')
    cbar2 = plt.colorbar(bar_data, cax2, ticks=[0, 255])
    cbar2.ax.set_yticklabels(['0', '∞'])

    # Plot the result
    map = ax.imshow(region,
                cmap=cmap,
                vmin=0, vmax=255,
                interpolation='none',
                # extent=[np.min(region.real), np.max(region.real), np.min(region.imag), np.max(region.imag)],
                aspect='equal')

    cbar = plt.colorbar(map, cax, ticks=[0, 64, 128, 192, 255])
    cbar.ax.set_yticklabels(['- π', '- π / 2', '0', 'π / 2', 'π'])

    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    plt.savefig(filename, dpi=region.shape[0]/6.1, bbox_inches='tight', pad_inches=0)