import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

def dot_line_plot(x, y, y2=None, title='title', xlabel='xlabel', ylabel='ylabel', line1_label='', line2_label='', line_color='b', line_width=2, save_path=None):
    marker_style = 'o'
    marker_size = 8
    plt.figure(figsize=(8, 6))
    if y2 is None:
        plt.plot(x, y, color=line_color, linewidth=line_width, marker=marker_style, markersize=marker_size, label=line1_label)
    else:
        plt.plot(x, y, linewidth=line_width, marker=marker_style, markersize=marker_size, label=line1_label)
        plt.plot(x, y2, linewidth=line_width, marker=marker_style, markersize=marker_size, label=line2_label)
    plt.legend()
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(False)
    if save_path is not None:
        plt.savefig(save_path + 'dot_line.png')
    plt.show()
    plt.close()

def hist_plot(vector, xlabel='', ylabel='', title='', bins=10):
    plt.figure(figsize=(10, 8))
    plt.hist(vector, bins=bins, density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()

def scatter_plot(x, y, xlabel='Hardware scaling to software', ylabel='software'):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()

def plot_heatmap(data, colormap='cool', title='', x_label='', y_label='', show_colorbar=True, vmin=0, vmax=7):
    
    plt.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    if show_colorbar:
        plt.colorbar()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.show()
from scipy.stats import norm

def plot_histogram_with_fit(data, bins=10):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', edgecolor='white')
    mean = np.mean(data)
    std = np.std(data)
    x = np.linspace(min(data), max(data), 100)
    fitted_curve = norm.pdf(x, mean, std)
    plt.plot(x, fitted_curve, 'r', label=f'Fit: mean={mean:.2f}, std={std:.2f}')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution Histogram with Normal Distribution Fit')
    plt.show()
    return [mean, std]

def hist_two_vector(vector1, vector2, bins=10, alpha=0.5, label1='1', label2='2', title=''):
    plt.hist(vector1, bins=bins, alpha=alpha, label=label1, color='blue')
    plt.hist(vector2, bins=bins, alpha=alpha, label=label2, color='green')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()