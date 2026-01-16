import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import re

hsv_css = np.array([
    [238, 100, 10],
    [223, 100, 28],
    [238, 100, 10],
    [28, 52, 3],
    [31, 100, 64],
    [28, 52, 3],
    [238, 100, 10],
    [223, 100, 28],
    [238, 100, 10],
    [28, 52, 3],
    [31, 100, 64],
    [28, 52, 3]
]) 

#hsv_css = np.array([
#    [196, 100, 9],
#    [192, 100, 37],
#    [196, 100, 9],
#    [28, 52, 3],
#    [31, 100, 64],
#    [28, 52, 3]
#])  # COL2

#hsv_css = np.array([
#    [196, 93, 5],
#    [195, 69, 32],
#    [195, 71, 100],
#    [195, 69, 32],
#    [196, 93, 5],
#    [28, 52, 3],
#    [28, 86, 100],
#    [28, 52, 3]
#]) # COL1

def css_hsv_to_normalized(css_hsv):
    # Extract numbers using regex

    H_css, S_css, V_css = css_hsv
    
    # Convert to 0-1 range
    H = H_css / 360
    S = S_css / 100
    V = V_css / 100
    
    return [H, S, V]

hsv_colors = []
for el in hsv_css:
    h, s, v = css_hsv_to_normalized(el)
    hsv_colors.append([h,s,v])
hsv_colors = np.array(hsv_colors)


def plot_hsv_colors(hsv_colors):
    rgb_colors = hsv_to_rgb(hsv_colors)  # convert to RGB for matplotlib
    n = len(hsv_colors)
    plt.figure(figsize=(6, 1))
    for i, color in enumerate(rgb_colors):
        plt.fill_between([i, i+1], 0, 1, color=color)
    plt.xlim(0, n)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.show()

plot_hsv_colors(hsv_colors)

def plot_palettes(low_pal, mid_pal, high_pal):
    palettes = [low_pal, mid_pal, high_pal]
    titles = ['LOW_PAL', 'MID_PAL', 'HIGH_PAL']
    fig, axes = plt.subplots(len(palettes), 1, figsize=(12, 4))
    for ax, pal, title in zip(axes, palettes, titles):
        n = len(pal)
        # Convert HSV to RGB
        rgb_pal = hsv_to_rgb(pal)
        # Plot each color as a rectangle
        for i, color in enumerate(rgb_pal):
            ax.axvspan(i, i+1, color=color)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=12, loc='left')
    plt.tight_layout()
    plt.show()

def build_pal(colors, num_output=256):
    orig_pos = np.linspace(0, 1, len(colors))
    interp_pos = np.linspace(0, 1, num_output)

    high_pal = np.zeros((num_output, 3))
    for i in range(3):
        high_pal[:, i] = np.interp(interp_pos, orig_pos, hsv_colors[:, i])
    return high_pal

p = build_pal(hsv_colors)
LOW_PAL = p
MID_PAL = p
HIGH_PAL = p
plot_palettes(p, p, p)