from skimage import data, io, color
import matplotlib.pyplot as plt
from pylab import rcParams

def plot_color_gradients(gray_img, cmap_list):
    nrows = len(cmap_list)
    fig, axs = plt.subplots(nrows=nrows, figsize=(5, 15))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0,
    wspace=0)
    for ax, name in zip(axs, cmap_list):
        ax.imshow(gray_img, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-.01, .5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)
    for ax in axs:
        ax.set_axis_off()

img = data.coffee()
io.imshow(img)
plt.axis('off')

gray_img = color.rgb2gray(img)
plt.imshow(gray_img)
plt.axis('off')

cmap_list = ['spring', 'gist_earth', 'magma', 'Dark2', 'gnuplot2', 'CMRmap']
plot_color_gradients(gray_img, cmap_list)

plt.show()