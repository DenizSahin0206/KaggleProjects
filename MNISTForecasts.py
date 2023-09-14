import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [6.4, 4.8]  # Set the default figure size
plt.rcParams['figure.dpi'] = 100  # Set the default figure DPI

num_plots = 1
num_images = 4

fig, axes = plt.subplot(
    num_plots,
    num_images,
    figsize=(7, 3)
)

for img, label, ax in zip(x[:num_images], y[:num_images], axes):
    ax.set_title(label)
    ax.imshow(img)
    ax.axis('off')
plt.show()