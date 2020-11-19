""" Functions used to vizualize, process, and augmente the Image dataset """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_digit(Xrow):
    """ 
    Plot a single picture
    """
    image = Xrow.reshape(100, 100)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.axis("off")
    plt.show()
    
    
def plot_digits(instances, images_per_row = 3, **options):
    """
    Plots a series of images

    Parameters
    ----------
    instances : ndarray
        Instances in the data one wants to plot
    images_per_row : int, optional
        The default is 3.
    **options : Dict
        Keyword arguments to give plt.imshow.

    -------

    """
    size = 100
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis = 1))
    image = np.concatenate(row_images, axis = 0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    plt.show()