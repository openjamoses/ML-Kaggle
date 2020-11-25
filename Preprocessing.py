""" Functions used to vizualize, process, and augmente the Image dataset """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_digit(Xrow):
    """ 
    Plot a single picture
    """
    size = int(np.sqrt(Xrow.shape[1]))
    image = Xrow.reshape(size, size)
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
    size = int(np.sqrt(instances.shape[1]))
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
    
    
    
def crop_images(instances, reduce = 1):
    
    nsize = int( np.sqrt(instances.shape[1]) )
    reduce_size = nsize - 2 * reduce
    cropped = np.zeros((instances.shape[0], reduce_size ** 2) , dtype = 'u1')
    for i in range(reduce_size):
        cropped[:, reduce_size*i: reduce_size*(i+1)] = \
                             instances[:, (i+1)*nsize+reduce:(i+2)*nsize-reduce]
    return cropped



def reverse_images(instances):
    """
    Take the horizontal symmetry of each image

    Parameters
    ----------
    instances : ndarray
        Images to be reversed horizontally, one image per row.

    Returns
    -------
    flipped : ndarray
        Flipped images

    """
    flipped = instances.copy()
    nsize = int( np.sqrt(instances.shape[1]) )
    for i in range(nsize):
        flipped[:, nsize*i:nsize*(i+1)] = \
                             np.flip(flipped[:, nsize*i:nsize*(i+1)], axis = 1)
    return flipped



def translate_images(instances, all_delta_i, all_delta_j):
    
    nimages = instances.shape[0]
    nsize = int( np.sqrt(instances.shape[1]) )
    translated = np.zeros( (len(all_delta_i) * nimages, 
                            instances.shape[1]) , dtype = np.dtype('u1'))
    nsize = int( np.sqrt(instances.shape[1]) )

    z = lambda i, j: i * nsize + j
    
    for t in range( len(all_delta_i) ):
        delta_i = all_delta_i[ t ]
        delta_j = all_delta_j[ t ]
        for pixel_idx in range(instances.shape[1]):
            row = int(pixel_idx / nsize)
            col = pixel_idx % nsize
            translated[t*nimages:(t+1)*nimages, pixel_idx] = \
                  instances[:, z(row - delta_i % nsize, col - delta_j % nsize)]
        
    return translated
