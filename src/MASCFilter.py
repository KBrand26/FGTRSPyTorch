from Filter import Filter
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.stats import sigma_clip
from skimage.morphology import dilation, square
from astropy.io import fits

def find_connected_component(mask, row, col):
    """
    Makes use of morphological dilations to construct and extract the connected component
    at the given row and column in the mask.

    Args:
        mask (ndarray): The mask within which to find the connected component.
        row (int): The row of the starting pixel for the connected component.
        col (int): The column of the starting pixel for the connected component.

    Returns:
        tuple: A tuple containing the extracted connected component, as well as its size.
    """
    comp = np.zeros_like(mask)
    comp[row, col] = 1.0

    se = square(3)

    prev_comp = np.copy(comp)
    comp = np.multiply(dilation(comp, se), mask)

    while (comp - prev_comp).any():
        prev_comp = np.copy(comp)
        comp = np.multiply(dilation(comp, se), mask)

    return comp, np.sum(comp)

def remove_clipping_artefacts(mask, thresh=10):
    """Removes small artefacts from a given mask that are unlikely to belong to the galaxy.

    Args:
        mask (ndarray): A mask that was calculated during sigma clipping.
        thresh (int, optional): The threshold to use when determining whether a component is too small. 
            Defaults to 10.

    Returns:
        ndarray: Returns the mask after all small clipping artefacts have been removed.
    """
    new_mask = np.zeros_like(mask)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                comp, size = find_connected_component(mask, r, c)
                mask -= comp
                if size > thresh:
                    new_mask += comp
    return new_mask

class MASCFilter(Filter):
    """
    Extends the Filter abstract class to implement a relatively straightforward filter that only conducts
    sigma clipping and basic artefact removal
    """    
    def filter(self, img):
        """Makes use of sigma clipping and morphological operators to extract the galaxies from the given iamges.

        Args:
            img (ndarray): The image on which to apply filtering.

        Returns:
            ndarray: A mask representing the locations of the galaxy pixels after filtering.
        """
        # Extract the pixels with extreme values
        mask = sigma_clip(img, maxiters=3).mask
        # Remove as many small clipping artefacts as possible
        mask = remove_clipping_artefacts(mask.astype("float32"))
        
        return mask

    def additional_filter(self, img):
        """This method is only added for completeness, it is not necessary when conducting MASC filtering

        Args:
            img (ndarray): The image to filter.

        Returns:
            ndarray: A mask representing the locations of the galaxy pixels after filtering.
        """
        return img