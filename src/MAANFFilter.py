from Filter import Filter
import numpy as np
from skimage.morphology import dilation, square
from scipy.ndimage import binary_fill_holes

def basic_thresh(image, s):
    '''
    Applies the basic thresholding algorithm to the given image.
    
    Parameters:
    -----------
    image : ndarray
        The image that needs to be thresholded.
    s : int
        Number of histogram bins that had more than 100 pixels.
    
    Returns:
    --------
    ndarray
        The thresholded image.
    '''
    
    if s <= 14:
        # Noise is almost non-existant, use a static threshold
        return np.where(image > 0.1, 1, 0)
    # Identify relevant quantiles
    q985 = np.quantile(image, 0.985)
    q98 = np.quantile(image, 0.98)
    
    # Threshold at high quantile to ensure only galaxy pixels are extracted
    first = np.where(image > q985, 1, 0)
    
    # Remove any small thresholding artefacts from background
    first = remove_small_components(np.copy(first), 10)
    
    # `Grow` the extracted pixels with dilations to include connected pixels that are above the secondary threshold
    exten = np.where(image >= q98, 1, 0)
    se = square(3)
    
    prev = np.copy(first)
    cur = np.copy(first)
    # Dilation adds surrounding pixels, multiplication removes pixels that are not larger than 98th quantile.
    cur = np.multiply(dilation(cur, se), exten)
    while (cur - prev).any():
        prev = np.copy(cur)
        cur = np.multiply(dilation(cur, se), exten)

    return cur

def remove_small_components(img, thresh=110):
    '''
    This function finds connected components in the given image and removes any
    components that are smaller than the given threshold.
    
    Parameters:
    -----------
    img : ndarray
        The image from which to remove the small components
        
    thresh : int
        The threshold to use when determining whether a component is too small.
        
    Returns:
    --------
    ndarray
        The image after all of the small components were removed
    '''
    total = np.sum(img)
    if total < thresh:
        # If the total count of pixels in the image is smaller than the threshold size, no thresholding is necessary.
        return img
    
    new_img = np.zeros_like(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] == 1:
                comp, size = find_connected_component(img, r, c)
                img -= comp
                if size > thresh:
                    new_img += comp
    return new_img

def find_connected_component(img, r, c):
    '''
    Finds all the pixels that belong to a component in an image, given the coordinates of a pixel
    that is known to belong to that components
    
    Parameters:
    -----------
    img : ndarray
        The image within which to look for the component
    r : int
        The row corresponding to the pixel from the component.
    c : int
        The column corresponding to the pixel from the component.
        
    Returns:
    --------
    tuple
        A tuple that contains the extracted component, as well as its size.
    '''
    comp = np.zeros_like(img)
    # Starting pixel for the connected component
    comp[r, c] = 1.0

    se = square(3)

    # Repeatedly `grow' the connected component to include surrounding pixels
    prev_comp = np.copy(comp)
    # Remove background and noise pixels by multiplying with the thresholded image.
    comp = np.multiply(dilation(comp, se), img)

    while (comp - prev_comp).any():
        prev_comp = np.copy(comp)
        comp = np.multiply(dilation(comp, se), img)

    return comp, np.sum(comp)

def eligible_for_basic_thresh(img, bins):
    '''
    This function investigates the histogram corresponding to an image to determine which thresholding approach to apply
    
    Parameters:
    -----------
    img : ndarray
        The image that is being tresholded.
    bins : ndarray
        The histogram that corresponds to the given image.
        
    Returns:
    --------
    tuple
        Tuple indicating whether basic thresholding should be used, the number of bins with more than 100 pixels and
        the standard deviation of the image.
    '''
    # Count how many bins have more than 100 pixels
    counts = np.array([len(img[bins == i]) for i in range(256)])
    s = np.sum(counts > 100)
    
    std = np.std(img)
    return s < 17 or std < 0.035, s, std

def thresh_image(img):
    '''
    Thresholds image to remove noise and background pixels.
    
    Parameters:
    -----------
    img : ndarray
        The image that needs to be thresholded.
        
    Returns:
    --------
    tuple
        Tuple containing the thresholded image, as well as a flag that indicates whether basic thresholding was used. 
        
    '''
    # Generate histogram of image
    bins = np.linspace(0, 1, 256)
    binned = np.digitize(img, bins)
    
    # Determine which thresholding approach to use 
    elig, s, std = eligible_for_basic_thresh(img, binned)
    if elig:
        # Perform basic thresholding
        threshed = basic_thresh(img, s)
        threshed = remove_small_components(np.copy(threshed), 5)
        return threshed, True
    else:
        # There is a lot of noise in the image. Apply more discriminate thresholding.
        val_dist = find_num_vals(img)
        
        # Extract quantiles
        q98 = np.quantile(val_dist, 0.985)
        q96 = np.quantile(val_dist, 0.96)
        
        # Pixels along the border of the galaxy will have a wider variety of pixel values
        # from galaxy pixels, noise pixels and background pixels. Thus we first extract pixels that
        # have a large number of pixel values in their neighbourhood.
        first = np.where(val_dist >= q98, 1, 0)
        # Remove artefacts
        first = remove_small_components(np.copy(first), 10)
        
        # Use dilations to `grow' the extracted pixels to include connected pixels that still have a large range of values
        # in their neighbourhood.
        exten = np.where(val_dist >= q96, 1, 0)
        se = square(3)
        
        prev = np.copy(first)
        cur = np.copy(first)
        cur = np.multiply(dilation(cur, se), exten)
        while (cur - prev).any():
            prev = np.copy(cur)
            cur = np.multiply(dilation(cur, se), exten)
            
        # Remove artefacts
        cleaned = remove_small_components(np.copy(cur))
        # Fill holes that are surrounded by extracted pixels. This ensures that the galaxy pixels in the center
        # of the galaxy are also extracted.
        threshed = binary_fill_holes(cleaned)
        
        return threshed, False

def find_num_vals(img):
    '''
    This function creates a new matrix where each element represents the number of unique pixel values in the
    9x9 neighbourhood of the corresponding pixel in the given image.
    
    Parameters:
    -----------
    img : ndarray
        The image to use to create the new matrix.
        
    Returns:
    --------
    ndarray
        The matrix representing the unique value counts in the neighbourhoods of the given image.
    '''
    vals_dist = np.zeros_like(img)
    size = 9
    step = size//2
    for r in range(step, img.shape[0]-step):
        for c in range(step, img.shape[1]-step):
            # Extract neighbourhood
            neigh = img[r-step:r+step+1, c-step:c+step+1]
            
            # Create local neighbourhood histogram
            bins = np.linspace(0, 1, 256)
            binned = np.digitize(neigh, bins)
            
            # Count how many bins are represented in the neighbourhood
            vals = len(np.unique(binned))
            vals_dist[r, c] = vals
    return vals_dist

def prep_bent_class(img, threshed):
    '''
    This function is used to finetune thresholding results in the presence of significant noise.
    This is necessary to ensure that the bent feature can be extracted accurately.

    Parameters:
    -----------
    img : ndarray
        The original image.
    threshed : ndarray
        The corresponding thresholded image that needs to be cleaned.

    Returns:
    --------
    ndarray
        Thresholded image after finetuning.
    '''
    # Replace pixels with their original intensities.
    processed = np.where(threshed, img, 0)

    # Determine how many standard deviations each pixel is from the mean 
    zscores = calc_zscores(processed)

    # Keep pixels that have a zscore larger than the median zscore
    return thresh_median(zscores, binary=True)

def thresh_median(img, binary=False):
    '''
    This function thresholds the given image and only keeps pixel intensities that lie above the median intensity

    Parameters:
    -----------
    img : ndarray
        The image that should be thresholded
    binary : boolean
        A flag that indicates whether the output should be a binary image.

    Returns:
    --------
    ndarray
        The image after extracting pixels larger than themedian
    '''
    min_val = img.min()
    tmp = np.where(img == min_val, np.nan, img)
    q50 = np.nanquantile(tmp, 0.5)
    if binary:
        return np.where(img < q50, 0, 1)
    else:
        return np.where(img < q50, min_val, img)

def calc_zscores(threshed_img):
    '''
    This function is used to calculate the number of standard deviations that each pixel is from the
    mean pixel intensity.

    Parameters:
    -----------
    threshed_img : ndarray
        The thresholded image for which to calculate z-scores.

    Returns:
    --------
    ndarray
        Array containing zscores that correspond to each pixel in the given image.
    '''
    thresh = np.copy(threshed_img)
    # Replace zero values with NaN
    tmp = np.where(thresh == 0, np.nan, thresh)

    # Calculate standard deviation of pixels extracted during thresholding.
    std = np.nanstd(tmp)

    # Calculate Z-scores for given image.
    zscores = (thresh - np.nanmean(tmp))/std         
    return zscores

class MAANFFilter(Filter):
    """
    Extends the Filter abstract class to implement an advanced filter that removes as much
    noise and artefacts as possible from the images.
    """    
    def filter(self, img):
        """Responsible for the first round of filtering noise from the image.

        Args:
            img (ndarray): The image to filter.

        Returns:
            ndarray: A mask representing the locations of the galaxy pixels after filtering.
        """
        self.threshed, self.basic = thresh_image(img)
        return self.threshed
        
    def additional_filter(self, img):
        """Conducts additional noise filtering if necessary.

        Args:
            img (ndarray): The original image before any filtering was conducted.

        Returns:
            ndarray: A mask representing the locations of the galaxy pixels after filtering.
        """
        if not self.basic:
            # If basic thresholding was not used, additional processing is necessary to account for the presence of noise
            self.threshed = prep_bent_class(img, self.threshed)
            self.threshed = remove_small_components(self.threshed, thresh=10)
        return self.threshed