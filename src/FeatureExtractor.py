import numpy as np
from skimage.morphology import dilation, square

def reduce_component(comp, img):
    '''
    This function is used to reduce the given component to a single pixel.

    Parameters:
    -----------
    comp : ndarray
        An image containing the component that needs to be reduced.
    img : ndarray
        The original image from which the component was extracted.

    Returns:
    tuple
        A tuple containing the row, column and value of the pixel identified as the center of the component.
    '''
    # Replace the component pixels with their original intensities.
    masked = np.where(comp, img, 0)

    # Find the maximum value in the component.
    max_val = np.amax(masked)

    # Find the coordinates of the brightest pixel.
    r, c = np.where(masked == max_val)

    return (r[0], c[0], max_val)

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

def calc_comp_size(comp):
    '''
    This function is used to calculate the size of the given component. It returns a third of this value
    which represents an approximation of the distance between the two cores that might be overlapping.

    Parameters:
    -----------
    comp : ndarray
        The extracted component for which the size should be calculated.

    Returns:
    --------
    float
        A third of the size of the component.
    '''
    # Find the coordinates of the pixels in the component
    rs, cs = np.where(comp == 1)
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)

    # Identify the distance between the min and max row and min and max column. Use the biggest distance as the size of the component.
    max_dist = np.max([abs(max_r - min_r), abs(max_c - min_c)])

    return np.round((1/3)*max_dist)

def calc_core_dist(coords):
    '''
    This function is used to calculated the distance between the cores.

    Parameters:
    -----------
    coords: list
        A list containing the coords and value of the center pixels in each identified core.
    
    Returns:
    --------
    float
        The average distance between the cores.
    '''
    total = 0
    for i in range(len(coords)):
        cur = coords[i]
        min_dist = 9999999999999
        first = True
        for j in range(len(coords)):
            # Calculate the distance between core i and each other core in the list
            if j == i:
                continue
            neigh = coords[j]
            dist = calc_euc(cur, neigh)
            if first:
                min_dist = dist
                first = False
            elif dist < min_dist:
                min_dist = dist

        # Keep track of the distances between all of the cores
        total += min_dist
    # Return the average distance
    return np.round(total/len(coords))

def calc_euc(coord1, coord2):
    '''
    This function calculate the euclidean distance between two coordinates.

    Parameters:
    -----------
    coord1 : tuple
        The tuple containing the first coordinate
    coord2 : tuple
        The tuple containing the second coordinate

    Returns:
    --------
    float
        The Euclidean distance between the two coordinates.
    '''
    c1 = np.array([coord1[0], coord1[1]])
    c2 = np.array([coord2[0], coord2[1]])
    return np.linalg.norm(c1 - c2)

def galaxy_deviation(img):
    '''
    This function calculates the deviation of the pixels that have been identified as
    belonging to the galaxy.

    Parameters:
    -----------
    img : ndarray
        The thresholded image for which the standard deviation should be calculated.

    Returns:
    --------
    float
        The standard deviation of the thresholded pixels in the given image.
    '''
    thresh = np.copy(img)

    # Replace zeros with NaN
    tmp = np.where(thresh == 0, np.nan, thresh)

    # Calculate standard deviation without considering NaN elements
    std = np.nanstd(tmp)

    return std

def segment_cores(img, std):
    '''
    This function is used to extract the pixels that most likely correspond to cores from the given image.

    Parameters:
    -----------
    img : ndarray
        The image from which to extract the cores
    std : float
        The standard deviation of the non-zero pixels in the given image.

    Returns:
    ndarray
        A thresholded version of the given image, containing only pixels believed to belong to the cores.
    '''
    threshed = np.copy(img)

    # Replace zero pixels with NaN
    tmp = np.where(threshed == 0, np.nan, threshed)
    
    # Determine which quantile to use as the threshold to extract the noise.
    # The std gives an indication of how heterogeneous the pixel values are.
    # Wider varieties of pixel values require a lower quantile to extract all of the possible core pixels.
    if std > 0.20:
        q = np.nanquantile(tmp, 0.80)
    elif std > 0.13:
        q = np.nanquantile(tmp, 0.93)
    else:
        q = np.nanquantile(tmp, 0.98)

    return threshed > q

def process_cores(thresh, img=None):
    '''
    This function is used to count the number of cores in the thresholded image and
    to calculate the average distance between the cores.

    Parameters:
    -----------
    thresh : ndarray
        The thresholded image that only contains pixels that are likely to be part of cores.
    img : ndarray
        The image corresponding to thresh with the original pixel intensities.

    Returns:
    --------
    tuple
        A tuple containing the number of cores and the average distance between them.
    '''
    val_img = np.where(thresh, 1, 0)
    coords = []

    count = 0
    for r in range(val_img.shape[0]):
        for c in range(val_img.shape[1]):
            if val_img[r, c] == 1:
                # Expand pixel to find the corresponding core
                comp, size = find_connected_component(val_img, r, c)

                # Approximate the center of the core
                coords.append(reduce_component(comp, img))

                # Remove the core from the image so it is not found again
                val_img -= comp

                count += 1
    
    if count == 1:
        # If the count is one, just take a third of the component size as the inter-core distance
        inter_core_size = calc_comp_size(comp)
    else:
        # If there are more than 1 core, calculate the average distance between the cores.
        inter_core_size = calc_core_dist(coords)
    return count, inter_core_size

def rotate_axes(img):
    '''
    This function is used to standardize the rotation of the galaxy pixels in the given image.

    Parameters:
    -----------
    img : ndarray
        The image for which rotation needs to be standardized.

    Returns:
    --------
    ndarray
        An array of the new pixel coordinates after standardizing rotation
    '''
    # Extract the coordinates of the galaxy pixels
    coords = [[], []]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                coords[0].append(j)
                coords[1].append(i)

    coords = np.array(coords)

    # Center the image at the mean coordinate (necessary before PCA)
    mean = np.mean(coords,axis=1)[:,np.newaxis]
    coords = (coords - mean)

    # Calculate the principal directions using the SVD
    u, s, vh = np.linalg.svd(coords,full_matrices=False)

    # U contains the principal components in the columns, premultiplying by this matrix
    # rotates the coordinates to align with these principal components
    rotated = u.T.dot(coords)

    return rotated

def calc_vertical_size(coords):
    '''
    This function is used to calculate the distance between the minimum and maximum coordinate
    along the second principal component. This can be seen as calculating the size of the galaxy along the vertical axis.

    Parameters:
    -----------
    coords : ndarray
        An array of galaxy pixel coordinates after standardising the rotation of the galaxy.

    Returns:
    --------
    float
        The distance between the minimum and maximum coordinate along the second principal component.
        
    '''
    start_row = coords[1, :].min()
    end_row = coords[1, :].max()

    return end_row - start_row

def calc_horizontal_size(coords):
    '''
    This function is used to calculate the distance between the minimum and maximum coordinate
    along the first principal component. This can be seen as calculating the size of the galaxy along the horizontal axis.

    Parameters:
    -----------
    coords : ndarray
        An array of galaxy pixel coordinates after standardising the rotation of the galaxy.

    Returns:
    --------
    float
        The distance between the minimum and maximum coordinate along the first principal component.
        
    '''
    start_col = coords[0, :].min()
    end_col = coords[0, :].max()

    return end_col - start_col

def potential_bent(v_size, h_size):
    '''
    This function determines whether the galaxy contains a curve.

    Parameters:
    -----------
    v_size : float
        The size of the galaxy along the second principal component
    h_size : float
        The size of the galaxy along the first principal component

    Returns:
    --------
    boolean
        A boolean that indicates whether the galaxy might contain a curve.
    '''
    ratio = h_size/v_size

    return 0.5 < ratio < 6.7 and 14 < v_size < 54

class FeatureExtractor():
    """
    This class provides a wrapper for all feature extraction functionality
    """
    def detect_cores(self, img, threshed):
        '''
        This function is used to identify the cores in a given image.

        Parameters:
        -----------
        img : ndarray
            The image within which to look for cores.
        threshed : ndarray
            The thresholded image that corresponds to img.

        Returns:
        --------
        tuple
            A tuple that contains the number of cores, the distance between the cores,
            the number of pixels in the cores and the total number of pixels in the galaxy. 
        '''
        # Replace the extracted galaxy pixels with their original values
        processed = np.where(threshed, img, 0)

        # Find the standard deviation of the non-zero pixels.
        std = galaxy_deviation(processed)

        # Extract the cores from the image
        hard_thresh = segment_cores(processed, std)

        # Count the number of pixels in the cores
        core_pixels = np.sum(hard_thresh)

        # Count the number of pixels in the galaxy
        bin_total = processed > 0
        total_pixels = np.sum(bin_total)

        # Count the cores and calculate the distance between them
        cores, self.inter_dist = process_cores(hard_thresh, img)
        
        return cores, self.inter_dist, core_pixels, total_pixels

    def extract_features(self, threshed):
        """Extracts the FR ratio and determines whether an object might have a bent shape.

        Args:
            threshed (ndarray): The thresholded image from which to extract the features.

        Returns:
            Tuple: A tuple containing the FR ratio as well as a flag indicating whether a galaxy is potentially a bent galaxy.
        """
        # Standardize the rotation of the thresholded pixels.
        rotated = rotate_axes(threshed)
        
        # Calculating the distance between the minimum and maximum coordinate along the first principal component.
        gal_size = max(rotated[0, :]) - min(rotated[0, :])

        # Calculate FR Ratio
        fr_ratio = np.round(self.inter_dist/gal_size, 2)
    
        # Calculate the size of the galaxy along the first and second principal component.
        v_size = calc_vertical_size(rotated)
        h_size = calc_horizontal_size(rotated)
        
        # If the vertical size is zero we don't need to do any further processing.
        if v_size == 0:
            return fr_ratio, False

        # Calculate the ratio between the size of the galaxy along the first and second principal component
        ratio = h_size/v_size

        # Determine whether the galaxy might have a bend in it
        pb = potential_bent(v_size, h_size)
        
        return fr_ratio, pb