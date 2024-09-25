import warnings
import os
import glob
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from MASCFilter import MASCFilter
from MAANFFilter import MAANFFilter
from FeatureExtractor import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rotate
from tqdm import tqdm
from time import time

FILTER_OPTIONS = {
    "MASC": MASCFilter,
    "MAANF": MAANFFilter
}

def normalize_image(img):
    """Normalize a given image

    Args:
        img (ndarray): The image to normalize

    Returns:
        ndarray: The normalized image
    """
    bot = np.min(img)
    top = np.max(img)
    norm = (img - bot)/(top - bot)
    return norm

def probe_dir(dir_path):
    """Check whether directory exists and if not create it

    Args:
        dir_path (String): Path to the required directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def standardise_galaxies(X, filter_type):
    '''
    Standardise the rotation of the given galaxies.

    Args:
        X (ndarray) : The radio galaxy samples that need to be derotated.
        filter_type (str): The type of filtering algorithm to use. Options include "simple" and
            "advanced".
    
    Returns
    -------
    ndarray
        An array of derotated galaxies that corresponds to the given array of galaxies.
    '''
    derotated = []
    identity = np.array([
        [1, 0],
        [0, 1]
    ])
    
    for i in tqdm(range(len(X))):
        # Preprocess images
        img = X[i]
        filter_driver = FILTER_OPTIONS[filter_type]()
        threshed = filter_driver.filter(img)

        # Construct matrix with galaxy pixel coordinates
        rows, cols = np.where(threshed > 0)
        coords = np.array([cols, rows])
        
        # Centre the images at the origin
        mean = np.mean(coords,axis=1)[:,np.newaxis]
        coords = (coords - mean)

        # Calculate the principal directions using the SVD
        u, _, _ = np.linalg.svd(coords,full_matrices=False)
        
        # Determine if any rotation is necessary
        if (np.abs(identity - u) < 1e-5).all():
            # No rotation
            derotated.append(img.copy())
            continue
            
        # Calculate the angle of rotation
        angle = np.arcsin(abs(u.T[0, 1]))*(180/np.pi)
        r_angle = np.radians(angle)
        tmp_cos = np.cos(r_angle)

        # Undo reflections
        if u[0, 0] != 0:
            if tmp_cos - u[0, 0] >= 1e-6:
                if (tmp_cos - (-1*u[0, 0]) < 1e-6):
                    u[0, 0] = -1*u[0, 0]
                    u[1, 0] = -1*u[1, 0]

            if tmp_cos - u[1, 1] >= 1e-6:
                if (tmp_cos - (-1*u[1, 1]) < 1e-6):
                    u[0, 1] = -1*u[0, 1]
                    u[1, 1] = -1*u[1, 1]

        # Identify and correct for special case where reflections cannot be detected
        sgns = np.sign(u.T)
        if u[0, 0] == 0:
            if sgns[0, 1] + sgns[1, 0] != 0:
                u[0, 1] = -1*u[0, 1]
                u[1, 1] = -1*u[1, 1]
                sgns = np.sign(u.T)

        sgn = sgns[0, 1]

        # Determine the direction of rotation
        if sgn >= 0:
            #Anti-clockwise rotation
            r_img = rotate(img.copy(),angle)
        else:
            #Clockwise rotation
            r_img = rotate(img.copy(),360-angle)

        derotated.append(r_img)
    derotated = np.array(derotated)
    print("Derotation completed...")
    return derotated

def normalize_feature_targets(train_y_aux, val_y_aux, test_y_aux):
    """Normalize the feature vectors to roughly be within the 0-1 range.

    Args:
        train_y_aux (ndarray): The feature vectors corresponding to the training data
        val_y_aux (ndarray): The feature vectors corresponding to the validation data
        test_y_aux (ndarray): The feature vectors corresponding to the test data.

    Returns:
        Tuple: A tuple containing the normalized feature vectors for each subset.
    """
    scaler = MinMaxScaler()
    train_y_aux = scaler.fit_transform(train_y_aux)
    val_y_aux = scaler.transform(val_y_aux)
    test_y_aux = scaler.transform(test_y_aux)
    return train_y_aux, val_y_aux, test_y_aux

def augment_galaxies(X, y, y_aux):
    """Performs rotational augmentation of a given dataset and its labels.

    Args:
        X (ndarray): The images in the dataset.
        y (ndarray): The class labels corresponding to the given images.
        y_aux (ndarray): The auxiliary feature vectors corresponding to the given images.

    Returns:
        tuple: Returns a tuple containing the augmented images and their corresponding
            class and feature vector labels.
    """
    results = list(map(augment_image, X, y, y_aux))
    new_Xs = []
    new_ys = []
    new_y_auxs = []
    for res in results:
        new_Xs.extend(res[0])
        new_ys.extend(res[1])
        new_y_auxs.extend(res[2])
    new_Xs = np.array(new_Xs)
    new_ys = np.array(new_ys)
    new_y_auxs = np.array(new_y_auxs)
    return new_Xs, new_ys, new_y_auxs

def augment_image(X, y, y_aux):
    """Augments a given image and its labels, by rotating the image a number of times and
    duplicating the labels appropriately.

    Args:
        X (ndarray): The image to augment.
        y (int): The class label corresponding to the image.
        y_aux (int): The feature vector label corresponding to the given image.

    Returns:
        tuple: Returns a tuple containing the augmented images and the corresponding
            class and feature vector labels.
    """
    augmented_X = [X]
    augmented_y = [y]
    augmented_y_aux = [y_aux]
    for deg in range(45,360,45):
        augmented_X.append(rotate(X.copy(),deg))
        augmented_y.append(y)
        augmented_y_aux.append(y_aux)
    return augmented_X, augmented_y, augmented_y_aux

def load_galaxy_data(filter_type="MASC"):
    """
    Loads the galaxy data and prepares the necessary npy files for use in model training and evaluation.
    
    Args:
        filter_type (str, Optional): Indicates which type of filter to use when removing noise.
            Options include MASC and MAANF. Defaults to MASC.
    """
    if not os.path.exists('data/FRGMRC-221022-RELEASE-V1.0/'):
        warnings.warn('Please download and extract the FRGMRC data from Zenodo into the data directory first', RuntimeWarning)
        return
    
    SOURCE_COUNTS = {
        "BENT": 214,
        "COMPACT": 209,
        "FRI": 183,
        "FRII": 358,
    }
    
    # Excluding ambiguous sources that are currently being reconsidered for inclusion in FRGMRC
    # Can remove this section once the final decisions have been made
    EXCLUSIONS = [
        "data/FRGMRC-221022-RELEASE-V1.0/BENT_210.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/COMPACT_17.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/COMPACT_20.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/COMPACT_40.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRI_39.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRI_45.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRI_49.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRI_86.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_14.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_70.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_90.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_117.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_182.fits",
        "data/FRGMRC-221022-RELEASE-V1.0/FRII_331.fits",
    ]
    
    rootdir = 'data/FRGMRC-221022-RELEASE-V1.0/'
    label_dict = {'BENT': 0, 'COMPACT': 1, 'FRI': 2, 'FRII': 3}
    filter_driver = FILTER_OPTIONS[filter_type]()
    feature_extractor = FeatureExtractor()
    
    X = []
    y = []
    y_aux  = []
    for cls, count in SOURCE_COUNTS.items():
        sub_path = f"{cls}_"
        for i in range(1, count):
            file = rootdir + sub_path + f'{i}.fits'
            if file in EXCLUSIONS:
                continue
            try:
                img = fits.open(file)[0].data
            except:
                print(f"{file} does not exist")
                continue
            label = label_dict[cls]
            
            # Get central 150x150 cutout
            img = img[75:225, 75:225]
            img = (img - 0)/1. # Data is read in as big endian which is incompatible with skimage. This calculation should not alter data, but fixes buffer type.
            
            # Normalize pixel intensities per image
            img = normalize_image(img)
            
            # Extract core features
            threshed = filter_driver.filter(img)
            cores, inter_dist, core_pixels, total_pixels = feature_extractor.detect_cores(img, threshed)
            core_frac = core_pixels/total_pixels
            
            threshed = filter_driver.additional_filter(img)
            
            # Extract remaining features (FR ratio, potential bent)
            fr_ratio, pb = feature_extractor.extract_features(threshed)
            bent = 1 if pb else 0
            feats = [bent, fr_ratio, cores, core_frac]
            
            X.append(img)
            y.append(label)
            y_aux.append(feats)
    # Save arrays
    X = np.array(X)
    y = np.array(y)
    y_aux = np.array(y_aux)
    
    probe_dir('data/FRGMRC/')
    
    np.save(f'data/FRGMRC/galaxy_X.npy', X)
    np.save(f'data/FRGMRC/galaxy_y.npy', y)
    np.save(f'data/FRGMRC/galaxy_y_aux_{filter_type}.npy', y_aux)

def create_holdout_test():
    """
    Isolates 10% of the data for a holdout test set.
    """
    root = 'data/FRGMRC/'
    X = np.load(root + f"galaxy_X.npy")
    y = np.load(root + f"galaxy_y.npy")
    y_aux_masc = np.load(root + f"galaxy_y_aux_MASC.npy")
    y_aux_maanf = np.load(root + f"galaxy_y_aux_MAANF.npy")
    
    train_X, test_X, train_y, test_y, train_y_aux_masc, test_y_aux_masc, train_y_aux_maanf, test_y_aux_maanf = train_test_split(
        X,
        y,
        y_aux_masc,
        y_aux_maanf,
        test_size=0.1,
        stratify=y,
        shuffle=True,
        random_state=42,
    )
    np.save(root+f"train_X.npy", train_X)
    np.save(root+f"test_X.npy", test_X)
    np.save(root+f"train_y.npy", train_y)
    np.save(root+f"test_y.npy", test_y)
    np.save(root+f"train_y_aux_MASC.npy", train_y_aux_masc)
    np.save(root+f"test_y_aux_MASC.npy", test_y_aux_masc)
    np.save(root+f"train_y_aux_MAANF.npy", train_y_aux_maanf)
    np.save(root+f"test_y_aux_MAANF.npy", test_y_aux_maanf)

if __name__ == "__main__":
    filter_type = "MASC"
    load_galaxy_data(filter_type=filter_type)
    filter_type = "MAANF"
    load_galaxy_data(filter_type=filter_type)
    create_holdout_test()