import lightning.pytorch as pl
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.insert(0,"data/datasets")
from FRGMRCDataset import FRGMRCDataset
sys.path.insert(0,"src")
from prepare_data import standardise_galaxies, augment_galaxies, normalize_feature_targets
from time import time

class StandardDataModule(pl.LightningDataModule):
    """
    This datamodule encapsulates all of the data logic necessary during experimentation, such as splitting the samples,
    conducting further preprocessing that is reliant on splits and creating dataloaders for easy use during training
    and testing. It is intended for models that only have the images as inputs.
    """    
    def __init__(self, batch_size=64, seed=42, rotation_method="augment", filter_method="advanced"):
        """Initialize a datamodule to use with most of the neural networks used in these experiments.

        Args:
            batch_size (int, optional): The size to use for batches. Defaults to 64.
            seed (int, optional): The random seed to use. Defaults to 42.
            rotation_method (str, optional): Indicates whether augmentation or rotational standardisation or neither should be used to address
                rotational variations. Defaults to "augment".
            filter_method (str, optional): Indicates whether a simple or advanced noise filtering strategy should be applied.
                Defaults to "advanced".
        """
        super().__init__()
        self.save_hyperparameters()
        self.rest_X = np.load("data/FRGMRC/train_X.npy")
        self.test_X = np.load("data/FRGMRC/test_X.npy")
        self.rest_y = np.load("data/FRGMRC/train_y.npy")
        self.test_y = np.load("data/FRGMRC/test_y.npy")
        self.rest_y_aux = np.load(f"data/FRGMRC/train_y_aux_{filter_method}.npy")
        self.test_y_aux = np.load(f"data/FRGMRC/test_y_aux_{filter_method}.npy")
        
        # If applying rotational standardisation, conduct it here.
        if self.hparams.rotation_method == "standardise":
            start = time()
            self.rest_X = standardise_galaxies(self.rest_X, filter_type=self.hparams.filter_method)
            self.test_X = standardise_galaxies(self.test_X, filter_type=self.hparams.filter_method)
            end = time()
            print(f"Standardisation took {end-start} seconds.")
        
        # Create KFold generator for later use
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.hparams.seed)
        
        self.fold_generator = skf.split(
            self.rest_X,
            self.rest_y
        )
        
        # Prepare first set of training and validation sets.
        self.create_next_train_val_sets()

    def create_next_train_val_sets(self):
        """
        Creates the subsets necessary for training and evaluating a model during experimentation.
        """
        train_idxs, val_idxs = next(self.fold_generator)
        self.train_X = self.rest_X[train_idxs]
        self.train_y = self.rest_y[train_idxs]
        self.train_y_aux = self.rest_y_aux[train_idxs]
        self.val_X = self.rest_X[val_idxs]
        self.val_y = self.rest_y[val_idxs]
        self.val_y_aux = self.rest_y_aux[val_idxs]
        
        # Normalize feature vectors
        self.train_y_aux, self.val_y_aux, self.test_y_aux = normalize_feature_targets(self.train_y_aux, self.val_y_aux, self.test_y_aux)
        
        # Rotational augmentation
        if self.hparams.rotation_method == "augment":
            self.train_X, self.train_y, self.train_y_aux = augment_galaxies(self.train_X, self.train_y, self.train_y_aux)
            
            # Shuffle datasets to improve training
            shuffle_idxs = np.random.permutation(len(self.train_X))
            self.train_X = self.train_X[shuffle_idxs]
            self.train_y = self.train_y[shuffle_idxs]
            self.train_y_aux = self.train_y_aux[shuffle_idxs]
        
        # Create datasets for the various subsets
        self.train = FRGMRCDataset(self.train_X, self.train_y, self.train_y_aux)
        self.val = FRGMRCDataset(self.val_X, self.val_y, self.val_y_aux)
        self.test = FRGMRCDataset(self.test_X, self.test_y, self.test_y_aux)
    
    def train_dataloader(self):
        """Creates a dataloader for the training dataset

        Returns:
            DataLoader: A dataloader that iterates through the training dataset.
        """
        print("Ensure that create_next_train_val_sets has been called if this is not the first run.")
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        """Creates a dataloader for the validation dataset

        Returns:
            DataLoader: A dataloader that iterates through the validation dataset.
        """
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=0, pin_memory=True)
    
    def test_dataloader(self):
        """Creates a dataloader for the testing dataset

        Returns:
            DataLoader: A dataloader that iterates through the testing dataset.
        """
        return DataLoader(self.test, batch_size=len(self.test), num_workers=0, pin_memory=True)

    def rest_dataloader(self):
        """Creates a dataloader for the rest dataset

        Returns:
            DataLoader: A dataloader that iterates through the rest dataset.
        """
        rest = FRGMRCDataset(self.rest_X, self.rest_y, self.rest_y_aux)
        return DataLoader(rest, batch_size=len(rest), num_workers=0, pin_memory=True)

if __name__ == '__main__':
    dm = StandardDataModule(rotation_method='augment', filter_method='advanced')
    dl = dm.train_dataloader()