from torch.utils.data import Dataset
from torchvision import transforms

class FRGMRCDataset(Dataset):
    """
    Very basic dataset to encapsulate our data for use in dataloader.
    """
    def __init__(self, X, y, y_aux):
        """Initialize the data for the dataset, along with any transforms that have to be performed.

        Args:
            X (numpy.ndarray): The array containing the image data.
            y (numpy.ndarray): The array containing the labels corresponding to the images.
            y_true (numpy.ndarray): The array containing the auxiliary feature labels corresponding to the images.
        """
        self.X = X
        self.y = y
        self.y_aux = y_aux
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """
        Determines the number of samples in the dataset
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """Returns an image and its corresponding labels from the dataset.

        Args:
            idx (int): The index from which to extract the image and labels.

        Returns:
            Tuple[numpy.ndarray, int, str]: A tuple containing the image, a label indicating the image class 
                and the auxiliary feature label for the image.
        """
        sample = self.X[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, self.y[idx], self.y_aux[idx]