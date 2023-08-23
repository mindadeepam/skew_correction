import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from skew_correction.helper import read_raw_image
from skew_correction.constants import angle2label, image_size

root_dir = "/".join(( os.path.realpath(__file__)).split("/")[:-2])

#######################################################################################################################################


# functools.partial(collate_fn, split=)

# Dataset
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.456], std=[0.225]),
    # transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(1),
    transforms.ColorJitter(0.2),
    transforms.RandomAutocontrast(0.5),
])

test_transform = tensor_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


#######################################################################################################################################

class DatasetClass(Dataset):
    def __init__(self, file, split=None):

        self.df = pd.read_csv(file)
        self.image_paths = self.df['path'].values
        self.labels = self.df['label'].values

        if split=='train' or file.split("/")[-1].startswith("train"):
            self.transform = train_transform
        else:
            self.transform = tensor_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        path = self.image_paths[idx]
        angle = self.labels[idx]
        image = read_raw_image(path)
        image = self.transform(image)
        label = angle2label[angle]
        return image, torch.tensor(label, dtype=torch.long)

#######################################################################################################################################


class RegressionDataset(Dataset):
    def __init__(self, csv_path=None, split="test", df = None):
        super().__init__()
        
        if isinstance(df, pd.DataFrame):
            self.df=df    
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            raise Exception("pass either csv_path or df")


        self.filepaths = self.df["filepath"]
        self.labels = self.df["angle"]
        self.split = split

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = read_raw_image(self.filepaths[idx])
        label = self.labels[idx]

        if self.split=="train":
            img = train_transform(img)
        else:
            img = test_transform(img)

        return img, torch.tensor(label, dtype=torch.float)


#######################################################################################################################################

class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.0, seed=42, train_bs=4, val_bs=4):
        super().__init__()
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.train_batch_size = train_bs
        self.val_batch_size = val_bs

    def setup(self, stage=None):
        # Calculate the sizes of train atasets
        total_size = len(self.dataset)
        train_size = int(self.train_ratio * total_size)
        
        # Calculate the test size and val sizes
        if self.test_ratio > 0:
            val_size = int(self.val_ratio * total_size)
            test_size = total_size - train_size - val_size
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            val_size = total_size - train_size
            self.train_ds, self.val_ds = random_split(
                self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False) 

    def test_dataloader(self):
        if self.test_ratio > 0:
            return DataLoader(self.test_ds, batch_size=self.val_batch_siz)
        else:
            return None

    def predict_dataloader(self):
        if self.test_ratio > 0:
            return pl.DataLoader(self.test_ds, batch_size=32)
        else:
            return None



#######################################################################################################################################

## utilities
import matplotlib.pyplot as plt
import numpy as np

def plot_random_images(dataset, num_images=10, figsize=(12, 8), cmap='gray'):
    """
    Plots random images from a given dataset class.

    Parameters:
        dataset (YourDatasetClass): The dataset class that provides access to the images.
        num_images (int): Number of random images to plot (default is 10).
        figsize (tuple): Figure size (default is (12, 8)).
        cmap (str): Color map for displaying the images (default is 'gray').

    Returns:
        None
    """
    # Get the number of available images in the dataset
    num_available_images = len(dataset)

    # Generate 10 random indices
    random_indices = np.random.randint(0, num_available_images, size=num_images)

    # Create a subplot grid to display the images
    num_rows = num_images // 5 if num_images > 5 else 1
    num_cols = min(5, num_images)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Load the image from the dataset using the random index
            image, angle = dataset[random_indices[i]]
            
            # If the image is in grayscale, remove the channel dimension for plotting
            image = image.squeeze(0)
            ax.set_title(f"Value: {angle:.2f}")
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
    plt.tight_layout()
    plt.show()