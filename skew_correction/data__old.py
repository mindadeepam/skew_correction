from PIL import Image
import os
from torchvision import transforms 
from torch.utils.data import Dataset
import random
import torch




# Dataset
data_transform = transforms.Compose([
    # transforms.Resize((400, 400)),
    # transforms.GaussianBlur(3),
    # transforms.ColorJitter(0.3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





class SkewAngleDataset(Dataset):
    def __init__(self, data_dir, skew_range=(-30, 30), step_size=5, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.skew_range = skew_range
        self.step_size = step_size
        self.ang2index = {0:0, 90:1, -90:2, 180:3}

        # List all image file names in the data directory
        self.image_files = [x for x in os.listdir(data_dir) if x[-3:]=='jpg']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        # print(img_name)
        image = Image.open(img_name)
        image = ImageOps.exif_transpose(image)

        # Apply skew angle
        # skew_angle = random.uniform(self.skew_range[0], self.skew_range[1])
        skew_angle = random.choice([0,90, 270, 360])
        # skew_angle = round(skew_angle / self.step_size) * self.step_size
        
        # print(f"inside {skew_angle}")
        image = image.rotate(skew_angle, expand=True)
        ## get hog features
        features, hog_image = get_hog(pixels = rgb2gray(image))

        if self.transform:
            image = self.transform(image)
        


        label = self.ang2index[skew_angle]

        return hog_image, torch.tensor(label, dtype=torch.long)


