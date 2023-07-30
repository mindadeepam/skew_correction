#general libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

#PIL
from PIL import Image, ImageOps

#torch
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#skimage
from skimage.feature import hog, canny
from skimage.filters import threshold_otsu, sobel
from skimage.io import imread, imshow
from skimage.transform import resize, rotate, hough_line, hough_line_peaks
from skimage import exposure, color
from skimage import io, transform, img_as_ubyte
from skimage.color import rgb2gray
from scipy.stats import mode

#more libraries
from skimage import io, color, transform, img_as_ubyte
import numpy as np
import os
import random
from PIL import ImageOps, Image

#tqdm
from tqdm import tqdm

#warnings
import warnings
warnings.filterwarnings('ignore')

# USE_OPENMP=1

#custom modules
from skew_correction.constants import label2angle, device

def get_images_in_dir(path, extensions=['jpg', 'png', 'jpeg'], return_path=False, shuffle=False):
    
    if return_path:
        res = [os.path.join(path, x) for x in os.listdir(path) if x.split('.')[-1] in extensions]
    else:
        res = [x for x in os.listdir(path) if x.split('.')[-1] in extensions]

    if shuffle:
        random.shuffle(res)
    return res


def np2pil(img):
    return Image.fromarray(img_as_ubyte(img))

def pil2np(img):
    return (np.array(img)/255).astype(np.float32)

def tensor2pil(img):

    if img.device.type == 'cuda':
        img = img.cpu()

    if len(img.shape)>3:
        img = img.squeeze(0)

    if img.shape[0] == 1:
        return Image.fromarray((img.numpy() * 255).astype('uint8').transpose(1,2,0).squeeze(), mode='L')
    else:
        return Image.fromarray((img.numpy() * 255).astype('uint8').transpose(1,2,0))


def read_raw_image(path, mode='L'):
    """reads img from path. default is gray scale. retursn pil image"""

    img = Image.open(path).convert(mode)
    try:
        img = ImageOps.exif_transpose(img)
    except:
        print("Error in exif_transpose")
    return img

def get_skew(image):

    # convert to gray scale and ensure output is numpy array(rgb2gray output is numpy)
    num_channels = len(image.mode)

    if num_channels == 3:
        image = rgb2gray(image)
    
    image_array = pil2np(image)

    # convert to edges
    edges = canny(image_array)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2), keepdims=True)[0]
        
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)[0]
    
    # print(skew_angle)
    return skew_angle



##HOUGH transform functions
def binarizeImage(RGB_image):

    if len(RGB_image.shape) == 3 and RGB_image.shape[2] == 3:
        image = color.rgb2gray(RGB_image)
    else:
        image = RGB_image  # Assume grayscale image if not 3-channel RGB
    threshold = threshold_otsu(image)
    bina_image = image < threshold
    return bina_image


def findEdges(bina_image):
  
    image_edges = sobel(bina_image)
    return image_edges


def findTiltAngle(image_edges):  
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)

    angles_mode = mode(angles)
    angle = np.rad2deg(angles_mode.mode)
    # print("org angle", angle)
    
    if (angle < 0):
        r_angle = angle + 90
    
    else:
        r_angle = angle - 90
    origin = np.array((0, image_edges.shape[1]))
    return r_angle


def rotateImage(image, angle, img_name=None):
    # Convert RGB image to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = color.rgb2gray(image)
    else:
        image = image

    # Rotate the image
    rotated_image = transform.rotate(image, angle, resize=True)

    # Create a copy of the rotated image to preserve pixel values
    cropped_image = np.copy(rotated_image)

    # Find the bounding box of the rotated image
    nonzero_pixels = np.argwhere(rotated_image > 0)
    min_row, min_col = np.min(nonzero_pixels, axis=0)
    max_row, max_col = np.max(nonzero_pixels, axis=0)

    # Crop the rotated image to remove black borders
    cropped_image = cropped_image[min_row:max_row+1, min_col:max_col+1]

    # Convert the image to unsigned 8-bit integer format
    final_image = img_as_ubyte(cropped_image)

    # Save the resulting image
    # if img_name is None:
    # io.imsave('output_image.png', final_image)
    # else:
    #     os.makedirs('logs', exist_ok=True)
    #     io.imsave(os.path.join('logs', img_name), final_image)

    return cropped_image, final_image


def rectify_skew1(img_path, angle=None):

    
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    image_array = np.array(image)

    bina_image = binarizeImage(image_array)
    image_edges = findEdges(bina_image)
    if angle==None:
        angle = findTiltAngle(image_edges)[0]
    print(angle)
    # if img_name is None:
    _, fixed_image = rotateImage(image_array, angle)
    # else:
    #     fixed_image, _ = rotateImage(io.imread(img), angle, img_name)
    # print(fixed_image)
    return fixed_image




## Model training functions
def get_hog(path=None, pixels=None):
    
    if path is not None and pixels is not None:
        raise ValueError("Only one parameter (path or pixels) should be provided.")
    if path is None and pixels is None:
        raise ValueError("Either path or pixels should be provided.")
    if pixels is not None:
        image_array = pixels
        image_array = np.array(Image.fromarray(image_array).resize((256, 256)))
    else:
        image_pil = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image_pil = image_pil.convert("L")
        image_pil = image_pil.resize((256, 256))
        image_array = np.array(image_pil)

    # Define HOG parameters
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (6, 6)

    # Compute HOG features
    features, hog_image = hog(image_array, orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              visualize=True, block_norm='L2-Hys')

    # Display the HOG image
    # hog_image_pil = (hog_image * 255).astype(np.uint8)
    
    return features, hog_image



class HOG_Dataset(Dataset):
    def __init__(self, df, threshold=None, transform=None):
        self.df = df
        self.path = df.path.values
        self.labels = df.angles.values
#         self.threshold = threshold

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file_path and label for index
        label = self.labels[idx]
        path = self.path[idx]

        #rotate to the nearest angle
        if label >= 0 and label < 90:
            # print("between 0 and 90")
            _, rotated_pixels = rotateImage(io.imread(path), 90-label)
            labels = 90

        if label >= 90 and label < 180:
            # print("between 90 and 180")
            _, rotated_pixels = rotateImage(io.imread(path), 180-label)
            labels = 180

        if label >= 180 and label < 270:
            # print("between 180 and 270")
            _, rotated_pixels = rotateImage(io.imread(path), 270-label)
            labels = 270

        if label >= 270 and label <= 360:
            # print("between 270 and 360")
            _, rotated_pixels = rotateImage(io.imread(path), 360-label)
            labels = 0

        #hog
        hog_pixels = get_hog(pixels=rotated_pixels)
        
        #transform
        features = self.transform(hog_pixels)
        label = label2angle.get(labels)
        label = torch.tensor(label)
        
        label = label.to(device)
        features = features.to(device, dtype=torch.float)

        # features = torch.tensor(features, dtype=torch.float)

        return {'pixels': features,
               'label': label}

