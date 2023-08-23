#general libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import re

#PIL
from PIL import Image, ImageOps

#torch
import torch
torch.set_num_threads(1)

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


def extract_numbers_from_end(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_skew(image):

    # convert to gray scale and ensure output is numpy array(rgb2gray output is numpy)
    num_channels = len(image.mode)

    if num_channels == 3:
        image = pil2np(np2pil(image).convert('L'))
    
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


def hough_transform(img):
    """"takes read_raw_image output as input (pil images). returns rectified pil img grayscale"""
    angle = get_skew(img)
    return img.rotate(angle, expand=True), angle


def remove_padding(image):
    # Get the bounding box of the non-zero (non-padding) region in the image (pil)
    bbox = image.getbbox()

    if bbox:
        # Crop the image to the bounding box to remove the padding
        cropped_image = image.crop(bbox)
        return cropped_image
    else:
        # If the entire image is padding, return the original image
        return image
    



