
from skew_correction.helper import read_raw_image, get_images_in_dir, pil2np, np2pil, tensor2pil, get_skew
from torchvision import transforms


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
    







