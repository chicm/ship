import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.transform import resize
import cv2

from utils import run_length_decoding
import pdb
import settings

def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (H x W).

    """
    #n_channels = image.shape[0]
    #resized_image = resize(image, (target_size[0], target_size[1]), mode='constant', anti_aliasing=True)
    resized_image = cv2.resize(image, target_size)
    return resized_image


def binarize(image, threshold):
    image_binarized = (image > threshold).astype(np.uint8)
    return image_binarized

def save_pseudo_label_masks(submission_file):
    df = pd.read_csv(submission_file, na_filter=False)
    print(df.head())

    img_dir = os.path.join(settings.TEST_DIR, 'masks')

    for i, row in enumerate(df.values):
        decoded_mask = run_length_decoding(row[1], (101,101))
        filename = os.path.join(img_dir, '{}.png'.format(row[0]))
        rgb_mask = cv2.cvtColor(decoded_mask,cv2.COLOR_GRAY2RGB)
        print(filename)
        cv2.imwrite(filename, decoded_mask)
        if i % 100 == 0:
            print(i)



if __name__ == '__main__':
    save_pseudo_label_masks('V456_ensemble_1011.csv')