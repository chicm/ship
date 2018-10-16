import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.transform import resize
import cv2
from scipy import ndimage

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

def split_mask(mask, threshold_obj=30, threshold=0.5): 
    #ignor predictions composed of "threshold_obj" pixels or less
    if threshold is not None:
        mask = mask > threshold
    #threshold = 0.5 
    labled, n_objs = ndimage.label(mask)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(np.uint8)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result

def mask_to_bbox(mask):
    #label = np.where(labeled_mask == label_id, 1, 0).astype(np.uint8)
    img_box = np.zeros_like(mask)
    mask = (mask > 0).astype(np.uint8)
    _, cnt, _ = cv2.findContours(mask, 1, 2)
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_box, [box], 0, 1, -1)
    
    return img_box

def masks_to_bounding_boxes(mask):
    labeled_mask, n_objs = ndimage.label(mask)

    if labeled_mask.max() == 0:
        return labeled_mask
    else:
        img_box = np.zeros_like(labeled_mask)
        for label_id in range(1, labeled_mask.max() + 1, 1):
            label = np.where(labeled_mask == label_id, 1, 0).astype(np.uint8)
            _, cnt, _ = cv2.findContours(label, 1, 2)
            rect = cv2.minAreaRect(cnt[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_box, [box], 0, label_id, -1)
        return img_box



if __name__ == '__main__':
    pass
    #save_pseudo_label_masks('V456_ensemble_1011.csv')