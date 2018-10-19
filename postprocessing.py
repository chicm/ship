import os
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.transform import resize
import cv2
from scipy import ndimage
from loader import get_train_val_loaders
from utils import run_length_decoding
from metrics import intersection_over_union, intersection_over_union_thresholds
from models import UNetShipV1
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
    FILL_VALUE = 1

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
            cv2.drawContours(img_box, [box], 0, FILL_VALUE, -1)
        return img_box

def get_val_result(batch_size=16, ckp=None):
    model = UNetShipV1(34)
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    if ckp is None:
        ckp = model_file
    model.load_state_dict(torch.load(ckp))
    model = model.cuda()
    model.eval()

    _, val_loader = get_train_val_loaders(batch_size=batch_size, drop_empty=True)
    outputs = []
    with torch.no_grad():
        for img, target, ship_target in val_loader:
            img, target, ship_target = img.cuda(), target.cuda(), ship_target.cuda()
            output, _ = model(img)
            #print(output.size(), salt_out.size())
            output = torch.sigmoid(output)
            
            for o in output.cpu():
                outputs.append(o.squeeze().numpy())
    return outputs, val_loader.y_true

def save_val_result():
    outputs, y_true = get_val_result(16)
    np.savez_compressed(os.path.join(settings.DATA_DIR, 'tmp', 'val_out.npz'), outputs=outputs, y_true=y_true)

def test_bbox():
    tgt_size = (settings.ORIG_H, settings.ORIG_W)
    outputs, y_true = get_val_result(16)
    resized = list(map(lambda x: resize_image(x, tgt_size), outputs))
    print(resized[0].shape, len(resized))
    y_pred = list(map(lambda x: (x > 0.5).astype(np.uint8), resized))
    print(y_pred[0].shape, len(y_pred))

    iou_score = intersection_over_union(y_true, y_pred)
    iout_score = intersection_over_union_thresholds(y_true, y_pred)
    print(iou_score, iout_score)

    processed = list(map(lambda x: masks_to_bounding_boxes(x), y_pred))

    iou_score = intersection_over_union(y_true, processed)
    iout_score = intersection_over_union_thresholds(y_true, processed)
    print(iou_score, iout_score)

def test_bbox_2():
    a = np.zeros((5,5))
    a[0,:] = 1
    a[4,:] = 1
    b = masks_to_bounding_boxes(a)
    print(b)

if __name__ == '__main__':
    #test_bbox()
    save_val_result()
    #test_bbox_2()
    #save_pseudo_label_masks('V456_ensemble_1011.csv')