import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as cocomask
from sklearn.utils import shuffle
import settings



def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b
    
    return ' '.join(str(rle_item) for rle_item in rle)


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape((shape[1], shape[0])).T


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def binary_from_rle(rle):
    return cocomask.decode(rle)

def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations

def get_train_val_meta(drop_empty=False):
    df = pd.read_csv(settings.TRAIN_META, na_filter=False)

    split_index = df.shape[0] - 10000

    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    df_train_ship = df_train[df_train['ship'] == 1]
    df_train_no_ship = shuffle(df_train[df_train['ship'] == 0]).iloc[:40000]
    
    df_val_ship = df_val[df_val['ship'] == 1]
    df_val_no_ship = df_val[df_val['ship'] == 0].iloc[:2000]

    if drop_empty:
        df_train = df_train_ship
        df_val = df_val_ship
    else:
        df_train = shuffle(df_train_ship.append(df_train_no_ship), random_state=1234)
        df_val = shuffle(df_val_ship.append(df_val_no_ship), random_state=1234)

    print(df_train.shape, df_val.shape)

    return df_train, df_val[:800]

def get_test_meta():
    return pd.read_csv(settings.SAMPLE_SUBMISSION, na_filter=False)

def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images

if __name__ == '__main__':
    train_meta, val_meta = get_train_val_meta()
    print(train_meta.shape)
    print(train_meta.head())
    print(val_meta.shape)
    print(val_meta.head())

    print(train_meta[train_meta['ship'] == 1].shape)
    print(train_meta[train_meta['ship'] == 0].shape)
    print(val_meta[val_meta['ship'] == 1].shape)
    print(val_meta[val_meta['ship'] == 0].shape)

