import os
import time
import numpy as np
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
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

def get_train_val_meta():
    df = pd.read_csv(settings.TRAIN_META, na_filter=False)

    split_index = df.shape[0] - 10000

    df_train = df.iloc[:split_index]

    df_val = df.iloc[split_index:]
    print(df_train.shape, df_val.shape)

    return df_train, df_val

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

def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed

def reseed(augmenter, deterministic=True):
    augmenter.random_state = ia.new_random_state(get_seed())
    if deterministic:
        augmenter.deterministic = True

    for lists in augmenter.get_children_lists():
        for aug in lists:
            aug = reseed(aug, deterministic=True)
    return augmenter

class ImgAug:
    def __init__(self, augmenters):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        self.augmenters = augmenters
        self.seq_det = None

    def _pre_call_hook(self):
        seq = iaa.Sequential(self.augmenters)
        seq = reseed(seq, deterministic=True)
        self.seq_det = seq

    def transform(self, *images):
        images = [self.seq_det.augment_image(image) for image in images]
        if len(images) == 1:
            return images[0]
        else:
            return images

    def __call__(self, *args):
        self._pre_call_hook()
        return self.transform(*args)

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

