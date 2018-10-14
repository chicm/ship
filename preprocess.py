import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import run_length_decoding
import settings

def create_train_meta():
    df_train_seg = pd.read_csv(settings.TRAIN_SHIP_SEGMENTATION, na_filter=False)
    df_train_seg = df_train_seg.groupby('ImageId')['EncodedPixels'].apply(' '.join).reset_index()
    df_train_seg['ship'] = df_train_seg.EncodedPixels.map(lambda x: 0 if len(x) == 0 else 1)
    df_train_seg = shuffle(df_train_seg)
    df_train_seg.to_csv(settings.TRAIN_META, index=False, columns=['ImageId', 'ship'])

def create_train_mask_imgs():
    df = pd.read_csv(settings.TRAIN_SHIP_SEGMENTATION, na_filter=False)
    df = df.groupby('ImageId')['EncodedPixels'].apply(' '.join).reset_index()

    for i, row in enumerate(df.values):
        decoded_mask = run_length_decoding(row[1], (768,768))
        filename = os.path.join(settings.TRAIN_MASK_DIR, row[0])
        #rgb_mask = cv2.cvtColor(decoded_mask,cv2.COLOR_GRAY2RGB)
        #print(filename)
        cv2.imwrite(filename, decoded_mask)
        if i % 100 == 0:
            print(i)

if __name__ == '__main__':
    create_train_meta()
    create_train_mask_imgs()