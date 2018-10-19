import os
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import run_length_decoding
from models import UNetShipV1, UNetShipV2
import settings

def create_train_meta():
    df_train_seg = pd.read_csv(settings.TRAIN_SHIP_SEGMENTATION, na_filter=False)
    df_train_seg = df_train_seg.groupby('ImageId')['EncodedPixels'].apply(' '.join).reset_index()
    df_train_seg['ship'] = df_train_seg.EncodedPixels.map(lambda x: 0 if len(x) == 0 else 1)
    df_train_seg = shuffle(df_train_seg, random_state=1234)
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

def convert_model_v1_v2():
    model_v1 = UNetShipV1(34)
    model_v2 = UNetShipV2(34)
    model_file = os.path.join(settings.MODEL_DIR, model_v1.name, 'best.pth')
    model_v1.load_state_dict(torch.load(model_file))

    model_v2.encoder1 = model_v1.encoder1
    model_v2.encoder2 = model_v1.encoder2
    model_v2.encoder3 = model_v1.encoder3
    model_v2.encoder4 = model_v1.encoder4
    model_v2.encoder5 = model_v1.encoder5

    model_v2.center = model_v1.center

    model_v2.decoder1 = model_v1.decoder1
    model_v2.decoder2 = model_v1.decoder2
    model_v2.decoder3 = model_v1.decoder3
    model_v2.decoder4 = model_v1.decoder4
    model_v2.decoder5 = model_v1.decoder5

    model_v2.logit_image = model_v1.logit_image

    model_file_v2 = os.path.join(settings.MODEL_DIR, model_v2.name, 'best.pth')
    torch.save(model_v2.state_dict(), model_file_v2)


if __name__ == '__main__':
    convert_model_v1_v2()
    #create_train_meta()
    #create_train_mask_imgs()