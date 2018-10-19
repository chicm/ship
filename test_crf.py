import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.utils import shuffle

from metrics import intersection_over_union_thresholds, intersection_over_union
import settings
#import matplotlib.pyplot as plt

TRAIN_META = '/mnt/g/ship/train_meta.csv'

def get_train_val_meta(drop_empty=False):
    df = pd.read_csv(TRAIN_META, na_filter=False)

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
#from postprocessing import get_val_result

def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))

def test_crf():
    loaded = np.load('/mnt/g/ship/tmp/val_out.npz')
    outputs = loaded['outputs']
    y_true = loaded['y_true']
    print(outputs.shape)
    print(y_true.shape)

    tgt_size = (settings.ORIG_H, settings.ORIG_W)

    resized = list(map(lambda x: resize(x, tgt_size), outputs))
    print(resized[0].shape, len(resized))
    y_pred = list(map(lambda x: (x > 0.5).astype(np.uint8), resized))
    print(y_pred[0].shape, len(y_pred))

    iou_score = intersection_over_union(y_true, y_pred)
    iout_score = intersection_over_union_thresholds(y_true, y_pred)
    print(iou_score, iout_score)

    _, val_meta = get_train_val_meta(True)
    img_ids = val_meta['ImageId'].values.tolist()
    crf_imgs = []
    for i, img_id in enumerate(img_ids):
        orig_img = imread('/mnt/g/ship/train_v2/{}'.format(img_id))        
        crf_output = crf(orig_img, y_pred[i])
        crf_imgs.append(crf_output)
        if i % 100 == 0:
            print(i)
    
    iou_score = intersection_over_union(y_true, crf_imgs)
    iout_score = intersection_over_union_thresholds(y_true, crf_imgs)
    print(iou_score, iout_score)

if __name__ == '__main__':
    #_, val_meta = get_train_val_meta(True)
    #print(val_meta.head(), val_meta.shape)
    test_crf()
