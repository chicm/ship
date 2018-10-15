import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy import ndimage
import cv2

import settings
from loader import get_test_loader
from models import UNetShipV1
from postprocessing import binarize, resize_image
from utils import run_length_encoding

def do_tta_predict(args, model, ckp_path, tta_num=1):
    '''
    return 18000x128x128 np array
    '''
    model.eval()
    preds = []
    cls_preds = []
    meta = None

    # i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
    for flip_index in range(tta_num):
        print('flip_index:', flip_index)
        test_loader = get_test_loader(args.batch_size, index=flip_index, dev_mode=args.dev_mode)
        meta = test_loader.meta
        outputs = None
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                img = img.cuda()
                output, cls_output = model(img)
                output, cls_output = torch.sigmoid(output), torch.sigmoid(cls_output)
                if outputs is None:
                    outputs = output.squeeze().cpu()
                else:
                    outputs = torch.cat([outputs, output.squeeze().cpu()], 0)
                
                cls_preds.extend(cls_output.squeeze().cpu().tolist())

                print('{} / {}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
        outputs = outputs.numpy()
        # flip back masks
        if flip_index == 1:
            outputs = np.flip(outputs, 2)
        elif flip_index == 2:
            outputs = np.flip(outputs, 1)
        elif flip_index == 3:
            outputs = np.flip(outputs, 2)
            outputs = np.flip(outputs, 1)
        #print(outputs.shape)
        preds.append(outputs)
    
    parent_dir = ckp_path+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred.npy')

    model_pred_result = np.mean(preds, 0)
    np.save(np_file, model_pred_result)

    return model_pred_result, cls_preds, meta


def predict(args, model, checkpoint, out_file):
    print('predicting {}...'.format(checkpoint))
    mask_outputs, cls_preds, meta = do_tta_predict(args, model, checkpoint, tta_num=1)
    print(mask_outputs.shape)
    #print(len(cls_preds))
    #print(cls_preds)
    #print(meta.head(10))
    #y_pred_test = generate_preds(pred)
    print(meta.shape)

    ship_list_dict = []
    for i, row in enumerate(meta.values):
        img_id = row[0]
        if cls_preds[i] < 0.5:
            ship_list_dict.append({'ImageId': img_id,'EncodedPixels': np.nan})
        else:
            ship_rles = generate_preds(mask_outputs[i])
            if ship_rles:
                for ship_rle in ship_rles:
                    ship_list_dict.append({'ImageId': img_id,'EncodedPixels': ship_rle})
            else:
                ship_list_dict.append({'ImageId': img_id,'EncodedPixels': np.nan})

    pred_df = pd.DataFrame(ship_list_dict)
    pred_df.to_csv(args.sub_file, columns=['ImageId', 'EncodedPixels'], index=False)
    #submission = create_submission(meta, y_pred_test)
    #submission.to_csv(out_file, index=None, encoding='utf-8')


def split_mask(mask):
    threshold = 0.5
    threshold_obj = 30 #ignor predictions composed of "threshold_obj" pixels or less
    labled, n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(np.uint8)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result


def generate_preds(output, target_size=(settings.ORIG_H, settings.ORIG_W), threshold=0.5):
    pred_rles = []

    #print(output.shape)
    mask = resize_image(output, target_size=target_size)
    #pred = binarize(cropped, threshold)

    mask_objects = split_mask(mask)
    if mask_objects:
        for obj in mask_objects:
            #print('detected obj:',obj.shape)
            #print(obj.max())
            pred_rles.append(run_length_encoding(obj))
            #cv2.imshow('mask', obj*255)
            #cv2.waitKey(0)
        #pred = binarize(cropped, threshold)
        #preds.append(pred)

    return pred_rles

def predict_model(args):
    model = eval(args.model_name)(args.layers)
    
    if args.exp_name is None:
        model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    else:
        model_file = os.path.join(settings.MODEL_DIR, args.exp_name, model.name, 'best.pth')

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    else:
        raise ValueError('model file not found: {}'.format(model_file))
    model = model.cuda()
    predict(args, model, model_file, args.sub_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--model_name', default='UNetShipV1', type=str, help='')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--exp_name', default=None, type=str, help='exp name')
    parser.add_argument('--sub_file', default='sub_1.csv', type=str, help='submission file')
    parser.add_argument('--dev_mode', action='store_true')

    args = parser.parse_args()
    print(args)

    predict_model(args)
    #ensemble_predict(args)
