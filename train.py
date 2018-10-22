import os
import argparse
import numpy as np
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import settings
from loader import get_train_val_loaders
from models import UNetShipV1, UNetShipV2
from lovasz_losses import lovasz_hinge, lovasz_softmax
from dice_losses import mixed_dice_bce_loss, FocalLoss2d
from postprocessing import binarize, resize_image
from metrics import intersection_over_union, intersection_over_union_thresholds
from postprocessing import split_mask, mask_to_bbox
from PIL import ImageDraw
import cv2
MODEL_DIR = settings.MODEL_DIR
focal_loss2d = FocalLoss2d()

def criterion(args, output, target, epoch=0):
    mask_output, ship_output = output
    mask_target, ship_target = target

    #dice_loss = mixed_dice_bce_loss(mask_output, mask_target)
    focal_loss = focal_loss2d(mask_output, mask_target)
    #lovasz_loss = lovasz_hinge(mask_output, mask_target)

    lovasz_loss = (lovasz_hinge(mask_output, mask_target) + lovasz_hinge(-mask_output, 1 - mask_target)) / 2

    bce_loss = F.binary_cross_entropy_with_logits(mask_output, mask_target)
    cls_loss = F.binary_cross_entropy_with_logits(ship_output, ship_target)

    if args.train_cls:
        #cls_loss = F.binary_cross_entropy_with_logits(ship_output, ship_target)
        return lovasz_loss + bce_loss + cls_loss, focal_loss.item(), lovasz_loss.item(), bce_loss.item(), cls_loss.item()

    # four losses for: 1. grad, 2, display, 3, display 4, measurement
    #if epoch < 10:
    #    return bce_loss, focal_loss.item(), lovasz_loss.item(), 0., lovasz_loss.item() + focal_loss.item()*focal_weight
    #else:
        #return lovasz_loss+focal_loss*focal_weight, focal_loss.item(), lovasz_loss.item(), 0., lovasz_loss.item() + focal_loss.item()*focal_weight
    return lovasz_loss + bce_loss*0.1, focal_loss.item(), lovasz_loss.item(), bce_loss.item(), cls_loss.item()


def train(args):
    print('start training...')

    model = eval(args.model_name)(args.layers)

    if args.exp_name is None:
        model_file = os.path.join(MODEL_DIR, model.name, 'best.pth')
    else:
        model_file = os.path.join(MODEL_DIR, args.exp_name, model.name, 'best.pth')

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.init_ckp is not None:
        CKP = args.init_ckp
    else:
        CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, dev_mode=args.dev_mode, drop_empty=not args.train_cls, img_sz=args.img_sz)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    print('epoch |   lr    |   %        |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  ship  |')

    best_iout, _iou, _f, _l, _b, _ship, cls_acc = validate(args, model, val_loader, args.start_epoch)
    print('val   |         |            |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      | {:.4f} |'.format(
        _f, _l, _b, _ship, _iou, best_iout, best_iout, cls_acc))
    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_iout)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target, salt_target = data
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            optimizer.zero_grad()
            output, salt_out = model(img)
            
            loss, *_ = criterion(args, (output, salt_out), (target, salt_target), epoch=epoch)
            loss.backward()
 
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                iout, iou, focal_loss, lovaz_loss, bce_loss, ship_loss, ship_acc = validate(args, model, val_loader, epoch=epoch)
                
                _save_ckp = ''
                if iout > best_iout:
                    best_iout = iout
                    torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                # print('epoch |   lr    |   %       |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  ship  |')
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
                    focal_loss, lovaz_loss, bce_loss, ship_loss, iou, iout, best_iout, (time.time() - bg) / 60, _save_ckp, ship_acc))

                #log.info('epoch {}: train loss: {:.4f} focal loss: {:.4f} lovaz loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {} {}'
                #    .format(epoch, train_loss, focal_loss, lovaz_loss, iout, best_iout, iou, current_lr, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(iout)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    del model, train_loader, val_loader, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader, epoch=0, threshold=0.5, cls_threshold=0.5):
    model.eval()
    #print('validating...')
    outputs = []
    cls_preds = []
    total_num = 0
    cls_corrects = 0
    focal_loss, lovaz_loss, bce_loss, ship_loss = 0, 0, 0, 0
    with torch.no_grad():
        for img, target, ship_target in val_loader:
            img, target, ship_target = img.cuda(), target.cuda(), ship_target.cuda()
            output, ship_out = model(img)
            #print(output.size(), salt_out.size())
            ship_pred = (torch.sigmoid(ship_out) > cls_threshold).byte()
            total_num += len(img)
            cls_corrects += ship_pred.eq(ship_target.byte()).sum().item()

            _, floss, lovaz, _bce_loss, _ship_loss = criterion(args, (output, ship_out), (target, ship_target), epoch=epoch)
            focal_loss += floss
            lovaz_loss += lovaz
            ship_loss += _ship_loss
            bce_loss += _bce_loss
            output = torch.sigmoid(output)
            
            for o in output.cpu():
                outputs.append(o.squeeze().numpy())
            cls_preds.extend(ship_pred.squeeze().cpu().numpy().tolist())

    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    # y_pred, list of 400 np array, each np array's shape is 101,101
    y_pred = generate_preds(args, outputs, cls_preds, (settings.ORIG_H, settings.ORIG_W), threshold)

    #draw
    if args.dev_mode:
        for p, y in zip(y_pred, val_loader.y_true):
            print(p.shape, y.shape)
            objs = split_mask(p, threshold_obj=30, threshold=None)
            if objs:
                if False:
                    objs = map(lambda x: mask_to_bbox(x), objs)
                cv2.imshow('image', np.hstack([*objs, y])*255)
            else:
            #bb_img = masks_to_bounding_boxes(p)
            #bb_img = (bb_img > 0).astype(np.uint8)
            #print(bb_img.max())
                cv2.imshow('image', np.hstack([p, y])*255)
            cv2.waitKey(0)


    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    #print('IOU score on validation is {:.4f}'.format(iou_score))
    #print('IOUT score on validation is {:.4f}'.format(iout_score))

    cls_acc = cls_corrects / total_num

    return iout_score, iou_score, focal_loss / n_batches, lovaz_loss / n_batches, bce_loss / n_batches, ship_loss / n_batches, cls_acc


def generate_preds(args, outputs, cls_preds, target_size, threshold=0.5):
    preds = []

    for i, output in enumerate(outputs):
        resized_img = resize_image(output, target_size=target_size)
        pred = binarize(resized_img, threshold)
        if args.train_cls:
            pred = pred*cls_preds[i]
        preds.append(pred)

    return preds

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=200, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--exp_name', default=None, type=str, help='exp name')
    parser.add_argument('--model_name', default='UNetShipV1', type=str, help='')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--img_sz', default=384, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)

    #log.basicConfig(
    #    filename = 'trainlog_{}.txt'.format(''.join([str(x) for x in ifolds])), 
    #    format   = '%(asctime)s : %(message)s',
    #    datefmt  = '%Y-%m-%d %H:%M:%S', 
    #    level = log.INFO)

    train(args)
