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
from torchvision.models import resnet34
import pdb
import settings
from loader import get_train_val_loaders, get_test_loader
from lovasz_losses import lovasz_hinge, lovasz_softmax
from dice_losses import mixed_dice_bce_loss, FocalLoss2d
from postprocessing import binarize, resize_image
from metrics import intersection_over_union, intersection_over_union_thresholds
from postprocessing import split_mask, mask_to_bbox
from PIL import ImageDraw
import cv2
MODEL_DIR = settings.MODEL_DIR
focal_loss2d = FocalLoss2d() 


def create_model():
    resnet = resnet34(pretrained=True)
    resnet.fc = nn.Linear(2048, 1)
    resnet.name = 'classifier'
    return resnet

def test_model():
    x = torch.randn(2, 3, 256, 256).cuda()
    model = create_model().cuda()
    y = model(x)
    print(y.size())

def train(args):
    print('start training...')

    model = create_model()
    model_file = os.path.join(MODEL_DIR, model.name, 'best.pth')

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
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, dev_mode=args.dev_mode, drop_empty=False, img_sz=args.img_sz)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    print('epoch |   lr    |   %        |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  ship  |')

    best_iout, _iou, _f, _l, _b, _ship, best_cls_acc = validate(args, model, val_loader, args.start_epoch)
    print('val   |         |            |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      | {:.4f} |'.format(
        _f, _l, _b, _ship, _iou, best_iout, best_cls_acc, best_cls_acc))
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
            salt_out = model(img)
            
            loss = F.binary_cross_entropy_with_logits(salt_out.squeeze(), salt_target)
            loss.backward()
 
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                iout, iou, focal_loss, lovaz_loss, bce_loss, cls_loss, cls_acc = validate(args, model, val_loader, epoch=epoch)
                
                _save_ckp = ''
                if cls_acc > best_cls_acc:
                    best_cls_acc = cls_acc
                    torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                # print('epoch |   lr    |   %       |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  ship  |')
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
                    focal_loss, lovaz_loss, bce_loss, cls_loss, iou, iout, best_cls_acc, (time.time() - bg) / 60, _save_ckp, cls_acc))

                #log.info('epoch {}: train loss: {:.4f} focal loss: {:.4f} lovaz loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {} {}'
                #    .format(epoch, train_loss, focal_loss, lovaz_loss, iout, best_iout, iou, current_lr, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(cls_acc)
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
    total_num = 0
    cls_corrects = 0
    ship_loss = 0
    with torch.no_grad():
        for img, target, ship_target in val_loader:
            img, target, ship_target = img.cuda(), target.cuda(), ship_target.cuda()
            ship_out = model(img)
            #print(output.size(), salt_out.size())
            loss  = F.binary_cross_entropy_with_logits(ship_out.squeeze(), ship_target)
            ship_loss += loss.item()

            ship_pred = (torch.sigmoid(ship_out) > cls_threshold).byte().squeeze()
            total_num += len(img)
            cls_corrects += ship_pred.eq(ship_target.byte()).sum().item()

            
    cls_acc = cls_corrects / total_num
    #print('total num:', total_num)
    #print('corrects:', cls_corrects)
    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    return 0, 0, 0, 0, ship_loss / n_batches, ship_loss/ n_batches, cls_acc

def pred_class(args):
    model = create_model()
    model_file = os.path.join(MODEL_DIR, model.name, 'best.pth')

    if not os.path.exists(model_file):
        raise AssertionError('{} does not exist'.format(model_file))
    print('loading {}...'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()

    preds = []

    test_loader = get_test_loader(64, 0, img_sz=256)
    for img in test_loader:
        img = img.cuda()
        output = model(img).squeeze()
        pred = (torch.sigmoid(output) > 0.5).byte().cpu().tolist()
        preds.extend(pred)
    
    df_test = test_loader.meta
    df_test['Target'] = preds
    print(sum(preds))
    df_test.to_csv('test_cls_preds.csv', index=False, columns=['patientId', 'Target'])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=96, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=100, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--img_sz', default=256, type=int, help='image size')
    parser.add_argument('--pred', action='store_true')
    
    args = parser.parse_args()
    print(args)

    if args.pred:
        pred_class(args)
    else:
        train(args)
    #test_model()
