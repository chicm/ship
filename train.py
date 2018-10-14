import os
import argparse
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import settings
from loader import get_train_loaders, add_depth_channel
from unet_models import UNetResNet, UNetResNetAtt, UNetResNetV3
from unet_new import UNetResNetV4, UNetResNetV5, UNetResNetV6, UNet7, UNet8
from unet_se import UNetResNetSE
from lovasz_losses import lovasz_hinge, lovasz_softmax
from dice_losses import mixed_dice_bce_loss, FocalLoss2d
from postprocessing import crop_image, binarize, crop_image_softmax, resize_image
from metrics import intersection_over_union, intersection_over_union_thresholds

MODEL_DIR = settings.MODEL_DIR
focal_loss2d = FocalLoss2d()

class CyclicExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, init_lr, min_lr=5e-7, restart_max_lr=1e-5, last_epoch=-1):
        self.gamma = gamma
        self.last_lr = init_lr
        self.min_lr = min_lr
        self.restart_max_lr = restart_max_lr
        super(CyclicExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.last_lr * self.gamma
        if lr < self.min_lr:
            lr = self.restart_max_lr
        self.last_lr = lr
        return [lr]*len(self.base_lrs)

def weighted_loss(args, output, target, epoch=0):
    mask_output, salt_output = output
    mask_target, salt_target = target

    lovasz_loss = lovasz_hinge(mask_output, mask_target)
    #dice_loss = mixed_dice_bce_loss(mask_output, mask_target)
    focal_loss = focal_loss2d(mask_output, mask_target)

    focal_weight = 0.2

    if salt_output is not None and args.train_cls:
        salt_loss = F.binary_cross_entropy_with_logits(salt_output, salt_target)
        return salt_loss, focal_loss.item(), lovasz_loss.item(), salt_loss.item(), lovasz_loss.item() + focal_loss.item()*focal_weight

    # four losses for: 1. grad, 2, display, 3, display 4, measurement
    if epoch < 0:
        return focal_loss, focal_loss.item(), lovasz_loss.item(), 0., lovasz_loss.item() + focal_loss.item()*focal_weight
    else:
        return lovasz_loss+focal_loss*focal_weight, focal_loss.item(), lovasz_loss.item(), 0., lovasz_loss.item() + focal_loss.item()*focal_weight

def train(args):
    print('start training...')

    model = eval(args.model_name)(args.layers, num_filters=args.nf)
    model_subdir = args.pad_mode
    if args.meta_version == 2:
        model_subdir = args.pad_mode+'_meta2'
    if args.exp_name is None:
        model_file = os.path.join(MODEL_DIR, model.name,model_subdir, 'best_{}.pth'.format(args.ifold))
    else:
        model_file = os.path.join(MODEL_DIR, args.exp_name, model.name, model_subdir, 'best_{}.pth'.format(args.ifold))

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

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=args.batch_size, dev_mode=False, pad_mode=args.pad_mode, meta_version=args.meta_version, pseudo_label=args.pseudo)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    print('epoch |   lr    |   %       |  loss  |  avg   | f loss | lovaz  |  iou   | iout   |  best  | time | save |  salt  |')

    best_iout, _iou, _f, _l, _salt, best_mix_score = validate(args, model, val_loader, args.start_epoch)
    print('val   |         |           |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      | {:.4f} |'.format(
        _f, _l, _iou, best_iout, best_iout, _salt))
    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_iout)
    else:
        lr_scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        #if epoch < 5:
        #    model.freeze_bn()
        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            img, target, salt_target = data
            add_depth_channel(img, args.pad_mode)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            optimizer.zero_grad()
            output, salt_out = model(img)
            
            loss, *_ = weighted_loss(args, (output, salt_out), (target, salt_target), epoch=epoch)
            loss.backward()
 
            # adamW
            #wd = 0.0001
            #for group in optimizer.param_groups:
            #    for param in group['params']:
            #        param.data = param.data.add(-wd * group['lr'], param.data)

            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

        iout, iou, focal_loss, lovaz_loss, salt_loss, mix_score = validate(args, model, val_loader, epoch=epoch)
        
        _save_ckp = ''
        if iout > best_iout:
            best_iout = iout
            torch.save(model.state_dict(), model_file)
            _save_ckp = '*'
        if args.store_loss_model and mix_score > best_mix_score:
            best_mix_score = mix_score
            torch.save(model.state_dict(), model_file+'_loss')
            _save_ckp += '.'
        # print('epoch |   %       |  loss  |  avg   | f loss | lovaz  |  iou   | iout   |  best  |   lr    | time | save |')
        print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
            focal_loss, lovaz_loss, iou, iout, best_iout, (time.time() - bg) / 60, _save_ckp, salt_loss))

        log.info('epoch {}: train loss: {:.4f} focal loss: {:.4f} lovaz loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {} {}'
            .format(epoch, train_loss, focal_loss, lovaz_loss, iout, best_iout, iou, current_lr, _save_ckp))

        model.train()
        
        if args.lrs == 'plateau':
            lr_scheduler.step(best_iout)
        else:
            lr_scheduler.step()

    del model, train_loader, val_loader, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader, epoch=0, threshold=0.5):
    model.eval()
    #print('validating...')
    outputs = []
    focal_loss, lovaz_loss, salt_loss, w_loss = 0, 0, 0, 0
    with torch.no_grad():
        for img, target, salt_target in val_loader:
            add_depth_channel(img, args.pad_mode)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            output, salt_out = model(img)
            #print(output.size(), salt_out.size())

            _, floss, lovaz, _salt_loss, _w_loss = weighted_loss(args, (output, salt_out), (target, salt_target), epoch=epoch)
            focal_loss += floss
            lovaz_loss += lovaz
            salt_loss += _salt_loss
            w_loss += _w_loss
            output = torch.sigmoid(output)
            
            for o in output.cpu():
                outputs.append(o.squeeze().numpy())

    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    # y_pred, list of 400 np array, each np array's shape is 101,101
    y_pred = generate_preds_softmax(args, outputs, (settings.ORIG_H, settings.ORIG_W), threshold)

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    #print('IOU score on validation is {:.4f}'.format(iou_score))
    #print('IOUT score on validation is {:.4f}'.format(iout_score))

    return iout_score, iou_score, focal_loss / n_batches, lovaz_loss / n_batches, salt_loss / n_batches, iout_score*4 - w_loss

def find_threshold(args):
    #ckp = r'G:\salt\models\152\ensemble_822\best_3.pth'
    ckp = r'D:\data\salt\models\UNetResNetV4_34\best_0.pth'
    model = UNetResNetV4(34)
    model.load_state_dict(torch.load(ckp))
    model = model.cuda()
    #criterion = lovasz_hinge
    _, val_loader = get_train_loaders(0, batch_size=args.batch_size, dev_mode=False)

    best, bestt = 0, 0.
    for t in range(40, 55, 1):
        print('threshold:', t/100.)
        iout, _, _ = validate(args, model, val_loader, epoch=10, threshold=t/100.)
        if iout > best:
            best = iout
            bestt = t/100.
    print('best:', best, bestt)

def generate_preds(outputs, target_size, threshold=0.5):
    preds = []

    for output in outputs:
        cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds

def generate_preds_softmax(args, outputs, target_size, threshold=0.5):
    preds = []

    for output in outputs:
        #print(output.shape)
        if args.pad_mode == 'resize':
            cropped = resize_image(output, target_size=target_size)
        else:
            cropped = crop_image_softmax(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--ifolds', default='0', type=str, help='kfold indices')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--pad_mode', default='edge', choices=['reflect', 'edge', 'resize'], help='pad method')
    parser.add_argument('--exp_name', default='depths', type=str, help='exp name')
    parser.add_argument('--model_name', default='UNetResNetV4', type=str, help='')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--store_loss_model', action='store_true')
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--meta_version', default=1, type=int, help='meta version')
    parser.add_argument('--pseudo', action='store_true')
    
    args = parser.parse_args()

    print(args)
    ifolds = [int(x) for x in args.ifolds.split(',')]
    print(ifolds)
    log.basicConfig(
        filename = 'trainlog_{}.txt'.format(''.join([str(x) for x in ifolds])), 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)

    for i in ifolds:
        args.ifold = i
        train(args)
