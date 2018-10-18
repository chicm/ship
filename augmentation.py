import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

from utils import reseed, from_pil, to_pil, ImgAug
import pdb

def _perspective_transform_augment_images(self, images, random_state, parents, hooks):
    result = images
    if not self.keep_size:
        result = list(result)

    matrices, max_heights, max_widths = self._create_matrices(
        [image.shape for image in images],
        random_state
    )

    for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
        warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
        if warped.ndim == 2 and images[i].ndim == 3:
            warped = np.expand_dims(warped, 2)
        if self.keep_size:
            h, w = images[i].shape[0:2]
            warped = ia.imresize_single_image(warped, (h, w))

        result[i] = warped

    return result


iaa.PerspectiveTransform._augment_images = _perspective_transform_augment_images


def get_affine_seq(pad_mode='reflect'):
    affine_seq = iaa.Sequential([
        # General
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5), 
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-20, 20),
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, mode=pad_mode #'reflect' #symmetric
        ),
        # Deformations
        #iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
        #iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
    ], random_order=True)
    return affine_seq

intensity_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)

brightness_seq =  iaa.Sequential([
    iaa.Multiply((0.8, 1.2)),
    iaa.Sometimes(0.3,
        iaa.GaussianBlur(sigma=(0, 0.5))
    )
], random_order=False)



def test_time_augmentation_transform(image, tta_parameters):
    if tta_parameters['ud_flip']:
        image = np.flipud(image)
    if tta_parameters['lr_flip']:
        image = np.fliplr(image)
    if tta_parameters['color_shift']:
        random_color_shift = reseed(intensity_seq, deterministic=False)
        image = random_color_shift.augment_image(image)
    image = rotate(image, tta_parameters['rotation'])
    return image


def test_time_augmentation_inverse_transform(image, tta_parameters):
    image = per_channel_rotation(image.copy(), -1 * tta_parameters['rotation'])

    if tta_parameters['lr_flip']:
        image = per_channel_fliplr(image.copy())
    if tta_parameters['ud_flip']:
        image = per_channel_flipud(image.copy())
    return image


def per_channel_flipud(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.flipud(channel)
    return x_


def per_channel_fliplr(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.fliplr(channel)
    return x_


def per_channel_rotation(x, angle):
    return rotate(x, angle, axes=(1, 2))


def rotate(image, angle, axes=(0, 1)):
    if angle % 90 != 0:
        raise Exception('Angle must be a multiple of 90.')
    k = angle // 90
    return np.rot90(image, k, axes=axes)


import os
import settings
from PIL import Image, ImageDraw
def test_augment():
    img = os.path.join(settings.TRAIN_IMG_DIR, '003c477d7c.png')
    mask = os.path.join(settings.TRAIN_MASK_DIR, '003c477d7c.png')
    img = Image.open(img)
    img = img.convert('RGB')
    mask = Image.open(mask)
    mask = mask.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
    print(type(mask))
    mask = from_pil(mask)
    Mi = [to_pil(mask == class_nr) for class_nr in [0, 1]]
    img, *Mi = from_pil(img, *Mi)

    aug = ImgAug(crop_seq(crop_size=(settings.H, settings.W), pad_size=(14,14), pad_method='edge'))
    aug2 = ImgAug(brightness_seq)
    img, *Mi = aug(img, *Mi)
    img = aug2(img)
    
    img, *Mi = to_pil(img, Mi[0]*255, Mi[1]*255)
    ImageDraw.Draw(img)
    ImageDraw.Draw(Mi[0])
    ImageDraw.Draw(Mi[1])
    img.show()
    Mi[0].show()
    Mi[1].show()

import torchvision.transforms.functional as F

class TTATransform(object):
    def __init__(self, index):
        self.index = index
    def __call__(self, img):
        trans = {
            0: lambda x: x,
            1: lambda x: F.hflip(x),
            2: lambda x: F.vflip(x),
            3: lambda x: F.vflip(F.hflip(x)),
            4: lambda x: F.rotate(x, 90, False, False),
            5: lambda x: F.hflip(F.rotate(x, 90, False, False)),
            6: lambda x: F.vflip(F.rotate(x, 90, False, False)),
            7: lambda x: F.vflip(F.hflip(F.rotate(x, 90, False, False)))
        }
        return trans[self.index](img)

# i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
def tta_back_mask_np(img, index):
    print(img.shape)
    trans = {
        0: lambda x: x,
        1: lambda x: np.flip(x, 2),
        2: lambda x: np.flip(x, 1),
        3: lambda x: np.flip(np.flip(x, 2), 1),
        4: lambda x: np.rot90(x, 3, axes=(1,2)),
        5: lambda x: np.rot90(np.flip(x, 2), 3, axes=(1,2)),
        6: lambda x: np.rot90(np.flip(x, 1), 3, axes=(1,2)),
        7: lambda x: np.rot90(np.flip(np.flip(x,2), 1), 3, axes=(1,2))
    }

    return trans[index](img)

def test_tta():
    img_f = os.path.join(settings.TEST_IMG_DIR, '0c2637aa9.jpg')
    img = Image.open(img_f)
    img = img.convert('RGB')

    tta_index = 7
    trans1 = TTATransform(tta_index)
    img = trans1(img)
    #img.show()

    img_np = np.array(img)
    img_np = np.expand_dims(img_np, 0)
    print(img_np.shape)
    img_np = tta_back_mask_np(img_np, tta_index)
    img_np = np.reshape(img_np, (768, 768, 3))
    img_back = F.to_pil_image(img_np)
    img_back.show()


def tta_4(img):
    return F.rotate(img, 90, False, False)

def tta_5(img):
    return F.hflip(tta_4(img))

def tta_6(img):
    return F.vflip(tta_4(img))

def tta_7(img):
    return F.vflip(F.hflip(tta_4(img)))

def tta_4_back(img):
    return F.rotate(img, 270, False, False)

def tta_5_back(img):
    return tta_4_back(F.hflip(img))

def tta_6_back(img):
    return tta_4_back(F.vflip(img))

def tta_7_back(img):
    return tta_4_back(F.vflip(F.hflip(img)))

def tta_back_np(img, tta_index):
    np_img = np.array(img)
    print(np_img.shape)

    trans = {
        4: lambda x: np.rot90(x, 3),
        5: lambda x: np.rot90(np.flip(x, 1), 3),
        6: lambda x: np.rot90(np.flip(x, 0), 3),
        7: lambda x: np.rot90(np.flip(np.flip(x,1), 0), 3)
    }
    np_img = trans[tta_index](np_img)

    return F.to_pil_image(np_img)


def test_rotate():
    img_f = os.path.join(settings.TEST_IMG_DIR, '0c2637aa9.jpg')
    img = Image.open(img_f)
    img = img.convert('RGB')
    #img_np = np.array(img)
    #img_np_r90 = np.rot90(img_np,1)
    #img_np_r90 = np.rot90(img_np_r90,3)
    #img_2 = F.to_pil_image(img_np_r90)
    #img = F.rotate(img, 90, False, False)
    #ImageDraw.Draw(img_2)
    #img_2.show()
    #img.show()

    img_aug = tta_7(img)
    #img_aug = tta_7_back(img_aug)
    img_aug = tta_back_np(img_aug, 7)
    img_aug.show()


if __name__ == '__main__':
    #test_augment()
    #test_rotate()
    test_tta()