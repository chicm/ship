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


def padding_seq(pad_size, pad_method):
    seq = iaa.Sequential([PadFixed(pad=pad_size, pad_method=pad_method),
                          ]).to_deterministic()
    return seq


class PadFixed(iaa.Augmenter):
    PAD_FUNCTION = {'reflect': cv2.BORDER_REFLECT_101,
                    'edge': cv2.BORDER_REPLICATE,
                    }

    def __init__(self, pad=None, pad_method=None, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        self.pad = pad
        self.pad_method = pad_method

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for i, image in enumerate(images):
            image_pad = self._pad(image)
            result.append(image_pad)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _pad(self, img):
        img_ = img.copy()

        if self._is_expanded_grey_format(img):
            img_ = np.squeeze(img_, axis=-1)

        h_pad, w_pad = self.pad
        img_ = cv2.copyMakeBorder(img_.copy(), h_pad, h_pad, w_pad, w_pad, PadFixed.PAD_FUNCTION[self.pad_method])

        if self._is_expanded_grey_format(img):
            img_ = np.expand_dims(img_, axis=-1)

        return img_

    def get_parameters(self):
        return []

    def _is_expanded_grey_format(self, img):
        if len(img.shape) == 3 and img.shape[2] == 1:
            return True
        else:
            return False


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


class RandomCropFixedSize(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(RandomCropFixedSize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
#        pdb.set_trace()
        self.px = px
        if isinstance(self.px, tuple):
            self.px_h, self.px_w = self.px
        elif isinstance(self.px, int):
            self.px_h = self.px
            self.px_w = self.px
        else:
            raise NotImplementedError

    def _augment_images(self, images, random_state, parents, hooks):
        #pdb.set_trace()
        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            image_cr = self._random_crop(seed, image)
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _random_crop(self, seed, image):
        height, width = image.shape[:2]
        #print('**_random_crop:', height, width)

        np.random.seed(seed)
        if height > self.px_h:
            crop_top = np.random.randint(height - self.px_h)
        elif height == self.px_h:
            crop_top = 0
        else:
            raise ValueError("To big crop height")
        crop_bottom = crop_top + self.px_h

        np.random.seed(seed + 1)
        if width > self.px_w:
            crop_left = np.random.randint(width - self.px_w)
        elif width == self.px_w:
            crop_left = 0
        else:
            raise ValueError("To big crop width")
        crop_right = crop_left + self.px_w

        if len(image.shape) == 2:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]
        return image_cropped

    def get_parameters(self):
        return []

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
if __name__ == '__main__':
    test_augment()