import os
import cv2
import numpy as np
import random
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, ColorJitter
import PIL
from PIL import Image
import collections

import settings


class RandomHFlipWithMask(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, *imgs):
        if random.random() < self.p:
            return map(F.hflip, imgs)
        else:
            return imgs

class RandomVFlipWithMask(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, *imgs):
        if random.random() < self.p:
            return map(F.vflip, imgs)
        else:
            return imgs

class RandomResizedCropWithMask(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(RandomResizedCropWithMask, self).__init__(size, scale, ratio, interpolation)
    def __call__(self, *imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        #print(i,j,h,w)
        return map(lambda x: F.resized_crop(x, i, j, h, w, self.size, self.interpolation), imgs)

class RandomRotateWithMask(object):
    def __init__(self, degrees, pad_mode='reflect', expand=False, center=None):
        self.pad_mode = pad_mode
        self.expand = expand
        self.center = center
        self.degrees = degrees

    def __call__(self, *imgs):
        angle = self.get_angle()
        if angle == int(angle) and angle % 90 == 0:
            if angle == 0:
                return imgs
            else:
                #print(imgs)
                return map(lambda x: F.rotate(x, angle, False, False, None), imgs)
        else:
            return map(lambda x: self._pad_rotate(x, angle), imgs)

    def get_angle(self):
        if isinstance(self.degrees, collections.Sequence):
            index = int(random.random() * len(self.degrees))
            return self.degrees[index]
        else:
            return random.uniform(-self.degrees, self.degrees)

    def _pad_rotate(self, img, angle):
        w, h = img.size
        img = F.pad(img, w//2, 0, self.pad_mode)
        img = F.rotate(img, angle, False, self.expand, self.center)
        img = F.center_crop(img, (w, h))
        return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

def test_transform():
    img_id = '0a48b7268.jpg'
    img = Image.open(os.path.join(settings.TRAIN_IMG_DIR, img_id)).convert('RGB')
    mask = Image.open(os.path.join(settings.TRAIN_MASK_DIR, img_id)).convert('L').point(lambda x: 0 if x < 128 else 1, 'L')

    #trans = RandomResizedCropWithMask(768, scale=(0.6, 1))
    trans = Compose([
        RandomHFlipWithMask(),
        RandomVFlipWithMask(),
        RandomRotateWithMask([0, 90, 180, 270]),
        #RandomRotateWithMask(15), 
        RandomResizedCropWithMask(768, scale=(0.81, 1))
    ])
    #trans = RandomRotateWithMask([0, 90, 180, 270])

    img, mask = trans(img, mask)

    img.show()
    mask.point(lambda x: x*255).show()

def test_color_trans():
    img_id = '00abc623a.jpg'
    img = Image.open(os.path.join(settings.TRAIN_IMG_DIR, img_id)).convert('RGB')
    trans = ColorJitter(0.1, 0.1, 0.1, 0.1)

    img2 = trans(img)
    img.show()
    img2.show()


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
    #test_tta()
    test_transform()
    #test_color_trans()