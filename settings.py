import os
import local_settings

ORIG_H, ORIG_W = 768, 768

DATA_DIR = local_settings.DATA_DIR
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_v2')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_v2')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks')

SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'sample_submission_v2.csv')
TRAIN_SHIP_SEGMENTATION = os.path.join(DATA_DIR, 'train_ship_segmentations_v2.csv')

TRAIN_META = os.path.join(DATA_DIR, 'train_meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')
