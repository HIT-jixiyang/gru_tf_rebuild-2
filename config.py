import os


#iterator
DATA_BASE_PATH = os.path.join("/extend", "sz17_data")
REF_PATH = os.path.join(DATA_BASE_PATH, "radarPNG_expand")

BASE_PATH = os.path.join("/extend", "gru_tf_data")
SAVE_PATH = os.path.join(BASE_PATH, "10_20_model")
CROP_DATA_PATH=os.path.join(BASE_PATH,'crop_data_10_20')
SAVE_MODEL = os.path.join(SAVE_PATH, "Save")
SAVE_VALID = os.path.join(SAVE_PATH, "Valid")
SAVE_TEST = os.path.join(SAVE_PATH, "Test")
SAVE_SUMMARY = os.path.join(SAVE_PATH, "Summary")

if not os.path.exists(SAVE_MODEL):
    os.makedirs(SAVE_MODEL)
if not os.path.exists(SAVE_VALID):
    os.makedirs(SAVE_VALID)


RAINY_TRAIN = ['201501010000', '201801010000']
RAINY_VALID = ['201801010000', '201809180000']
RAINY_TEST = ['201805110000', '201806080000']

#train
MAX_ITER = 150000
SAVE_ITER = 10000
VALID_ITER = 10000


# project
DTYPE = "single"
NORMALIZE = False

H = 720
W = 900
H_TRAIN=480
H_TEST=720
W_TRAIN=480
W_TEST=900

BATCH_SIZE = 2
IN_CHANEL = 1

# Encoder Forecaster
IN_SEQ = 10
OUT_SEQ = 20

LR = 0.0001

RESIDUAL = False

FIRST_CONV = (8, 5, 3, 1)              # pad
LAST_DECONV = (8, 5, 3, 1)


DOWNSAMPLE = [(3, 2),              # kernel stride
              (3, 2)]

UPSAMPLE = [(3, 2),
            (3, 2)]

TRAIN_FEATMAP_H = [160, 80, 40]
TRAIN_FEATMAP_W = [160, 80, 40]
TEST_FEATMAP_H = [240, 120, 60]
TEST_FEATMAP_W = [300, 150, 75]
I2H_KERNEL = [3, 3, 3]
H2H_KERNEL = [5, 5, 3]
NUM_FILTER = [32, 64, 64]

# EVALUATION

ZR_a = 58.53
ZR_b = 1.56

USE_BALANCED_LOSS = False
THRESHOLDS = [0.5, 2, 5, 10, 30]
BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]

TEMPORAL_WEIGHT_TYPE = "same"
TEMPORAL_WEIGHT_UPPER = 5

L1_LAMBDA = 0
L2_LAMBDA = 1.0
GDL_LAMBDA = 0

PREDICT_LENGTH = 20

MIN_PIXEL_NUM=2000
PRED_H=720
PRED_W=900