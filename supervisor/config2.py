import os

#BASE_DIR='/var/www/html/image/model6/'
NAME = "real_time"
BASE_DIR = '/extend/deploy_test'
MID_DIR = os.path.join(BASE_DIR,"mid_result")
RADAR_DIR = os.path.join(BASE_DIR,"18_2500_radar")
DISPLAY_DIR=os.path.join(BASE_DIR,"display_dir")
IN_PERIODS = 10
OUT_PERIODS = 11
PRED_LENGTH=10
OUT_LENGTH=10
IN_LENGTH=5

#OUT_DIR = os.path.join(BASE_DIR,"OUT")