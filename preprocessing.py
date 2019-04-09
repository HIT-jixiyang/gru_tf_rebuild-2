import numpy as np
import cv2
import pandas as pd
from image import quick_read_frames,get_480x480_clip
import config as c
import os

def crop_dataset(start,end,in_len,group_size):
    datetime_clip = pd.date_range(start=start,
                                  end=end,
                                  freq='6T')

    for i in range(in_len-1,len(datetime_clip)):
        # print(i)
        paths = []
        for j in range(i-c.IN_SEQ+1,i+1+c.OUT_SEQ):
            if j>=len(datetime_clip):
                break
            path = convert_datetime_to_filepath(datetime_clip[j])
            if not os.path.exists(path):
                break
            paths.append(path)
        if len(paths)!=c.IN_SEQ+c.OUT_SEQ:
            print(datetime_clip[i].strftime("%Y%m%d%H%M") + ' Miss')
            continue
        train_data = quick_read_frames(paths,700, 900)
        crop_data=get_480x480_clip(clip=train_data,size=group_size)
        if crop_data is None:
            print(datetime_clip[i].strftime("%Y%m%d%H%M")+' Too little pixels')
            continue
        dir = os.path.join(c.CROP_DATA_PATH,datetime_clip[i].strftime("%Y%m%d%H%M"))
        if not os.path.exists(dir):
            os.makedirs(dir)
        for k in range(0,group_size):
            save_in_dir=os.path.join(dir,'in',str(k+1))
            save_out_dir=os.path.join(dir,'out',str(k+1))
            if not os.path.exists(save_in_dir):
                os.makedirs(save_in_dir)
            if not os.path.exists(save_out_dir):
                os.makedirs(save_out_dir)
            save_crop_data(save_in_dir,save_out_dir, crop_data[k])

            # print('SAVE ---------',save_dir)
def save_crop_data(in_dir,out_dir,crop_datas):
    #[n,5,480,480]
    for i in range(c.IN_SEQ):
        path=os.path.join(in_dir,str(i+1)+'.png')
        image=crop_datas[i]
        cv2.imwrite(path,image)
    for i in range(c.OUT_SEQ):
        path=os.path.join(out_dir,str(i+1)+'.png')
        image=crop_datas[i+c.IN_SEQ]
        cv2.imwrite(path,image)
def convert_datetime_to_filepath(date_time):
    """Convert datetime to the filepath

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    # ret = os.path.join(cfg.REF_PATH, "cappi_ref_"+"%04d%02d%02d%02d%02d" %(
    #     date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute)
    #     +"_2500_0.ref")
    ret = os.path.join(c.REF_PATH, "cappi_ref_" + date_time.strftime("%Y%m%d%H%M")
                       + "_2500_0.png")
    return ret
if __name__ == '__main__':
    start='201502210600'
    # end='201502221900'
    end='201809172354'
    crop_dataset(start,end,c.IN_SEQ,2)