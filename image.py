# Python plugin that supports loading batch of images in parallel
import numpy
import os
from concurrent.futures import ThreadPoolExecutor, wait
import cv2
import numpy as np
import config as c


_imread_executor_pool = ThreadPoolExecutor(max_workers=16)


def read_img(path, read_storage, im_h, im_w):
    # read_storage[:] = numpy.fromfile(path, dtype=numpy.uint8).reshape(im_h, im_w)
    # read_storage[:] = cv2.imread(path, 0)
    if im_h==720:
        read_storage[:] = cv2.imread(path, 0)[90:-90]
    else:
        read_storage[:] = cv2.imread(path, 0)

def get_480x480_clip(clip,size=1,mode='train'):
    if mode=='train':
        train_clips = []
        n=0
        iterator=0
        # find_flag=False
        while n<size:  # cap at 100 trials in case the clip has no movement anywhere
            use_flag = True
            crop_x = np.random.choice(c.W - c.W_TRAIN + 1)
            crop_y = np.random.choice(c.H - c.H_TRAIN + 1)
            temp=clip[:,crop_y:crop_y + c.H_TRAIN, crop_x:crop_x + c.W_TRAIN,:]
            for p in temp[0:c.IN_SEQ]:
                if len(p[p>0])<c.MIN_PIXEL_NUM:
                    use_flag=False
                    break
            if use_flag:
                train_clips.append(temp)
                n=n+1
            iterator=iterator+1
            if iterator>=100:
                return None
                break
        train_clips=np.array(train_clips)
        return np.reshape(train_clips,(size,c.IN_SEQ+c.OUT_SEQ,c.H_TRAIN,c.W_TRAIN,1))
    # if mode=='online':
    #     train_clips = []
    #     crop_x=[0,0,220,220]
    #     crop_y=[0,220,0,220]
    #     for i in range(0,4):
    #         temp = clip[:, crop_y[i]:crop_y[i] + c.H_TRAIN, crop_x[i]:crop_x[i] + c.W_TRAIN, :]
    #         train_clips.append(temp)
    #     train_clips=np.array(train_clips)
    #     return train_clips.reshape([4,5,480,480,1])

def quick_read_frames(path_list, im_h, im_w):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    im_h : height of image
    im_w : width of image

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            print(path_list[i])
            raise IOError

    read_storage = numpy.empty((img_num, im_h, im_w), dtype=numpy.uint8)

    if img_num == 1:
        read_img(path_list[0], read_storage[0], im_h, im_w)
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(read_img, path_list[i], read_storage[i], im_h, im_w)
            future_objs.append(obj)
        wait(future_objs)

    read_storage = read_storage.reshape((img_num, im_h, im_w, c.IN_CHANEL))
    # train_data=get_480x480_clip(read_storage)
    return read_storage
if __name__ == '__main__':
    path_list=['/extend/sz17_data/radarPNG_expand/cappi_ref_201502210600_2500_0.png',
               '/extend/sz17_data/radarPNG_expand/cappi_ref_201502210606_2500_0.png',
               '/extend/sz17_data/radarPNG_expand/cappi_ref_201502210612_2500_0.png',
               '/extend/sz17_data/radarPNG_expand/cappi_ref_201502210618_2500_0.png',
               '/extend/sz17_data/radarPNG_expand/cappi_ref_201502210624_2500_0.png']
    train_data=quick_read_frames(path_list,720,900)