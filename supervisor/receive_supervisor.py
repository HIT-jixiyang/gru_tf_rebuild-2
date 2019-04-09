#!/usr/bin/env python
# encoding:utf-8
import logging
import os
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
import supervisor.color_map
import supervisor.config2 as cfg
from supervisor.color_map import transfer, multi_process_transfer,array2RGB,array2GRAY
from pyinotify import WatchManager, Notifier, \
    ProcessEvent,   IN_DELETE, IN_CREATE, IN_MODIFY
from deploy import Deploy
from supervisor.utils2 import auto_mail
def config_log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='/extend/deploy-data/deploy-sunny.log',
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
def get_in_refs(file_name,file_dir):
    """
    :param file_name:
    :param file_dir:
    :return:
    """
    miss_dates=[]
    error_dates=[]
    strs=file_name.split('_')
    current_date=strs[2]
    current_radar_num=int(strs[4][0])
    refs=[]
    dates = pd.date_range(end=current_date, periods=cfg.IN_PERIODS, freq='6T')
    pred_dir=os.path.join(cfg.MID_DIR,current_date,'pred')

    display_pred_dir=os.path.join(cfg.DISPLAY_DIR,current_date,'pred')
    if not os.path.exists(display_pred_dir):
        os.makedirs(display_pred_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    for i in range(0,cfg.IN_PERIODS):
        step = 1
        flag=1
        date=dates[i].strftime('%Y%m%d%H%M')
        new_file_name=file_name.replace(current_date,date)
        new_file=os.path.join(file_dir,new_file_name)
        new_radar_num = current_radar_num
        flag_6=False
        flag_11=False
        while True:
            if new_radar_num==-1 :
                flag_6=True
            if new_radar_num==20:
                flag_11=True
            if flag_6 and flag_11 and (new_radar_num<6 or new_radar_num>11):
                logging.warning('Miss Frame:'+new_file)
                miss_dates.append(new_file_name.split('_')[2])
                break
            if os.path.exists(new_file):
                try:
                    pad_ref=np.zeros([900,900],dtype=np.uint8)
                    ref = np.fromfile(new_file, dtype=np.uint8).reshape(700, 900)
                    ref[ref <= 15] = 0
                    ref[ref >= 80] = 0
                    pad_ref[100:-100,:]=ref
                    refs.append(pad_ref)
                    print('collect '+new_file)
                    break
                except:
                    logging.warning('Error Ref :' + new_file)
                    error_dates.append(new_file)
                    old_radar_num = new_radar_num
                    new_radar_num = current_radar_num + step * flag
                    miss_dates.append(new_file_name.split('_')[2])
                    if flag < 0:
                        step = step + 1
                    flag = -1 * flag
                    new_file_name = new_file_name.replace(str(old_radar_num) + '.ref', str(new_radar_num) + '.ref')
                    new_file = os.path.join(file_dir, new_file_name)
            else:
                old_radar_num=new_radar_num
                new_radar_num=current_radar_num+step*flag
                if flag<0:
                    step=step+1
                flag = -1 * flag
                new_file_name = new_file_name.replace(str(old_radar_num)+'.ref',str(new_radar_num)+'.ref')
                new_file=os.path.join(file_dir,new_file_name)
    if len(miss_dates)>0 or len(error_dates)>0:
        auto_mail(1,'Miss'+' '.join(miss_dates)+'  Error:'+' '.join(error_dates))
    return np.array(refs)
#
def send2out_in(file_name, in_refs):
    """
    transfer in_refs to colored-pngs and store them to display-dir/date/in
    transfer current input ref to gray-pngs and store it to midresult-dir/date/out
    :param file_name:file name of the input ref
    :param in_refs:numpy array for GRU prediction(shape:[10,900,900])
    :return:
    """
    strs = file_name.split('_')
    current_date = strs[2]
    out_dates = pd.date_range(end=current_date, periods=cfg.OUT_PERIODS, freq='6T')
    in_dates = pd.date_range(end=current_date, periods=cfg.IN_PERIODS, freq='6T')
    out_paths = []
    in_paths = []
    original_out_paths = []
    original_in_paths = []

    ## generate out_paths to sending
    for i in range(0,len(out_dates)-1):
        date = out_dates[i].strftime("%Y%m%d%H%M")
        out_dir = os.path.join(cfg.DISPLAY_DIR,date,'out')
        original_out_dir = os.path.join(cfg.MID_DIR, date,'out')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(original_out_dir):
            os.makedirs(original_out_dir)
        out_paths.append(os.path.join(out_dir, '{}.png'.format(cfg.OUT_LENGTH-i)))
        original_out_paths.append(os.path.join(original_out_dir, '{}.png'.format(cfg.OUT_LENGTH-i)))
    # generate in_paths to send
    in_dir = os.path.join(cfg.DISPLAY_DIR,out_dates[-1].strftime("%Y%m%d%H%M"),'in')
    original_in_dir = os.path.join(cfg.MID_DIR,out_dates[-1].strftime("%Y%m%d%H%M"),'in')
    for i in range(0, len(in_dates)):
        if not os.path.exists(in_dir):
            os.makedirs(in_dir)
        if not os.path.exists(original_in_dir):
            os.makedirs(original_in_dir)
        in_paths.append(os.path.join(in_dir, '{}.png'.format(i + 1)))
        original_in_paths.append(os.path.join(original_in_dir, '{}.png'.format(i + 1)))

    p = Pool()
    for i in range(len(in_paths)):
        result=p.apply_async(array2RGB, args=(in_refs[i], in_paths[i]))
            # logging.error('save error : from: '+in_refs[i]+' to: '+in_paths[i])
    for i in range(len(out_paths)):
        result=p.apply_async(array2RGB, args=(in_refs[-1], out_paths[i]))
    for i in range(len(original_in_paths)):
        result=p.apply_async(array2GRAY, args=(in_refs[i], original_in_paths[i]))
    for i in range(len(original_out_paths)):
        result=p.apply_async(array2GRAY, args=(in_refs[-1], original_out_paths[i]))
    print('pool end')
    p.close()
    p.join()


def save_pred(predict,current_date):
    """
    execute histogram and corlor operations,and save the result to display-dir,
    :param predict:the predict result of the model
    :param current_date:
    :return:
    """
    result = predict[0]
    for i in range(0, cfg.PRED_LENGTH):
        pred=result[i]
        pred[pred<0]=0
        pred=pred.astype(np.uint8)
        path=os.path.join(cfg.MID_DIR,current_date,'pred',str(i+1)+'.png')
        array2GRAY(pred.reshape(720, 900),path)
        # logging.info('SAVE PRED:'+path)
    pred_dir=os.path.join(cfg.MID_DIR,current_date)
    os.system(r'../post_processing/postprocessing'+' '+pred_dir)
    pred_display_dir=os.path.join(cfg.DISPLAY_DIR,current_date,'pred')
    multi_process_transfer(os.path.join(pred_dir,'pred'),pred_display_dir)
    logging.info('Transfer End------ DATE:'+current_date)

class EventHandler(ProcessEvent):
    def __init__(self,predict):
        self.predict=predict
    """事件处理"""
    def process_IN_CREATE(self, event):
        print(event.path,'-----', event.name)
        logging.info('A new ref has been created:'+event.path+'-----'+event.name)
        time.sleep(3)
        strs = event.name.split('_')
        current_date = strs[2]
        in_refs=get_in_refs(event.name,event.path)#5*900*900
        if len(in_refs)==cfg.IN_PERIODS:
            logging.info('start send '+event.name)
            send2out_in(event.name, in_refs)
            in_data = np.zeros([8, 5, 720, 900, 1])
            temp=np.array(in_refs[5:10, 90:-90, :])
            in_data[0] = temp.reshape([5, 720, 900, 1])
            pred=self.predict.predict(in_data)
            save_pred(pred,current_date)


def FSMonitor(path='.',deploy=None):
    """
    Monitor the path-folder,when there is a new file generated,the IN_Create event will be triggered
    :param path:the path to be monitored
    :param deploy:the object if gru-model
    :return:
    """
    wm = WatchManager()
    mask =IN_CREATE
    if deploy is None:
        logging.error('Model Uninitial!')
        return
    notifier = Notifier(wm, EventHandler(predict=deploy))
    wm.add_watch(path, mask, auto_add=True, rec=True)
    print('now starting monitor %s' % (path))
    while True:
        try:
            notifier.process_events()
            if notifier.check_events():
                notifier.read_events()
        except KeyboardInterrupt:
            notifier.stop()
            break
if __name__ == "__main__":
    config_log()
    deploy = Deploy("/home/ices/work/gru_tf_rebuild/save/69999")
    FSMonitor('/tmp/refs',deploy)
    # file_name='cappi_ref_201502210800_2500_8.ref'
    # file_dir = '/home/ices/work/deploy'
    # get_in_refs(file_name,file_dir)
