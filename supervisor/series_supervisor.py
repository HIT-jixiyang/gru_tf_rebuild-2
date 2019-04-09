import os
import re
import time
import logging
from scipy.misc import imread
from PIL import Image
import pandas as pd
from multiprocessing import Pool
from supervisor.color_map import transfer, multi_process_transfer
from pyinotify import WatchManager, Notifier, ProcessEvent, IN_DELETE, IN_CREATE, IN_MODIFY


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


def gru_completeness(path):
    count = 2
    while count:
        if len(os.listdir(os.path.join(path, 'pred'))) != 10:
            count -= 1
            time.sleep(10)
        else:
            return True
    logging.warning("{} not complete! Waiting".format(path))
    return False


def transfer_gru_serial(file_path):
    date = re.findall(r'\d{12}', file_path)[0]

    target_path = os.path.join(file_path, 'pred')
    des_path = '/var/www/html/image/model6/{}/pred'.format(date)
    logging.info('transferring gru serial prediction {}'.format(date))
    multi_process_transfer(target_path, des_path)


def gather_gru_real(file_path):
    date = re.findall(r'\d{12}', file_path)[0]
    dates = pd.date_range(end=date, periods=11, freq='6T')
    pred_date = dates[0].strftime("%Y%m%d%H%M")
    in_date = date

    logging.info("generate gru serial ground truth {}".format(pred_date))
    aim_dir = '/var/www/html/image/model6/{}/out'.format(pred_date)
    aim_in_dir = '/var/www/html/image/model6/{}/in'.format(in_date)

    if not os.path.exists(aim_dir):

        os.makedirs(aim_dir)
    if not os.path.exists(aim_in_dir):
        os.makedirs(aim_in_dir)

    tar_paths = []
    des_paths = []
    in_paths = []

    for i, date in enumerate(dates[1:]):
        date = date.strftime("%Y%m%d%H%M")
        path = os.path.join('/extend/deploy-midresult/real_time', '{}.png'.format(date))
        if os.path.exists(path):
            tar_paths.append(path)
            des_paths.append(os.path.join(aim_dir, '{}.png'.format(i+1)))
            in_paths.append(os.path.join(aim_in_dir, '{}.png'.format(i+1)))
    p = Pool()
    for t, d, i in zip(tar_paths, des_paths, in_paths):
        p.apply_async(transfer, args=(t, d))
        p.apply_async(transfer, args=(t, i))
    p.close()
    p.join()


class GruEventHandler(ProcessEvent):
    """
    Event Processing
    """
    def process_IN_CREATE(self, event):
        logging.info("Detected file creation: {} ".format(os.path.join(event.path, event.name)))
        file_path = os.path.join(event.path, event.name)

        if os.path.isdir(file_path) and re.match(r'(.*)\d{12}(.*)', file_path):
            aim_path = os.path.join(file_path, 'pred', '10.png')
            pred_dir =  os.path.join(file_path, 'pred')
            count = 5
            while not os.path.exists(pred_dir) and count:
                count -= 1
                time.sleep(3)
            count = 3
            while not gru_completeness(file_path):
                count -= 1
                if count == 0:
                    logging.error("out time {}".format(file_path))
                    return
            transfer_gru_serial(file_path)


class RealTimeEventHandler(ProcessEvent):
    """
    Event Processing
    """
    def process_IN_CREATE(self, event):
        logging.info("Detected file creation: {} ".format(os.path.join(event.path, event.name)))
        file_path = os.path.join(event.path, event.name)
        if file_path.endswith('png'):
            time.sleep(3)
            img = imread(file_path)
            if img.shape == (900, 900):
                print("reshaping")
                img = img[100:800, :]
                Image.fromarray(img).save(file_path)
            gather_gru_real(file_path)


def monitor(watch):
    gru_path = os.path.join(watch, 'series')
    real_path = os.path.join(watch, 'real_time')
    mask = IN_CREATE

    config_log()
    wm = WatchManager()

    wm.add_watch(gru_path, mask, proc_fun=GruEventHandler(), rec=True)
    wm.add_watch(real_path, mask, proc_fun=RealTimeEventHandler(), rec=True)

    notifier = Notifier(wm)

    # print('now starting monitor %s' % (watch))

    logging.info('now starting monitor %s' % (watch))

    notifier.loop()


if __name__ == "__main__":
    path = "/extend/deploy-midresult"
    monitor(path)
