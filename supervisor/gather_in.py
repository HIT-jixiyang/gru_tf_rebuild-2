import os
import re
import pandas as pd
import numpy as np
from supervisor.color_map import transfer, mapping
from multiprocessing import Pool


def gather(file_path):
    base_dir = '/home/foggy/data/received'
    date = re.findall(r'\d{12}', file_path)[0]
    # if date < "201809010000":
    #     t_date = pd.to_datetime(date) + pd.Timedelta("24m")
    #     t_date = t_date.strftime("%Y%m%d%H%M")
    #     print("justed")
    # else:
    #     t_date = date
    t_date = date
    dates = pd.date_range(end=t_date, periods=11, freq='6T')
    in_date = date

    print("generate gru serial in truth {}".format(in_date))
    # aim_in_dir = '/var/www/html/image/model6/{}/in'.format(in_date)
    aim_in_dir = os.path.join(file_path, 'in')

    if not os.path.exists(aim_in_dir):
        os.makedirs(aim_in_dir)
    else:
        return

    tar_paths = []
    in_paths = []
    for i, date in enumerate(dates[1:]):
        date = date.strftime("%Y%m%d%H%M")
        path = os.path.join(base_dir, 'cappi_ref_{}_2500_11.ref'.format(date))
        if os.path.exists(path):
            tar_paths.append(path)
            in_paths.append(os.path.join(aim_in_dir, '{}.png'.format(i + 1)))
    p = Pool()
    for t, i in zip(tar_paths, in_paths):
        p.apply_async(deal, args=(t, i))
    p.close()
    p.join()


def deal(tar_path, in_path):
    if tar_path.endswith('ref'):
        ref = np.fromfile(tar_path, dtype=np.uint8).reshape(700, 900)
        ref[ref <= 15] = 0
        ref[ref >= 80] = 0
        img = mapping(ref)
        img.save(in_path)
        print("dealed {}".format(os.path.basename(tar_path)))


def find():
    target = "/var/www/html/image/model6/"
    targets = os.listdir(target)
    targets = sorted(targets)
    for t in targets:
        file_path = os.path.join(target, t)
        gather(file_path)


if __name__ == '__main__':
    find()
