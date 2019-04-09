import logging
import os
import numpy as np

from model import Model
from iterator import Iterator
import config as c
from utils import config_log, save_png
from utils import normalize_frames, denormalize_frames
import tensorflow as tf

class Runner(object):
    def __init__(self, para_tuple=None):

        self.para_tuple = para_tuple

        self.model = Model(para_tuple[0],para_tuple[1])
        if not para_tuple:
            self.model.init_params()

    def train(self):
        iter = 0
        train_iter = Iterator(time_interval=c.RAINY_TRAIN,
                              sample_mode="random",
                              seq_len=c.IN_SEQ+c.OUT_SEQ)
        merged = tf.summary.merge_all()  # 合并所有的summary data的获取函数，merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
        writer = tf.summary.FileWriter("/extend/rain_data/Logs", self.model.sess.graph)
        while iter < c.MAX_ITER:
            data,*_= train_iter.sample(batch_size=c.BATCH_SIZE)
            in_data = data[:, :c.IN_SEQ, ...]
            gt_data = data[:,c.IN_SEQ:c.IN_SEQ+c.OUT_SEQ, ...]

            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl = self.model.train_step(in_data, gt_data)
            logging.info(f"Iter {iter}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")

            if (iter+1) % c.SAVE_ITER == 0:

                self.model.save_model(iter)

            if (iter + 1) % c.VALID_ITER == 0:
                self.run_benchmark(iter)
            iter += 1

    def run_benchmark(self, iter, mode="Valid"):
        if mode == "Valid":
            time_interval = c.RAINY_VALID
        else:
            time_interval = c.RAINY_TEST
        test_iter = Iterator(time_interval=time_interval,
                             sample_mode="sequent",
                             seq_len=c.IN_SEQ+c.OUT_SEQ,
                             stride=20,mode=mode)
        i = 1
        while not test_iter.use_up:
            data , date_clip, *_ = test_iter.sample(batch_size=c.BATCH_SIZE)

            if mode=='Valid':
                in_data = np.zeros(shape=(c.BATCH_SIZE, c.IN_SEQ, c.H_TRAIN, c.W_TRAIN, c.IN_CHANEL))
                gt_data = np.zeros(shape=(c.BATCH_SIZE, c.OUT_SEQ, c.H_TRAIN, c.W_TRAIN, c.IN_CHANEL))
            else:
                in_data = np.zeros(shape=(c.BATCH_SIZE, c.IN_SEQ, c.H_TEST, c.W_TEST, c.IN_CHANEL))
                gt_data = np.zeros(shape=(c.BATCH_SIZE, c.OUT_SEQ, c.H_TEST, c.W_TEST, c.IN_CHANEL))

            if type(data) == type([]):
                break
            in_data[:, :, :, :, :] = data[:, :c.IN_SEQ, :, :, :]
            gt_data[:, :, :, :, :] = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, :, :, :]
            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)
            if mode=='Valid':
                mse, mae, gdl, pred = self.model.valid_step(in_data, gt_data)
                logging.info(f"Iter {iter} {i}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")
            else:
                pred=self.model.pred_step(in_data)
            i += 1
            for b in range(c.BATCH_SIZE):
                predict_date = date_clip[b]
                logging.info(f"Save {predict_date} results")
                if mode == "Valid":
                    save_path = os.path.join(c.SAVE_VALID, str(iter), predict_date.strftime("%Y%m%d%H%M"))
                else:
                    save_path = os.path.join(c.SAVE_TEST, str(iter), predict_date.strftime("%Y%m%d%H%M"))


                path = os.path.join(save_path, "in")
                save_png(in_data[0], path)

                path = os.path.join(save_path, "pred")
                save_png(pred[0], path)

                path = os.path.join(save_path, "out")
                save_png(gt_data[0], path)

    def test(self):
        iter = self.para_tuple[-1]+"_test"
        self.run_benchmark(iter, mode="Test")


if __name__ == '__main__':
    config_log()
    paras = ('/extend/gru_tf_data/10_20_model/Save/model.ckpt/99999',"train")
    # paras = None
    runner = Runner(para_tuple=paras)
    runner.train()
    # runner.test()
