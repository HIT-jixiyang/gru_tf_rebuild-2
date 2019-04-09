import numpy as np
from model import Model
import config as c
import cv2

class Deploy(object):
    def __init__(self, load_path=""):
        assert load_path is not None
        self.model = Model(load_path, mode="online")

    def predict(self, in_data):
        # gt_data = np.zeros((c.BATCH_SIZE, c.PREDICT_LENGTH, c.H, c.W, c.IN_CHANEL))
        pred = self.model.pred_step(in_data)
        return pred


if __name__ == '__main__':
    from iterator import Iterator
    it = Iterator(time_interval=c.RAINY_TEST,
                  sample_mode="sequent",
                  seq_len=10,
                  stride=1,
                  mode='online'
                  )
    deploy = Deploy("/extend/gru_tf_data/10_20_model/Save/model.ckpt/99999")
    while True:

        test_data,data_times,*_=it.sample(c.BATCH_SIZE)
        print(test_data.shape)
        pred= deploy.predict(test_data[:,0:10])

    # new_test_data=np.zeros([8,5,720,900,1],dtype=np.float32)
    # new_test_data[0]=test_data
    # data, *_ = it.sample(batch_size=c.BATCH_SIZE)
    # print(data.shape)
    # deploy = Deploy("/home/ices/work/gru_tf_rebuild/save/149999")
    # pred1= deploy.predict(new_test_data)
    # # print(pred1.min(), pred1.max())
    # # print(pred1.shape)
    # pred2=pred1
    # pred2[pred2<0]=0
    # pred = pred2.astype(np.uint8)
    # result=pred[0]
    # for i in range(len(result)):
    #     cv2.imwrite('/extend/gru_tf_data/pred/'+str(i)+'-'+'.png',result[i].reshape(720,900))
