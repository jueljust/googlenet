import os, sys
import numpy as np
import logging
from skimage import io 
from skimage import color 
from PIL import Image
from optparse import OptionParser
import json

import paddle.utils.image_util as image_util

from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


class ImageClassifier():
    def __init__(self,
                 train_conf,
                 use_gpu=True,
                 model_dir=None,
                 batch_size=16,
                 is_color=True):
        self.train_conf = train_conf
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.img_mean = np.array([123.54631805,  122.70828247,  119.50905609])
        if model_dir is None:
            self.model_dir = os.path.dirname(train_conf)

        self.is_color = is_color

        gpu = 1 if use_gpu else 0
        conf_args = "is_test=1,use_gpu=%d,is_predict=1" % (gpu)
        conf = parse_config(train_conf, conf_args)
        swig_paddle.initPaddle("--use_gpu=%d" % (gpu))
        self.network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
        assert isinstance(self.network, swig_paddle.GradientMachine)
        self.network.loadParameters(self.model_dir)
        channels = 3 if is_color else 1
        self.data_size = channels * 224 * 224
        slots = [dense_vector(self.data_size)]
        self.converter = DataProviderConverter(slots)
    def predict(self, data_file):
        with open(data_file, 'r') as fdata:
            batch = []
            sample_ids = []
            for line in fdata:
                ex = line.strip().split(' ')
                file_name = ex[0]
                label = ex[1]
                sample_id = os.path.basename(file_name)[:-4]
                #print [sample_id, file_name, label]
                img = io.imread(file_name)
                img = img.astype('float32')
                if len(img.flatten()) != self.data_size:
                    if self.is_color:
                        img = color.gray2rgb(img)
                    else:
                        img = color.rgb2gray(img)
                #img = img - self.img_mean
                img = img/256.0
                img = img.flatten()
                #print file_name, len(img)
                batch.append([img.tolist()])
                sample_ids.append(sample_id)
                if len(batch) == self.batch_size:
                    self.batch_predict(batch, sample_ids)
                    batch = []
                    sample_ids = []
        if len(batch) > 0:
            self.batch_predict(batch, sample_ids)
            batch = []
            sample_ids = []
    def batch_predict(self, data_batch, sample_ids):
        input = self.converter(data_batch)
        output = self.network.forwardTest(input)
        prob = output[0]["value"]
        labs = np.argsort(-prob)
        result = zip(prob, labs)
        for idx, (prob, lab) in enumerate(result):
            one_result = { 
               "idx": idx,
               "argmax": int(lab[0]),
               #"label_text": self.label[lab[0]],
               "probs": map(float, prob),
               "sample_id": sample_ids[idx]
            }   
            print json.dumps(one_result, ensure_ascii=False)


if __name__ == '__main__':
    config = "googlenet.py"
    model_path = sys.argv[1]
    test_list = sys.argv[2]
    use_gpu = bool(int(sys.argv[3]))

    obj = ImageClassifier(
        train_conf=config,
        model_dir=model_path,
        use_gpu=use_gpu)
    obj.predict(test_list)
