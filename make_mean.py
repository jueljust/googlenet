import io, os
import random
import numpy as np
import datetime
from skimage import io 
from skimage import color
from paddle.trainer.PyDataProvider2 import *


def process(settings, file_list):
    mean_values = []
    with open(file_list, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            ex = line.strip().split(' ')
            file_name = ex[0]
            print line_count
            img = io.imread(file_name)
            if len(img.flatten()) != settings['data_size']:
               if settings['color']:
                   img = color.gray2rgb(img)
               else:
                   img = color.rgb2gray(img)
            img = img.astype('float32')
            mean_value = img.mean(axis=0).mean(axis=0)
            mean_values.append(mean_value)
    mean_values = np.array(mean_values)
    print mean_values.mean(axis=0)

settings = {
    "color": True,
    "data_size": 3*224*224
}
process(settings, 'train.txt')
