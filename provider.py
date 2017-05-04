import io, os
import random
import numpy as np
import datetime
from skimage import io 
from skimage import color
import cv2
from paddle.trainer.PyDataProvider2 import *


def initHook(settings, height, width, color, num_class, is_test, is_predict, **kwargs):
    settings.height = height
    settings.width = width
    settings.color = color
    settings.num_class = num_class
    #settings.img_mean = np.array([121.28258514,  120.19075012,  117.60987091])
    settings.img_mean = np.array([123.54631805,  122.70828247,  119.50905609])
    settings.is_test = is_test
    settings.is_predict = is_predict
    if settings.color:
        settings.data_size = settings.height * settings.width * 3
    else:
        settings.data_size = settings.height * settings.width
    if is_predict:
        settings.slots = [dense_vector(settings.data_size)]
    else:
        settings.slots = [dense_vector(settings.data_size), integer_value(1)]

@provider(init_hook=initHook, pool_size=768)
def process(settings, file_list):
    with open(file_list, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            ex = line.strip().split(' ')
            file_name = ex[0]
            img = io.imread(file_name)
            img = img.astype('float32')
            if len(img.flatten()) != settings.data_size:
               if settings.color:
                   img = color.gray2rgb(img)
               else:
                   img = color.rgb2gray(img)
            img = img - settings.img_mean
            img = img.flatten()
            if settings.is_predict:
                yield img.astype('float32')
            else:
                yield img.astype('float32'), int(ex[1])
