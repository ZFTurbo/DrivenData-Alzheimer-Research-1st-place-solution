# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
import random
import shutil
import operator
from PIL import Image
import platform
import json
import base64
import typing as t
import zlib
import tqdm
import gc

random.seed(2016)
np.random.seed(2016)

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('acc', 'val_acc')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def normalize_array(cube, new_max, new_min):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(cube), np.max(cube)
    if maximum - minimum != 0:
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        cube = m * cube + b
    return cube


def reduce_model(model_path):
    from kito import reduce_keras_model
    from keras.models import load_model

    m = load_model(model_path)
    m_red = reduce_keras_model(m)
    m_red.save(model_path[:-3] + '_reduced.h5')


def read_video(video_path, frames=None):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frame_list = []
    print('Video length: {} Width: {} Height: {} FPS: {}'.format(length, width, height, fps))
    if frames is None:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame.copy())
            current_frame += 1
    else:
        frame_num = 0
        while (cap.isOpened()):
            ret = cap.grab()
            if not ret:
                break
            if frame_num in frames:
                ret, frame = cap.retrieve()
                frame_list.append(frame.copy())
            frame_num += 1
        cap.release()
    frame_list = np.array(frame_list)
    print(frame_list.shape)
    return frame_list


def merge_models(shape_size, models_list, out_model_file, avg=True):
    from keras.models import load_model, save_model, Model
    from keras.layers import Input, Average

    if not os.path.isfile(out_model_file):
        models = []
        for m_path in models_list:
            m = load_model(m_path)
            models.append(m)

        inp = Input(shape_size)
        outs = []
        for i, m in enumerate(models):
            m.name += '_{}'.format(i)
            x = m(inp)
            outs.append(x)

        if avg is True:
            outs = Average()(outs)
        final_model = Model(inputs=inp, outputs=outs)
        print(final_model.summary())
        print('Single model memory: {} GB'.format(get_model_memory_usage(1, final_model)))
        save_model(final_model, out_model_file)
    else:
        final_model = load_model(out_model_file)

    return final_model


def get_best_model_list(fold_num, folder):
    model_list = []
    for i in range(fold_num):
        files = sorted(glob.glob(folder + '*fold_{}_-*.h5'.format(i)))
        best_model = ''
        best_score = 0.0
        for f in files:
            if '_reduced.h5' in f:
                continue
            score = float(os.path.basename(f).split('-')[-2])
            if score > best_score:
                best_score = score
                best_model = f
        model_list.append(best_model)
        print('Fold: {} Model: {} Score: {}'.format(i, os.path.basename(best_model), best_score))
    return model_list
