import time
import os
import shutil
import numpy as np
from PIL import Image
import logging
import random

import torch


def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter(object):

    def __init__(self, window=-1):
        self.window = window
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = -np.Inf

        if self.window > 0:
            self.val_arr = np.zeros(self.window)
            self.arr_idx = 0

    def update(self, val, n=1):
        self.val = val
        self.cnt += n
        self.max = max(self.max, val)

        if self.window > 0:
            self.val_arr[self.arr_idx] = val
            self.arr_idx = (self.arr_idx + 1) % self.window
            self.avg = self.val_arr.mean()
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


class FrameSecondMeter(object):

    def __init__(self):
        self.st = time.time()
        self.fps = None
        self.ed = None
        self.frame_n = 0

    def add_frame_n(self, frame_n):
        self.frame_n += frame_n

    def end(self):
        self.ed = time.time()
        self.fps = self.frame_n / (self.ed - self.st)


def gct(f='l'):
    '''
    get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    '''
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%m_%d_%H_%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def count_model_size(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def print_mem(info=None):
    if info:
        print(info, end=' ')
    mem_allocated = round(torch.cuda.memory_allocated() / 1048576)
    mem_cached = round(torch.cuda.memory_cached() / 1048576)
    print(f'Mem allocated: {mem_allocated}MB, Mem cached: {mem_cached}MB')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out