import os
import time
import numpy
import torch
import numpy as np


class Logger(object):
    """
    generate operating log
    """
    def __init__(self, fp):
        self.fp = fp

    def __call__(self, string, end='\n'):
        new_string = '[%s]' % time.strftime('%Y-%m-%d %H:%M:%S') + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))

def set_seed(seed):
    """
    set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def to_tensor(x):
    if torch.cuda.is_available():
        return torch.from_numpy(x).cuda()
    else:
        return torch.from_numpy(x).cpu()

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0



if __name__ == '__main__':
    logger = Logger(None)
    logger('Load data')
