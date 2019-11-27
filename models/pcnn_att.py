import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import *
from utils import *
from config import get_args
# from train import train
from torch.utils.data import DataLoader

class PCNN_ATT(BasicModel):
    def __init__(self, opt, data_word_vec):
        super(PCNN_ATT, self).__init__(opt, data_word_vec)
        self.embedding = Embedding(opt, data_word_vec)
        self.encoder = PCNN(opt)
        self.selector = Attention(opt, opt.hidden_size * 3)