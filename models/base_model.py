import torch
import torch.nn as nn
from module.embedding import *
from module.encoder import *
from module.selector import *
from module.classifier import *


class BasicModel(nn.Module):
    def __init__(self, opt, data_word_vec):
        super(BasicModel, self).__init__()
        self.opt = opt
        self.embedding = Embedding(opt, data_word_vec)
        self.encoder = None
        self.selector = None
        self.classifier = Classifier(opt)

    def forward(self):
        embedding = self.embedding()
        # print(embedding.shape) torch.Size([854, 120, 60])
        sen_embedding = self.encoder(embedding)
        # print(sen_embedding.shape) torch.Size([854, 690])
        logits = self.selector(sen_embedding)
        # print(logits.shape) torch.Size([512, 53])
        return self.classifier(logits)

    def test(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        return self.selector.test(sen_embedding)


