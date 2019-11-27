import torch.nn as nn
import torch
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.label = None
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits):
        # print(self.label.shape) torch.Size([512])
        loss = self.loss(logits, self.label)
        _, output = torch.max(logits, dim=1)
        return loss, output.data