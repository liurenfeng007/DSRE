import torch
import torch.nn as nn

class _CNN(nn.Module):
    def __init__(self, opt):
        super(_CNN, self).__init__()
        self.opt = opt
        self.in_channels = 1
        self.out_channels = self.opt.hidden_size
        self.in_height = self.opt.sent_max_length
        self.in_width = self.opt.word_dim + self.opt.pos_dim * 2
        self.kernal_size = (self.opt.window_size, self.in_width)
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernal_size, self.stride, self.padding)

    def forward(self, embedding):
        return self.cnn(embedding)

class _MaxPooling(nn.Module):
    def __init__(self):
        super(_MaxPooling, self).__init__()
    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim=2)
        return x.view(-1, hidden_size)  # [854,230]


class _PiecewisePooling(nn.Module):
    def __init__(self):
        super(_PiecewisePooling, self).__init__()
    def forward(self, x, mask, hidden_size):
        # print(x.shape) torch.Size([854, 230, 120, 1])
        mask = torch.unsqueeze(mask, 1)  # [sent_num, 1, sent_max_length, 3]
        # print(mask.shape) torch.Size([854, 1, 120, 3])
        # print((mask+x).shape) torch.Size([854, 230, 120, 3])
        x, _ = torch.max(mask + x, dim=2)  # [sent_num, hidden_size, 3]
        # print(x.shape) torch.Size([854, 230, 3])
        x = x - 100
        return x.view(-1, hidden_size * 3)  # [sent_num, hidden_size * 3]

class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.opt = opt
        self.cnn = _CNN(opt)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()
    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)  # [sent_num, 1, sent_max_length, word_dim+2*pos_dim]
        x = self.cnn(embedding)  # [sent_num, hidden_size, H_out,W_out]
        x = self.pooling(x, self.opt.hidden_size)  # [sent_num  ,hidden_size]
        return self.activation(x)  # [sent_num  ,hidden_size]



class PCNN(nn.Module):
    def __init__(self, opt):
        super(PCNN, self).__init__()
        self.opt = opt
        self.mask = None
        self.cnn = _CNN(opt)
        self.pooling = _PiecewisePooling()
        self.activation = nn.ReLU()
    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)  # [sent_num, 1, sent_max_length, word_dim+2*pos_dim]
        x = self.cnn(embedding)  # [sent_num, hidden_size, H_out,W_out]
        x = self.pooling(x, self.mask, self.opt.hidden_size)  # [sent_num, hidden_size * 3]
        return self.activation(x)  # [sent_num, hidden_size * 3]


if __name__ == '__main__':
    from config import get_args
    opt = get_args()
    out = torch.randn(2,4,60)
    print(out.shape)
    model1 = PCNN(opt)
    model1.mask = torch.Tensor([[[100,0,0],[100,0,0],[0,100,0],[0,0,100]],[[100,0,0],[100,0,0],[0,100,0],[0,0,100]]])
    out1 = model1(out)
    print(out1.shape)


