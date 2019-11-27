import torch.nn as nn
import torch
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, opt, data_word_vec):
        super(Embedding, self).__init__()
        self.opt = opt
        self.word_embedding = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embedding = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embedding = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.init_word_weights(data_word_vec)
        self.init_pos_weights()
        self.word = None
        self.pos1 = None
        self.pos2 = None


    def init_pos_weights(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def init_word_weights(self, data_word_vec):
        self.word_embedding.weight.data.copy_(torch.from_numpy(data_word_vec))
        # nn.init.xavier_uniform_(self.word_embedding.weight.data)


    def forward(self):
        word = self.word_embedding(self.word)  # [sent_num, sent_max_length, word_dim]
        pos1 = self.pos1_embedding(self.pos1)  # [sent_num, sent_max_length, pos_dim]
        pos2 = self.pos2_embedding(self.pos2)  # [sent_num, sent_max_length, pos_dim]
        embedding = torch.cat((word, pos1, pos2), dim=2)  # [sent_num, sent_max_length, word_dim+2*pos_dim]
        return embedding



if __name__ == '__main__':
    from config import get_args
    import numpy as np
    opt = get_args()
    data_word_vec = np.load('F:\LCS\RE\DSRE\data/vec.npy')
    model = Embedding(opt, data_word_vec)
    model.word = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    model.pos1 = torch.LongTensor([[2, 2, 4, 5], [4, 3, 2, 9]])
    model.pos2 = torch.LongTensor([[3, 2, 4, 5], [4, 3, 2, 9]])
    out = model()
    print(out.shape)

