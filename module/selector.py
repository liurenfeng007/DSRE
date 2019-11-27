import torch.nn as nn
import torch
import torch.nn.functional as F

class Selector(nn.Module):
    def __init__(self, opt, relation_dim):
        super(Selector, self).__init__()
        self.opt = opt
        self.relation_matrix = nn.Embedding(self.opt.rel_num, relation_dim)  # (53,690)
        self.bias = nn.Parameter(torch.Tensor(self.opt.rel_num))  # (53)
        self.attention_matrix = nn.Embedding(self.opt.rel_num, relation_dim)  # (53,690)
        self.linear = nn.Linear(relation_dim, self.opt.rel_num)
        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.label = None
        self.dropout = nn.Dropout(self.opt.drop_out)
    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)
        nn.init.xavier_uniform_(self.attention_matrix.weight.data)
    def get_logits(self, x):
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1)) + self.bias
        return logits
    def forward(self, x):
        raise NotImplementedError
    def test(self, x):
        raise NotImplementedError


class One(Selector):
    def forward(self, x):
        tower_logits = []
        for i in range(len(self.scope) - 1):
            # 分包
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            sen_matrix = self.dropout(sen_matrix)
            # print(sen_matrix.shape) [n, 690]
            logits = self.get_logits(sen_matrix)  # [n, 53]
            # logits = self.linear(sen_matrix)
            score = F.softmax(logits, 1)  # [n, 53]
            # score = self.linear(sen_matrix)
            _, k = torch.max(score, dim=0)
            # print(k) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
            # print(_.shape) [53]
            # 在对应的relation上选最大的
            k = k[self.label[i]]
            tower_logits.append(logits[k])  # [512 * [53]]
        # return torch.cat(tower_logits, 0).view(len(self.scope)-1, -1)
        return torch.stack(tower_logits)  # tensor([512,53])

    def test(self, x):
        tower_score = []
        for i in range(len(self.scope)-1):
            sen_matrix = x[self.scope[i]: self.scope[i+1]]  # [n, 690]
            logits = self.get_logits(sen_matrix)  # [n,53]
            score = F.softmax(logits, 1)  # [n,53]
            # 每个类别最大的score
            score, _ = torch.max(score, dim=0)  # [53]
            tower_score.append(score)  # [512,tensor[53]]
        tower_score = torch.stack(tower_score)  # tensor[512 * 53]
        return list(tower_score.data.cpu().numpy())  # [512*[53]]


class Attention(Selector):
    def _attention_train_logit(self, x):
        relation_query = self.relation_matrix(self.attention_query)
        attention = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * attention * relation_query, 1, True)
        return attention_logit

    def _attention_test_logit(self, x):
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0, 1))
        return attention_logit

    def forward(self, x):
        attention_logit = self._attention_train_logit(x)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]:self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        # logits = self.linear(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        attention_logit = self._attention_test_logit(x)
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]:self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre)
            # logits = self.linear(final_repre)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())






if __name__ == '__main__':
    from config import get_args
    opt = get_args()
    out = torch.randn(2, 690)
    print(out.shape)
    model1 = One(opt,690)
    out1 = model1(out)
    print(out1.shape)