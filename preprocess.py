import os
import numpy as np
import json
from config import get_args
from utils import Logger

class Preprocess(object):
    def __init__(self, opt, logger):
        super(Preprocess, self).__init__()
        self.opt = opt
        self.logger = logger
        train_file_name = os.path.join(self.opt.raw_data_path + 'train.json')
        test_file_name = os.path.join(self.opt.raw_data_path + 'test.json')
        word_file_name = os.path.join(self.opt.raw_data_path + 'word_vec.json')
        rel_file_name = os.path.join(self.opt.raw_data_path + 'rel2id.json')
        self.preprocess(train_file_name, word_file_name, rel_file_name, case_sensitive=False, is_training=True)
        self.preprocess(test_file_name, word_file_name, rel_file_name, case_sensitive=False, is_training=False)

    def preprocess(self, file_name, word_vec_file_name, rel2id_file_name, case_sensitive=False, is_training=True):
        self.logger('Loading data file...')
        ori_data = json.load(open(file_name, 'r'))
        ori_word_vec = json.load(open(word_vec_file_name))
        rel2id = json.load(open(rel2id_file_name, 'r'))
        self.logger('Finish loading')

        # 是否区分大小写
        if not case_sensitive:
            self.logger("Eliminating case sensitive problem")
            for i in ori_data:
                i['sentence'] = i['sentence'].lower()
                i['head']['word'] = i['head']['word'].lower()
                i['tail']['word'] = i['tail']['word'].lower()
            for i in ori_word_vec:
                i['word'] = i['word'].lower()
            self.logger("Finish eliminating")

        # build word vec
        self.logger("Building word vector matrix and mapping")
        word2id = {}
        id2word = {}
        word_vec_matrix = []
        word_dim = len(ori_word_vec[0]['vec'])
        self.logger("Got {} words of {} dims".format(len(ori_word_vec), word_dim))  # [114042,50]
        for i in ori_word_vec:
            word2id[i['word']] = len(word2id)
            word_vec_matrix.append(i['vec'])
        word2id['UNK'] = len(word2id)
        word2id['BLANK'] = len(word2id)
        for i in word2id:
            id2word[word2id[i]] = i
        # 高斯分布
        word_vec_matrix.append(np.random.normal(loc=0, scale=0.05, size=word_dim))  # UNK
        # 均匀分布
        # word_vec_matix.append(np.random.uniform(low=-0.5, high=0.5, size=word_dim))
        word_vec_matrix.append(np.zeros(word_dim, dtype=np.float32))  # BLANK
        word_vec_matrix = np.array(word_vec_matrix, dtype=np.float32)
        # print(word_vec_matrix.shape)  # [114044,50]
        self.logger("Finish building")

        # sort 为了确定包的范围先进行排序
        self.logger("Sort data")
        ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
        self.logger("Finish sorting")

        def find_pos(sentence, head, tail):
            def find(sentence, entity):
                p = sentence.find(' ' + entity + ' ')
                if p == -1:
                    if sentence[:len(entity) + 1] == entity + ' ':
                        p = 0
                    elif sentence[-len(entity) - 1:] == ' ' + entity:
                        p = len(sentence) - len(entity)
                    else:
                        p = 0
                else:
                    p += 1
                return p

            sentence = ' '.join(sentence.split())
            p1 = find(sentence, head)
            p2 = find(sentence, tail)
            words = sentence.split()
            cur_pos = 0
            pos1 = -1
            pos2 = -1
            for i, word in enumerate(words):
                if cur_pos == p1:
                    pos1 = i
                if cur_pos == p2:
                    pos2 = i
                cur_pos += len(word) + 1
            if pos1 == -1 or pos2 == -1:
                raise Exception("[ERROR] Position error, sentence = {}, head = {}, tail = {}".format(sentence, head, tail))
            if pos1 >= self.opt.sent_max_length:
                pos1 = self.opt.sent_max_length - 1
            if pos2 >= self.opt.sent_max_length:
                pos2 = self.opt.sent_max_length - 1
            return pos1, pos2

        # 将转化为id
        sen_num = len(ori_data)  # 522611
        sen_word = np.zeros((sen_num, self.opt.sent_max_length), dtype=np.int64)  # [522611,]
        sen_pos1 = np.zeros((sen_num, self.opt.sent_max_length), dtype=np.int64)
        sen_pos2 = np.zeros((sen_num, self.opt.sent_max_length), dtype=np.int64)
        sen_mask = np.zeros((sen_num, self.opt.sent_max_length, 3), dtype=np.float32)
        sen_label = np.zeros((sen_num), dtype=np.int64)
        sen_len = np.zeros((sen_num), dtype=np.int64)
        bag_label = []
        bag_scope = []
        bag_key = []
        for i in range(sen_num):
            if i%1000 ==0:
                print(i)
            sen = ori_data[i]
            # sen_label -> id
            if sen['relation'] in rel2id:
                sen_label[i] = rel2id[sen['relation']]
            else:
                sen_label[i] = rel2id['NA']
            words = sen['sentence'].split()
            # sen_len
            sen_len[i] = min(len(words), self.opt.sent_max_length)
            # sen_word -> id
            for j, word in enumerate(words):
                if j < self.opt.sent_max_length:
                    if word in word2id:
                        sen_word[i][j] = word2id[word]
                    else:
                        sen_word[i][j] = word2id['UNK']
            for j in range(j + 1, self.opt.sent_max_length):
                sen_word[i][j] = word2id['BLANK']
            # entity position
            pos1, pos2 = find_pos(sen['sentence'], sen['head']['word'], sen['tail']['word'])
            pos_min = min(pos1, pos2)
            pos_max = max(pos1, pos2)
            for j in range(self.opt.sent_max_length):
                sen_pos1[i][j] = j-pos1+self.opt.sent_max_length
                sen_pos2[i][j] = j-pos2+self.opt.sent_max_length
                if j >= sen_len[i]:
                    sen_mask[i][j] = [0,0,0]
                elif j - pos_min <= 0:
                    sen_mask[i][j] = [100,0,0]
                elif j - pos_max <= 0:
                    sen_mask[i][j] = [0,100,0]
                else:
                    sen_mask[i][j] = [0,0,100]
            # bag scope
            if is_training:
                tuple = (sen['head']['id'], sen['tail']['id'], sen['relation'])
            else:
                tuple = (sen['head']['id'], sen['tail']['id'])
            if bag_key == [] or bag_key[len(bag_key) - 1] != tuple:
                bag_key.append(tuple)
                bag_scope.append([i, i])
            bag_scope[len(bag_scope) - 1][1] = i

        self.logger("Processing bag label...")
        if is_training:
            for i in bag_scope:
                bag_label.append(sen_label[i[0]])
        else:
            # 存在多标签的情况
            for i in bag_scope:
                muti_hot = np.zeros(len(rel2id), dtype=np.int64)
                for j in range(i[0], i[1] + 1):
                    muti_hot[sen_label[j]] = 1
                bag_label.append(muti_hot)
        self.logger("Finish processing...")

        ins_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))])
        # [[     0      1      2 ... 522608 522609 522610]
        #  [     0      1      2 ... 522608 522609 522610]]

        self.logger("Processing instance label...")
        if is_training:
            ins_label = sen_label
        else:
            ins_label = []
            for i in sen_label:
                one_hot = np.zeros(len(rel2id), dtype=np.int64)
                one_hot[i] = 1
                ins_label.append(one_hot)
            ins_label = np.array(ins_label, dtype=np.int64)
        self.logger("Finish processing...")
        bag_scope = np.array(bag_scope, dtype=np.int64)
        bag_label = np.array(bag_label, dtype=np.int64)
        ins_scope = np.array(ins_scope, dtype=np.int64)
        ins_label = np.array(ins_label, dtype=np.int64)

        self.logger("Saving files")
        if is_training:
            name_prefix = "train"
        else:
            name_prefix = "test"
        np.save(os.path.join(self.opt.data_path, 'vec.npy'), word_vec_matrix)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_word.npy'), sen_word)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_pos1.npy'), sen_pos1)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_pos2.npy'), sen_pos2)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_mask.npy'), sen_mask)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_bag_label.npy'), bag_label)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_bag_scope.npy'), bag_scope)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_ins_label.npy'), ins_label)
        np.save(os.path.join(self.opt.data_path, name_prefix + '_ins_scope.npy'), ins_scope)
        self.logger("Finish saving")


if __name__ == '__main__':
    opt = get_args()
    logger = Logger(None)
    logger('Preprocess Data ......')
    Preprocess(opt, logger)
