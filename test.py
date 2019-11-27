from config import get_args
import torch
import numpy as np
import models
from utils import Logger, to_tensor, Accuracy, set_seed
import torch.optim as optim
import os
from tqdm import tqdm
import sklearn.metrics


def test(opt, model, use_bag=1):
    # def load_test_data(opt, use_bag=1):
    logger("Reading testing data...")
    data_word_vec = np.load(os.path.join(opt.data_path, 'vec.npy'))
    data_test_word = np.load(os.path.join(opt.data_path, 'test_word.npy'))
    data_test_pos1 = np.load(os.path.join(opt.data_path, 'test_pos1.npy'))
    data_test_pos2 = np.load(os.path.join(opt.data_path, 'test_pos2.npy'))
    data_test_mask = np.load(os.path.join(opt.data_path, 'test_mask.npy'))
    if use_bag:
        data_test_label = np.load(os.path.join(opt.data_path, 'test_bag_label.npy'))
        # print(data_test_label.sum())
        data_test_scope = np.load(os.path.join(opt.data_path, 'test_bag_scope.npy'))
    else:
        data_test_label = np.load(os.path.join(opt.data_path, 'test_ins_label.npy'))
        data_test_scope = np.load(os.path.join(opt.data_path, 'test_ins_scope.npy'))
    logger("Finish reading")
    test_batches = len(data_test_label) // opt.batch_size
    if len(data_test_label) % opt.batch_size != 0:
            test_batches += 1
    # 可能正例的个数
    total_recall = data_test_label[:, 1:].sum()  # 1950

    # def set_test_model(opt, model):
    testModel = model(opt, data_word_vec)
    if opt.use_gpu:
        testModel.cuda()
    testModel.eval()

    # test
    best_auc = 0.0
    best_p = None
    best_r = None
    best_epoch = 0
    for epoch in range(12, 13):
        path = os.path.join(opt.checkponit_path, model.__name__ + '-' + str(epoch) + '.pth')
        if not os.path.exists(path):
            continue
        testModel.load_state_dict(torch.load(path, map_location='cpu'))


        test_score = []
        for batch in tqdm(range(test_batches)):
            # get_test_batch
            input_scope = data_test_scope[batch * opt.batch_size: (batch + 1) * opt.batch_size]
            index = []
            scope = [0]
            for num in input_scope:
                index = index + list(range(num[0], num[1] + 1))
                # [0, 1 , 2, 3, 4, .....]
                scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                # 包的起点
            batch_word = data_test_word[index, :]
            batch_pos1 = data_test_pos1[index, :]
            batch_pos2 = data_test_pos2[index, :]
            batch_mask = data_test_mask[index, :]
            batch_scope = scope

            # test one step
            testModel.embedding.word = to_tensor(batch_word)
            testModel.embedding.pos1 = to_tensor(batch_pos1)
            testModel.embedding.pos2 = to_tensor(batch_pos2)
            testModel.encoder.mask = to_tensor(batch_mask)
            testModel.selector.scope = batch_scope
            batch_score = testModel.test()  # [512,[53]]
            test_score = test_score + batch_score  # [96678,[53]]
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):
                test_result.append([data_test_label[i][j], test_score[i][j]])
        test_result = sorted(test_result, key=lambda x: x[1])  # list[96678*52]
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        logger("AUC:%f" % auc)

        if auc > best_auc:
            best_auc = auc
            best_p = pr_x
            best_r = pr_y
            best_epoch = epoch

        logger("Finish testing epoch %d" %  (epoch))
    logger("Best epoch = %d | AUC = %f" % (best_epoch, best_auc))
    if not os.path.isdir(opt.test_result_path):
        os.mkdir(opt.test_result_path)
    np.save(os.path.join(opt.test_result_path, model.__name__ + '_x.npy'), best_p)
    np.save(os.path.join(opt.test_result_path, model.__name__ + '_y.npy'), best_r)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    opt = get_args()
    set_seed(opt.seed)
    logger = Logger(None)
    logger("Start testing......")
    model = {
        'pcnn_one': models.PCNN_ONE,
        'pcnn_att': models.PCNN_ATT
    }
    test(opt, model[opt.model_name], use_bag=1)