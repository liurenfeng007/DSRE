from config import get_args
import torch
import numpy as np
import models
from utils import Logger, to_tensor, Accuracy, set_seed
import torch.optim as optim
import os
from tqdm import tqdm
import sklearn.metrics


def train(opt, model, use_bag=1):
# def load_train_data(opt, use_bag=1):
    logger("Reading training data...")
    data_word_vec = np.load(os.path.join(opt.data_path, 'vec.npy'))
    # print(data_word_vec.shape) #(114044, 50)
    data_train_word = np.load(os.path.join(opt.data_path, 'train_word.npy'))
    # print(data_train_word.shape) (522611, 120)
    data_train_pos1 = np.load(os.path.join(opt.data_path, 'train_pos1.npy'))
    # print(data_train_pos1.shape) (522611, 120)
    data_train_pos2 = np.load(os.path.join(opt.data_path, 'train_pos2.npy'))
    # (522611, 120)
    data_train_mask = np.load(os.path.join(opt.data_path, 'train_mask.npy'))
    # print(data_train_mask.shape) (522611, 120, 3)
    if use_bag:
        data_query_label = np.load(os.path.join(opt.data_path, 'train_ins_label.npy'))  # (522611,)
        # print(data_query_label[:1000]) [0  0  0  0  0  0  0  0  0  0  0  0  0 48 48 48 48 0]
        data_train_label = np.load(os.path.join(opt.data_path, 'train_bag_label.npy'))
        # print(data_train_label.shape) (281270,) [0  0  0  0  0  0  0  0  0  0  0  0  0 4 48 48 48 0]
        data_train_scope = np.load(os.path.join(opt.data_path, 'train_bag_scope.npy'))
        # print(data_train_scope.shape) (281270, 2)
    else:
        data_train_label = np.load(os.path.join(opt.data_path, 'train_ins_label.npy'))
        # (522611,)
        data_train_scope = np.load(os.path.join(opt.data_path, 'train_ins_scope.npy'))
        # (522611,)
    logger("Finish reading")
    train_order = list(range(len(data_train_label)))
    train_batches = len(data_train_label) // opt.batch_size
    if len(data_train_label) % opt.batch_size != 0:
        train_batches += 1

# def load_test_data(opt, use_bag=1):
    logger("Reading testing data...")
    data_word_vec = np.load(os.path.join(opt.data_path, 'vec.npy'))
    # (114044, 50)
    data_test_word = np.load(os.path.join(opt.data_path, 'test_word.npy'))
    # print(data_test_word.shape) (172448, 120)
    data_test_pos1 = np.load(os.path.join(opt.data_path, 'test_pos1.npy'))
    # print(data_test_pos1.shape) (172448, 120)
    data_test_pos2 = np.load(os.path.join(opt.data_path, 'test_pos2.npy'))
    # (172448, 120)
    data_test_mask = np.load(os.path.join(opt.data_path, 'test_mask.npy'))
    # (172448, 120, 3)
    if use_bag:
        data_test_label = np.load(os.path.join(opt.data_path, 'test_bag_label.npy'))
        # print(data_test_label.shape) # (96678, 53)
        data_test_scope = np.load(os.path.join(opt.data_path, 'test_bag_scope.npy'))
        # print(data_test_scope.shape) (96678, 2)
    else:
        data_test_label = np.load(os.path.join(opt.data_path, 'test_ins_label.npy'))
        data_test_scope = np.load(os.path.join(opt.data_path, 'test_ins_scope.npy'))
    logger("Finish reading")
    test_batches = len(data_test_label) // opt.batch_size
    if len(data_test_label) % opt.batch_size != 0:
        test_batches += 1
    # 正例的个数
    total_recall = data_test_label[:, 1:].sum()
    # print(total_recall)

# def set_train_model(opt, model):
    trainModel = model(opt, data_word_vec)
    best_auc = 0.0
    best_p = None
    best_r = None
    best_epoch = 0

    if opt.use_gpu:
        trainModel.cuda()
    if opt.optimizer_name == 'Adagrad' or opt.optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(trainModel.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_name == 'Adadelta' or opt.optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(trainModel.parameters(), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    elif opt.optimizer_name == 'Adam' or opt.optimizer_name == 'adam':
        optimizer = optim.Adam(trainModel.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(trainModel.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

#   train
    best_auc = 0.0
    best_p = None
    best_r = None
    best_epoch = 0
    acc_NA = Accuracy()
    acc_not_NA = Accuracy()
    acc_total = Accuracy()
    if not os.path.exists(opt.checkponit_path):
        os.mkdir(opt.checkponit_path)
    for epoch in range(opt.num_epoches):
        logger('Epoch' + str(epoch) + ':start')
        acc_NA.clear()
        acc_not_NA.clear()
        acc_total.clear()
        np.random.shuffle(train_order)
        for batch in tqdm(range(train_batches)):
            # get train batch
            input_scope = np.take(data_train_scope, train_order[batch * opt.batch_size: (batch + 1)*opt.batch_size], axis=0)
            index = []
            scope = [0]
            for num in input_scope:
                index = index + list(range(num[0], num[1]+1))
                # print(index) [252409, 161763, 497387, 381598, 423632, 242579, 242580, 242581, 242582]
                scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                # print(scope) [0, 1, 2, 3, 4, 5, 9]
            batch_word = data_train_word[index, :]
            batch_pos1 = data_train_pos1[index, :]
            batch_pos2 = data_train_pos2[index, :]
            batch_mask = data_train_mask[index, :]
            batch_label = np.take(data_train_label, train_order[batch * opt.batch_size: (batch + 1)*opt.batch_size], axis=0)
            batch_attention_query = data_query_label[index]
            batch_scope = scope

            # train one step
            trainModel.embedding.word = to_tensor(batch_word)
            trainModel.embedding.pos1 = to_tensor(batch_pos1)
            trainModel.embedding.pos2 = to_tensor(batch_pos2)
            trainModel.encoder.mask = to_tensor(batch_mask)
            trainModel.selector.scope = batch_scope
            trainModel.selector.attention_query = to_tensor(batch_attention_query)
            trainModel.selector.label = to_tensor(batch_label)
            trainModel.classifier.label = to_tensor(batch_label)
            optimizer.zero_grad()
            loss, _output = trainModel()
            loss.backward()
            optimizer.step()
            for i, prediction in enumerate(_output):
                if batch_label[i] == 0:
                    acc_NA.add(prediction == int(batch_label[i]))
                else:
                    acc_not_NA.add(prediction == int(batch_label[i]))
                acc_total.add(prediction == int(batch_label[i]))
            loss = loss.item()

            logger("epoch %d batch %d | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (
                    epoch, batch, loss, acc_NA.get(), acc_not_NA.get(), acc_total.get()))
        if (epoch + 1) % opt.save_epoch == 0:
            logger('Saving model...')
            path = os.path.join(opt.checkponit_path, model.__name__ + '-' + str(epoch) + '.pth')
            torch.save(trainModel.state_dict(), path)
            logger('Have save model to' + path)

        #dev
        if (epoch + 1) % opt.test_epoch == 0:
            testModel = trainModel
            # auc, pr_x, pr_y =
            test_score = []
            for batch in tqdm(range(test_batches)):
                # get_test_batch
                input_scope = data_test_scope[batch * opt.batch_size: (batch + 1) * opt.batch_size]
                index = []
                scope = [0]
                for num in input_scope:
                    index = index + list(range(num[0], num[1] + 1))
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                batch_word = data_test_word[index, :]
                batch_pos1 = data_test_pos1[index, :]
                batch_pos2 = data_test_pos2[index, :]
                batch_mask = data_test_mask[index, :]
                batch_scope = scope
                # print(scope) [0, 1, 2, 3, 4, 5, 9]

                # test one step
                testModel.embedding.word = to_tensor(batch_word)
                testModel.embedding.pos1 = to_tensor(batch_pos1)
                testModel.embedding.pos2 = to_tensor(batch_pos2)
                testModel.encoder.mask = to_tensor(batch_mask)
                testModel.selector.scope = batch_scope
                batch_score = testModel.test()  # [512* 53]
                test_score = test_score + batch_score
            test_result = []
            for i in range(len(test_score)):
                for j in range(1, len(test_score[i])):
                    test_result.append([data_test_label[i][j], test_score[i][j]])
            test_result = sorted(test_result, key=lambda x: x[1])
            test_result = test_result[::-1]
            pr_x = []
            pr_y = []
            correct = 0
            for i, item in enumerate(test_result):
                correct += item[0]
                pr_y.append(float(correct)/(i + 1))
                pr_x.append(float(correct)/total_recall)
            auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
            logger("AUC:%f" % auc)

            if auc > best_auc:
                best_auc = auc
                best_p = pr_x
                best_r = pr_y
                best_epoch = epoch

    logger("Finish training")
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
    logger("Start training......")
    model = {
        'pcnn_one': models.PCNN_ONE,
        'pcnn_att': models.PCNN_ATT
    }
    train(opt, model[opt.model_name], use_bag=1)


