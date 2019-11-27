import argparse

Hyperparameter = {
    'pcnn_one': 'lr = 0.01, optimizer_name=Adadelta, ',
    'pcnn_att': 'lr = 0.5, optimizer_name=SGD'
}


def get_args():
    parser = argparse.ArgumentParser("""Arguments for relation extraction""")
    parser.add_argument("--seed", type=int, default=123, help='random seed')##
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')##
    parser.add_argument('--model_name', type=str, default='pcnn_att', help='name of the model')##
    parser.add_argument('--optimizer_name', type=str, default='SGD', help='name of the model')  ##
    parser.add_argument("--num_workers", type=int, default=0, help='how many workers for loading data')
    parser.add_argument("--sent_max_length", type=int, default=120, help='max length for (each sentence + two padding)')##
    parser.add_argument("--limit", type=int, default=50, help='the position range(-limit, limit)*')
    parser.add_argument("--vocab_size", type=int, default=114044, help='the num of the vocab')##
    parser.add_argument("--pos_size", type=int, default=240, help='the num of the position')##
    parser.add_argument("--rel_num", type=int, default=53, help='the num of the relation')##
    parser.add_argument("--hidden_size", type=int, default=230, help='the num of the feature maps')##
    parser.add_argument("--window_size", type=int, default=3, help='the num of window')  ##
    parser.add_argument("--save_epoch", type=int, default=1, help='the interval num of saving epoch')  ##
    parser.add_argument("--test_epoch", type=int, default=1, help='the interval num of test epoch')  ##

    parser.add_argument("--batch_size", type=int, default=160)##
    parser.add_argument("--num_epoches", type=int, default=15)##
    parser.add_argument("--lr", type=float, default=0.5, help='learning rate')##
    parser.add_argument("--lr_decay", type=float, default=0.95, help='learning rate decay')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help='learning rate')##
    parser.add_argument("--drop_out", type=float, default=0.5, help='the num of the relation')##
    parser.add_argument("--word_dim", type=int, default=50, help='dimension of word embedding')##
    parser.add_argument("--pos_dim", type=int, default=5, help='dimension of position embedding*')##

    # text preprocess
    parser.add_argument('--raw_data_path', type=str, default='./raw_data/', help='path to read NYT10 raw data')##
    parser.add_argument('--data_path', type=str, default='./data', help='path to read NYT10 preprocessed data')##
    parser.add_argument("--checkponit_path", type=str, default="./checkponits", help='path of checkpoint')  ##
    parser.add_argument('--test_result_path', type=str, default='./test_result', help='path of test result')  ##

    parser.add_argument("--train_path", type=str, default="./data/bags_train.txt")
    parser.add_argument("--test_path", type=str, default="./data/bags_test.txt")
    parser.add_argument("--word2vec_path", type=str, default="./data/vec.npy", help='path to word vector*')
    parser.add_argument("--p1_2v_path", type=str, default="dataset/NYT/p1_2v.npy")
    parser.add_argument("--p2_2v_path", type=str, default="dataset/NYT/p2_2v.npy")
    parser.add_argument("--result_path", type=str, default="./out")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()