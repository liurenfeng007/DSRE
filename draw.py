import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import sklearn.metrics


test_result_path = './test_result'

def PR_curve():
    models = ['PCNN_ONE', 'PCNN_ATT']
    for model in models:
        x = np.load(os.path.join(test_result_path, model + '_x.npy'))
        y = np.load(os.path.join(test_result_path, model + '_y.npy'))
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x = x, y = y)
        plt.plot(x, y, lw=2, label=model)
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(
            f1) + '    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300],
                                                                            (y[100] + y[200] + y[300]) / 3))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.3, 1.0)
    plt.xlim(0.0, 0.4)
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(test_result_path, 'pr_curve'))

if __name__ == '__main__':
    PR_curve()
