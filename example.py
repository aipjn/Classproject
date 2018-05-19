from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.svm import SVM
from utils.scorer import report_score
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    report_score(gold_labels, predictions, True)

    svm = SVM(language)
    svm.train(data.trainset)
    predictions2 = svm.test(data.testset)
    report_score(gold_labels, predictions2, True)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')
    plt.show()


