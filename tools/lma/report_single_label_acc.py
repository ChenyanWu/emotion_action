# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import load
import csv
import numpy as np
from scipy.special import softmax
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

from mmaction.core.evaluation import (get_weighted_score, mean_class_accuracy,confusion_matrix,
                                      top_k_accuracy, mean_average_precision, multi_class_AUC)

def parse_args():
    parser = argparse.ArgumentParser(description='Fusing multiple scores')
    parser.add_argument(
        '--scores',
        nargs='+',
        help='list of scores',
        default=['demo/fuse/rgb.pkl', 'demo/fuse/flow.pkl'])
    parser.add_argument(
        '--coefficients',
        nargs='+',
        type=float,
        help='coefficients of each score file',
        default=[1.0, 1.0])
    parser.add_argument(
        '--datalist',
        help='list of testing data',
        default='demo/fuse/data_list.txt')
    parser.add_argument('--apply-softmax', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.scores) == len(args.coefficients)
    score_list = args.scores
    score_list = [load(f) for f in score_list]
    if args.apply_softmax:

        def apply_softmax(scores):
            return [softmax(score) for score in scores]

        score_list = [apply_softmax(scores) for scores in score_list]

    weighted_scores = get_weighted_score(score_list, args.coefficients)
    csvfile = open(args.datalist)
    csvreader = csv.reader(csvfile)

    labels = []

    task_list = ['bold', 'lma']
    task_type = task_list[1]
    # task_type = task_list[1]
    if task_type == 'bold':
        class_name = 26
        for row in csvreader:
            emotion_cls = np.zeros(26)
            for idx in range(4, 4+26):
                emotion_cls[idx-4] = float(row[idx])
            emotion_cls = (emotion_cls > 0.5).astype(np.int64)
            labels.append(emotion_cls)
    elif task_type == 'lma':
        class_name = 11
        for row in csvreader:
            if row[0].endswith('mp4'):
                emotion_cls = np.zeros(11)
                for idx in range(3, 3+11):
                    try:
                        emotion_cls[idx-3] = float(row[idx])
                    except:
                        # emotion_cls[idx-3] = 0
                        raise
                emotion_cls = (emotion_cls > 0.5).astype(np.int64)
                labels.append(emotion_cls)
    else:
        raise

    # emotion_idx=1
    for emotion_idx in range(class_name):
        single_class_score = [np.array([1 - sigmoid(weighted_score[emotion_idx]), 9 * sigmoid(weighted_score[emotion_idx])]) for weighted_score in weighted_scores]
        # single_class_score = [np.array([-(weighted_score[emotion_idx]), (weighted_score[emotion_idx])]) for weighted_score in weighted_scores]
        single_class_label = [label[emotion_idx] for label in labels]

        mean_class_acc = mean_class_accuracy(single_class_score, single_class_label)
        top_1_acc, top_5_acc = top_k_accuracy(single_class_score, single_class_label, (1, 5))
        fusion_matrix = confusion_matrix(np.argmax(single_class_score, axis=1), single_class_label).astype(float)
        print('***{}***'.format(emotion_idx))
        print(f'Mean Class Accuracy: {mean_class_acc:.04f}')
        print(f'Top 1 Accuracy: {top_1_acc:.04f}')
        print(f'Top 5 Accuracy: {top_5_acc:.04f}')
        print(fusion_matrix)

if __name__ == '__main__':
    main()
