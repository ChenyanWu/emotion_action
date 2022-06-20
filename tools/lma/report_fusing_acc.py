# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import load
import csv
import numpy as np
from scipy.special import softmax

from mmaction.core.evaluation import (get_weighted_score, mean_class_accuracy,
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
    for row in csvreader:
        emotion_cls = np.zeros(26)
        for idx in range(4, 4+26):
            emotion_cls[idx-4] = float(row[idx])
        emotion_cls = (emotion_cls > 0.5).astype(np.int16)
        labels.append(emotion_cls)

    mAP = mean_average_precision(weighted_scores, labels)
    mAR = multi_class_AUC(weighted_scores, labels)

    print(f'Mean AP: {mAP:.04f}')
    print(f'Mean AUC ROC: {mAR:.04f}')

if __name__ == '__main__':
    main()
