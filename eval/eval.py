import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='../pred_mst_joint2d.npy')

if __name__ == "__main__":

    args = parser.parse_args()

    # load label and pred
    label =np.load('../eval/test_label_B.npy')

    pred = np.load(args.pred_path).argmax(axis=1)

    correct = (pred == label).sum()

    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))