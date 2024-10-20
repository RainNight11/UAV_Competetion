import torch
import pickle
import argparse
import numpy as np
import pandas as pd


def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        score = np.load(file)  # 使用 numpy 加载 .npy 文件
        score = torch.tensor(score)
        final_score += Rate[idx] * score
    return final_score


def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)

    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num* 100
    return f'{Acc:.2f}%'


def gen_label(val_npy_path):
    true_label = np.load(val_npy_path)  # 直接加载 .npy 文件
    true_label = torch.from_numpy(true_label)  # 将 numpy 数组转换为 torch 张量
    return true_label

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 

    parser.add_argument('--ctrgcn_joint_Score', type = str,default = './pred_ctr_joint.npy')
    parser.add_argument('--ctrgcn_bone_Score', type=str, default='./pred_ctr_bone.npy')
    parser.add_argument('--ctrgcn_joint2d_Score', type=str, default='./pred_ctr_joint2d.npy')
    parser.add_argument('--tegcn_joint_Score', type=str, default='./pred_te_joint.npy')
    parser.add_argument('--mst_joint2d_Score', type=str, default='./pred_mst_joint2d.npy')
    parser.add_argument('--val_sample', type=str, default='./eval/test_label_B.npy')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()


    val_npy_path = args.val_sample

    ctrgcn_joint_file = args.ctrgcn_joint_Score
    # ctrgcn_bone_file = args.ctrgcn_bone_Score
    ctrgcn_joint2d_file = args.ctrgcn_joint2d_Score
    te_joint_file = args.tegcn_joint_Score
    mst_joint2d_file = args.mst_joint2d_Score

    File = [
        ctrgcn_joint_file,te_joint_file,ctrgcn_joint2d_file,mst_joint2d_file
    ]

    Numclass = 155
    Sample_Num = 4599
    Rate = [0.9,0.8,0.32,0.25]
    # Rate = [0.7, 0.7, 0.3, 0.3,
    #         0.3, 0.3, 0.3, 0.3,
    #         0.7, 0.7, 0.3, 0.3,
    #         0.05, 0.05, 0.05, 0.05]
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    true_label = gen_label(val_npy_path)
    
    Acc = Cal_Acc(final_score, true_label)

    print('acc:', Acc)
