import torch
import numpy as np
from skopt import gp_minimize
import argparse
import sys
from tqdm import tqdm

# 日志保存
log_file = './prediction.log'
log = open(log_file, 'w')  # 打开日志文件用于保存输出

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    """计算多流模型的加权得分."""
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        score = np.load(file)  # 使用 numpy 加载 .npy 文件
        score = torch.tensor(score)
        final_score += Rate[idx] * score
    return final_score


def Cal_Acc(final_score, true_label):
    """计算准确率."""
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)

    wrong_num = len(wrong_index)
    total_num = true_label.shape[0]
    Acc = (total_num - wrong_num) / total_num * 100
    return Acc


def gen_label(val_npy_path):
    """生成真实标签."""
    true_label = np.load(val_npy_path)
    true_label = torch.from_numpy(true_label)
    return true_label



def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble')

    parser.add_argument('--ctrgcn_joint_Score', type = str,default = './pred_ctr_joint.npy')
    parser.add_argument('--ctrgcn_bone_Score', type=str, default='./pred_ctr_bone.npy')
    parser.add_argument('--ctrgcn_jm_Score', type=str, default='./pred_ctr_jm.npy')
    parser.add_argument('--tdgcn_joint_Score', type = str,default = './pred_td_joint.npy')
    parser.add_argument('--tdgcn_bone_Score', type=str, default='./pred_td_bone.npy')
    parser.add_argument('--tdgcn_jm_Score', type=str, default='./pred_td_jm.npy')
    parser.add_argument('--tdgcn_jb_Score', type=str, default='./pred_td_jb.npy')
    parser.add_argument('--ctrgcn_joint2d_Score', type=str, default='./pred_ctr_joint2d.npy')
    parser.add_argument('--ctrgcn_bone2d_Score', type=str, default='./pred_ctr_bone2d.npy')
    parser.add_argument('--tegcn_joint_Score', type=str, default='./pred_te_joint.npy')
    parser.add_argument('--tegcn_bone_Score', type=str, default='./pred_te_bone.npy')
    parser.add_argument('--mst_joint2d_Score', type=str, default='./pred_mst_joint2d.npy')
    parser.add_argument('--mixformer_joint_Score', type=str, default='./pred_mixformer_joint.npy')
    parser.add_argument('--mixformer_joint2k_Score', type=str, default='./pred_mixformer_joint2k.npy')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    ctrgcn_joint_file = args.ctrgcn_joint_Score
    ctrgcn_bone_file = args.ctrgcn_bone_Score
    ctrgcn_jm_file = args.ctrgcn_jm_Score
    tdgcn_joint_file = args.tdgcn_joint_Score
    tdgcn_bone_file = args.tdgcn_bone_Score
    tdgcn_jm_file = args.tdgcn_jm_Score
    tdgcn_jb_file = args.tdgcn_jb_Score
    te_joint_file = args.tegcn_joint_Score
    te_bone_file = args.tegcn_bone_Score
    ctrgcn_joint2d_file = args.ctrgcn_joint2d_Score
    ctrgcn_bone2d_file = args.ctrgcn_bone2d_Score
    mst_joint2d_file = args.mst_joint2d_Score
    mixformer_joint_file = args.mixformer_joint_Score
    mixformer_joint2k_file = args.mixformer_joint2k_Score

    File = [
        ctrgcn_joint_file, ctrgcn_bone_file,ctrgcn_jm_file,
        tdgcn_joint_file, tdgcn_bone_file, tdgcn_jm_file ,
        te_joint_file, te_bone_file,
        ctrgcn_joint2d_file, ctrgcn_bone2d_file,

        mixformer_joint_file,mixformer_joint2k_file
    ]
    Numclass = 155
    Sample_Num = 4599
    Rate = [2.07, 0.01, 0.5877840709272881,
            0.9096500782040964, 0.4920203478822007, 0.9864170104440244,
            0.01, 0.01, 0.7042257515428801,
            0.8650336414607624, 0.01, 0.40304553848426633, 0.01]

    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    result_file = 'pred_test.npy'
    np.save(result_file, final_score)
