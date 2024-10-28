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


def objective(Rate):
    """目标函数：根据不同权重计算准确率，返回负的准确率供最小化."""
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    Acc = Cal_Acc(final_score, true_label)
    print(f'当前权重 {Rate} 的准确率: {Acc:.3f}%')
    log.write(f'当前权重: {Rate}\n')
    return -Acc  # 由于我们希望最大化准确率，因此返回负的准确率以供最小化

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble')

    parser.add_argument('--ctrgcn_joint_Score', type = str,default = './pred_ctr_jointA.npy')
    parser.add_argument('--ctrgcn_bone_Score', type=str, default='./pred_ctr_boneA.npy')
    parser.add_argument('--ctrgcn_jm_Score', type=str, default='./pred_ctr_jmA.npy')
    parser.add_argument('--tdgcn_joint_Score', type = str,default = './pred_td_jointA.npy')
    parser.add_argument('--tdgcn_bone_Score', type=str, default='./pred_td_boneA.npy')
    parser.add_argument('--tdgcn_jm_Score', type=str, default='./pred_td_jmA.npy')
    parser.add_argument('--tdgcn_jb_Score', type=str, default='./pred_td_jbA.npy')
    parser.add_argument('--ctrgcn_joint2d_Score', type=str, default='./pred_ctr_joint2dA.npy')
    parser.add_argument('--ctrgcn_bone2d_Score', type=str, default='./pred_ctr_bone2dA.npy')
    parser.add_argument('--tegcn_joint_Score', type=str, default='./pred_te_jointA.npy')
    parser.add_argument('--tegcn_bone_Score', type=str, default='./pred_te_boneA.npy')
    parser.add_argument('--mst_joint2d_Score', type=str, default='./pred_mst_joint2dA.npy')
    parser.add_argument('--mixformer_joint_Score', type=str, default='./pred_mixformer_jointA.npy')
    parser.add_argument('--mixformer_joint2k_Score', type=str, default='./pred_mixformer_joint2kA.npy')
    parser.add_argument('--val_sample', type=str, default='./eval/test_A_label.npy')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    val_npy_path = args.val_sample

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
    Sample_Num = 2000
    # 生成真实标签
    true_label = gen_label(val_npy_path)
    # 优化权重的范围
    space = [(0.01, 2.0) for _ in range(12)]  # 为每个流设置一个权重范围 [0.01, 2.0]
    # 使用高斯过程最小化来优化权重

    # 使用 tqdm 显示进度条
    n_calls = 300
    with tqdm(total=n_calls, desc="优化进度", file=sys.stdout) as pbar:
        def on_step(res):
            pbar.update()
            log.write(f'当前优化进度: {pbar.n}/{n_calls}\n')
        # 使用高斯过程最小化来优化权重
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=0, callback=[on_step])

    # 打印最优结果
    print('最大准确率: {:.4f}%'.format(-result.fun))
    print('最优权重: {}'.format(result.x))
    log.write('最优权重: {}'.format(result.x))

    log.close()  # 关闭日志文件
