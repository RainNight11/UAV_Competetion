import torch
import numpy as np
from skopt import gp_minimize
import argparse
import sys
from tqdm import tqdm
from pyswarm import pso 

# 日志保存
log_file = './prediction_former.log'
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


    parser.add_argument('--mixformer_joint', type=str, default='./npy_former/pred_mixformer_joint.npy')
    parser.add_argument('--mixformer_bone', type=str, default='./npy_former/pred_mixformer_bone.npy')
    parser.add_argument('--mixformer_jm', type=str, default='./npy_former/pred_mixformer_jm.npy')
    parser.add_argument('--mixformer_bm', type=str, default='./npy_former/pred_mixformer_bm.npy')

    parser.add_argument('--mixformerk2_joint', type=str, default='./npy_former/pred_mixformerk2_joint.npy')
    parser.add_argument('--mixformerk2_bone', type=str, default='./npy_former/pred_mixformerk2_bone.npy')
    parser.add_argument('--mixformerk2_jm', type=str, default='./npy_former/pred_mixformerk2_jm.npy')
    parser.add_argument('--mixformerk2_bm', type=str, default='./npy_former/pred_mixformerk2_bm.npy')

    parser.add_argument('--val_sample', type=str, default='./data_test/test_label.npy')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    val_npy_path = args.val_sample

    mixformer_joint = args.mixformer_joint
    mixformer_bone = args.mixformer_bone
    mixformer_jm = args.mixformer_jm
    mixformer_bm = args.mixformer_bm

    mixformerk2_joint = args.mixformerk2_joint
    mixformerk2_bone = args.mixformerk2_bone
    mixformerk2_jm = args.mixformerk2_jm
    mixformerk2_bm = args.mixformerk2_bm



        # 定义文件和标签路径
    File = [
        mixformer_joint, mixformer_bone, mixformer_jm, mixformer_bm,
        mixformerk2_joint, mixformerk2_bone, mixformerk2_jm,mixformerk2_bm
    ]
    Numclass = 155
    Sample_Num = 4307
    # 生成真实标签
    true_label = gen_label(val_npy_path)
    # 优化权重的范围
    space = [(0.01, 2.0) for _ in range(8)]  # 为每个流设置一个权重范围 [0.01, 2.0]
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

    best_final_score = Cal_Score(File, result.x, Sample_Num, Numclass)

    # 保存最优得分到 npy 文件
    np.save('pred_former.npy', best_final_score.numpy())
    log.write('最优得分已保存到 pred_former.npy')
    print('Done')
    log.close()  # 关闭日志文件