# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging

root_path = './'
raw_data_file = osp.join(root_path, 'raw_data', 'raw_skes_data.pkl')
save_path = osp.join(root_path, 'denoised_data')

if not osp.exists(save_path):
    os.mkdir(save_path)

rgb_ske_path = osp.join(save_path, 'rgb+ske')
if not osp.exists(rgb_ske_path):
    os.mkdir(rgb_ske_path)

actors_info_dir = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

missing_count = 0
noise_len_thres = 4
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2

noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))
noise_len_logger.info('{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))
noise_spr_logger.info('{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))
noise_mot_logger.info('{:^20}\t{:^17}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion'))

fail_logger_1 = logging.getLogger('noise_outliers_1')
fail_logger_1.setLevel(logging.INFO)
fail_logger_1.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_1.log')))

fail_logger_2 = logging.getLogger('noise_outliers_2')
fail_logger_2.setLevel(logging.INFO)
fail_logger_2.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_2.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))


import numpy as np

# 设置去噪参数
noise_len_thres = 5  # 例子：去噪帧长阈值
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.1
noise_mot_thres_hi = 5.0

# 加载数据
data = np.load('your_data_file.npy')  # 假设文件名为your_data_file.npy
N, C, T, V, M = data.shape

def denoising_by_length(frames_data):
    new_frames_data = frames_data.copy()
    noise_info = ""
    # 对每个人的轨迹去噪
    for m in range(M):
        if frames_data[:, :, :, m].sum() == 0:
            continue
        num_frames = np.count_nonzero(frames_data[:, :, :, m])
        if num_frames <= noise_len_thres:
            noise_info += f"Filter out: person {m}, length {num_frames}.\n"
            new_frames_data[:, :, :, m] = 0
    return new_frames_data, noise_info

def get_valid_frames_by_spread(frames_data):
    valid_frames = []
    for t in range(T):
        x = frames_data[t, :, 0]
        y = frames_data[t, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):
            valid_frames.append(t)
    return valid_frames

def denoising_by_spread(frames_data):
    new_frames_data = frames_data.copy()
    noise_info = ""
    for m in range(M):
        if frames_data[:, :, :, m].sum() == 0:
            continue
        valid_frames = get_valid_frames_by_spread(frames_data[:, :, :, m])
        num_frames = T
        num_noise = num_frames - len(valid_frames)
        if num_noise / float(num_frames) >= noise_spr_thres2:
            noise_info += f"Filter out: person {m} (spread rate >= {noise_spr_thres2}).\n"
            new_frames_data[:, :, :, m] = 0
        else:
            new_frames_data[:, :, :, m] = frames_data[:, :, :, m][:, valid_frames, :]
    return new_frames_data, noise_info

# 主处理函数
def process_data(data):
    processed_data = []
    for i in range(N):  # 遍历样本
        frames_data, noise_info_len = denoising_by_length(data[i])
        frames_data, noise_info_spr = denoising_by_spread(frames_data)
        processed_data.append(frames_data)
    return np.array(processed_data)

processed_data = process_data(data)
np.save('processed_data.npy', processed_data)
