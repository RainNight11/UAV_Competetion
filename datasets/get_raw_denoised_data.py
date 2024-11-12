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


missing_count = 0
noise_len_thres = 4
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2


# data_shape:(N,C,T,V,M)
# data_shape:(C,T,V,M)
def denoising_by_length(data):
    """
    根据每个 bodyID 的帧长度进行去噪。
    过滤掉帧长度小于或等于预定义阈值的 bodyID
    """
    N = data.shape[0]
    total_frames = 300
    num_preson = 2
    for i in range(N):
        for m in range(num_preson):
            drop_frames = 0
            for t in range(total_frames):
                body = data[i,: ,t ,: ,m]
                if np.all(body == 0):
                    drop_frames += 1
            num_frames = total_frames - drop_frames
            if num_frames <= noise_len_thres: # 该帧数是否小于或等于预定义的长度阈值,若小于则筛除
                data[i,:, :, :, m] = 0
    return data

# data_shape:(C,T,V,M)
def get_valid_frames_by_spread(data):
    """
    根据 X 和 Y 坐标的扩展范围来查找有效（或合理）的帧（索引）。
    points: 关节或颜色
    """
    total_frames = 300
    valid_frames = []
    for i in range(total_frames):
        x = data[0,i,:,:]
        z = data[2,i,:,:]
        if (x.max() - x.min()) <= noise_spr_thres1 * (z.max() - z.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames


def denoising_by_spread(data):
    """
    根据 Y 值和 X 值的扩展范围对数据进行降噪。
    过滤掉噪声帧比例高于预定义阈值的 bodyID。
    bodies_data：至少包含 2 个 bodyID。
    """

    N = data.shape[0]
    total_frames = 300
    num_preson = 2

    motion = np.zeros((N, num_preson))
    for i in range(N):
        for m in range(num_preson):
            joint_variances = np.var(data[i, :, :, :, m], axis=2)
            motion[i, m] = np.sum(joint_variances)

            drop_frames = 0
            for t in range(total_frames):
                body = data[i, :, t, :, m]
                if np.all(body == 0):
                    drop_frames += 1
            num_frames = total_frames - drop_frames

            valid_frames = get_valid_frames_by_spread(data[i,:,:,:])
            num_noise = num_frames - len(valid_frames)
            if num_noise == 0:
                continue

            ratio = num_noise / float(num_frames)
            if ratio >= noise_spr_thres2:  # 0.69754
                data[i,:, :, :, m] = 0
            else:  # Update motion
                motion[:,m] = min(motion[:,m], np.var(data[i, :, :, :, m], axis=2))
                # TODO: Consider removing noisy frames for each bodyID

    return data


# def denoising_by_motion(ske_name, bodies_data, bodies_motion):
#     """
#     过滤掉运动量超出预定义区间范围的 bodyID
#
#     """
#     # Sort bodies based on the motion, return a list of tuples
#     # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
#     bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
#
#     # Reserve the body data with the largest motion
#     denoised_bodies_data = [(bodies_motion[0][0], bodies_data[bodies_motion[0][0]])]
#     noise_info = str()
#
#     for (bodyID, motion) in bodies_motion[1:]:
#         if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
#             noise_info += 'Filter out: %s, %.6f (motion).\n' % (bodyID, motion)
#             noise_mot_logger.info('{}\t{}\t{:.6f}'.format(ske_name, bodyID, motion))
#         else:
#             denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
#     if noise_info != '':
#         noise_info += '\n'
#
#     return denoised_bodies_data, noise_info


def denoising_bodies_data(data):
    """
    基于一些启发式方法进行数据去噪，不一定适用于所有样本。

    返回： denoised_bodies_data（列表）：元组：（bodyID, body_data）。
    """

    data = denoising_by_length(data)

    data, denoised_by_spr = denoising_by_spread(data)
    # for (bodyID, body_data) in bodies_data.items():
    #     bodies_motion[bodyID] = body_data['motion']
    # # Sort bodies based on the motion
    # # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    # denoised_bodies_data = list()
    # for (bodyID, _) in bodies_motion:
    #     denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return data

    # TODO: Consider denoising further by integrating motion method


# def get_one_actor_points(body_data, num_frames):
#     """
#     Get joints and colors for only one actor.
#     For joints, each frame contains 75 X-Y-Z coordinates.
#     For colors, each frame contains 25 x 2 (X, Y) coordinates.
#     """
#     joints = np.zeros((num_frames, 51), dtype=np.float32)
#     start, end = body_data['interval'][0], body_data['interval'][-1]
#     joints[start:end + 1] = body_data['joints'].reshape(-1, 51)
#     return joints


import numpy as np


def remove_missing_frames(data):
    """
    去除所有关节位置为零的帧
    如果数据包含两个演员，还记录每个演员的缺失帧数，便于调试。
    """
    N = data.shape[0]
    total_frames = 300
    missing_indices_1 = []
    missing_indices_2 = []

    for i in range(N):
        drop_frames_1 = 0
        drop_frames_2 = 0

        for t in range(total_frames):
            body_1 = data[i, :, t, :, 0]
            body_2 = data[i, :, t, :, 1]

            if np.all(body_1 == 0):
                drop_frames_1 += 1
            if np.all(body_2 == 0):
                drop_frames_2 += 1

        if drop_frames_1 > 0:
            missing_indices_1.append(i)
        if drop_frames_2 > 0:
            missing_indices_2.append(i)

    # 计算有效的索引
    valid_indices = np.where(np.sum(data, axis=(1, 2, 3)) != 0)[0]  # 只保留有效的样本
    data = data[valid_indices]
    return data


def get_two_actors_points(data):
    """
    获取第一个和第二个演员的关节位置和颜色位置。
    # 参数：
        bodies_data (dict): 包含3个键值对的字典：'name'、'data'、'num_frames'。
        bodies_data['data'] 也是一个字典，其中键是 bodyID，值是
        对应的 body_data 也是一个字典，包含4个键：
          - joints: 原始的3D关节位置，形状为 (num_frames x 25, 3)。
          - colors: 原始的2D颜色位置，形状为 (num_frames, 25, 2)。
          - interval: 记录帧索引的列表。
          - motion: 动作量。

    # 返回：
        joints, colors。
    """
    ske_name = data['name']
    label = int(ske_name[-2:])
    num_frames = data['num_frames']

    data = denoising_bodies_data(data)  # Denoising data

    if len(bodies_data) == 1:  # Only left one actor after denoising
        if label >= 50:  # DEBUG: Denoising failed for two-subjects action
            fail_logger_2.info(ske_name)

        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
    else:
        if label < 50:  # DEBUG: Denoising failed for one-subject action
            fail_logger_1.info(ske_name)

        joints = np.zeros((num_frames, 102), dtype=np.float32)
        colors = np.ones((num_frames, 2, 17, 2), dtype=np.float32) * np.nan

        bodyID, actor1 = bodies_data[0]  # the 1st actor with largest motion
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :51] = actor1['joints'].reshape(-1, 51)
        colors[start1:end1 + 1, 0] = actor1['colors']
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del bodies_data[0]

        start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                joints[start:end + 1, :51] = actor['joints'].reshape(-1, 51)
                colors[start:end + 1, 0] = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor1
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                joints[start:end + 1, 51:] = actor['joints'].reshape(-1, 51)
                colors[start:end + 1, 1] = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor2
                start2 = min(start, start2)
                end2 = max(end, end2)
            del bodies_data[0]

        bodies_info += ('\n' + actor1_info + '\n' + actor2_info)

    with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
        fw.write(bodies_info + '\n')

    return joints, colors


def get_raw_denoised_data():
    """
    Get denoised data (joints positions and color locations) from raw skeleton sequences.

    For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
    by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
    3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
    two actor's joints positions constituting a 150-dim vector. If there is only one actor,
    then the last 75 values are filled with zeros. Otherwise, select the main actor and the
    second actor based on the motion amount. Each 150-dim vector as a row vector is put into
    a 2D numpy array where the number of rows equals the number of valid frames. All such
    2D arrays are put into a list and finally the list is serialized into a cPickle file.

    For the skeleton sequence which contains two or more actors (mostly corresponds to the
    last 11 classes), the filename and actors' information are recorded into log files.
    For better understanding, also generate RGB+skeleton videos for visualization.
    """

    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)

    raw_denoised_joints = []
    raw_denoised_colors = []
    frames_cnt = []

    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print('Processing %s' % ske_name)
        num_bodies = len(bodies_data['data'])

        if num_bodies == 1:  # only 1 actor
            num_frames = bodies_data['num_frames']
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = get_one_actor_points(body_data, num_frames)
        else:  # more than 1 actor, select two main actors
            joints, colors = get_two_actors_points(bodies_data)
            # Remove missing frames
            joints, colors = remove_missing_frames(ske_name, joints, colors)
            num_frames = joints.shape[0]  # Update
            # Visualize selected actors' skeletons on RGB videos.

        raw_denoised_joints.append(joints)
        raw_denoised_colors.append(colors)
        frames_cnt.append(num_frames)

        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d), ' % \
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes) + \
                  'Missing count: %d' % missing_count)

    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    raw_skes_colors_pkl = osp.join(save_path, 'raw_denoised_colors.pkl')
    with open(raw_skes_colors_pkl, 'wb') as f:
        pickle.dump(raw_denoised_colors, f, pickle.HIGHEST_PROTOCOL)

    frames_cnt = np.array(frames_cnt, dtype=int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw denoised positions of {} frames into {}'.format(np.sum(frames_cnt),
                                                                     raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)

if __name__ == '__main__':
    get_raw_denoised_data()
