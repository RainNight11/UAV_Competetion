import numpy as np
import os
import os.path as osp
import logging


def get_raw_bodies_data(data, sample_index, frames_drop_skes, frames_drop_logger):
    """
    从骨架序列中获取原始身体数据。

    每个身体的数据是一个包含以下键的字典：

    joints:原始 3D 关节位置。形状为 (num_frames x 17, 3)。
    colors:原始 2D 颜色位置信息。形状为 (num_frames, 17, 2)。
    interval:存储该身体对应帧索引的列表。
    motion:运动量(仅用于包含 2 个或更多 bodyID 的序列)。
    返回值： 一个包含骨架序列的字典，具有 3 个键值对：

    name:骨架文件名。
    data:一个字典，存储每个身体的原始数据。
    num_frames:有效帧的数量。
    """
    sample_data = data[sample_index]  # 获取指定样本的数据，shape=(C, T, V, M)
    num_frames = sample_data.shape[1]  # T 维表示帧数
    frames_drop = []
    bodies_data = dict()

    for m in range(sample_data.shape[3]):  # 遍历每个人 M=2
        if not np.any(sample_data[:, :, :, m]):  # 如果此人的数据为空
            continue
        joints = np.transpose(sample_data[:3, :, :, m], (1, 2, 0))  # (T, V, 3)

        bodyID = f'person_{m}'
        bodies_data[bodyID] = {
            'joints': joints,  # (T, V, 3)
            'interval': list(range(num_frames))  # 所有帧的索引
        }

    num_frames_drop = len(frames_drop)
    frames_drop_skes[sample_index] = np.array(frames_drop, dtype=int)
    frames_drop_logger.info('{}: {} frames missed: {}\n'.format(sample_index, num_frames_drop, frames_drop))

    return {'name': f'sample_{sample_index}', 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data_npy(npy_file_path, save_path):
    data = np.load(npy_file_path)  # 加载整个 npy 文件
    num_samples = data.shape[0]  # N 维表示样本数

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    frames_drop_skes = dict()
    raw_skes_data = []

    for sample_index in range(num_samples):
        bodies_data = get_raw_bodies_data(data, sample_index, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)

    # 将结果保存为 npy 文件
    np.save(osp.join(save_path, 'raw_skes_data.npy'), raw_skes_data)
    np.save(osp.join(save_path, 'frames_drop_skes.npy'), frames_drop_skes)
    print('Done')

if __name__ == '__main__':
    npy_file_path = './your_data.npy'  # 替换为你的 npy 文件路径
    save_path = './raw_data'
    if not osp.exists(save_path):
        os.makedirs(save_path)

    get_raw_skes_data_npy(npy_file_path, save_path)

