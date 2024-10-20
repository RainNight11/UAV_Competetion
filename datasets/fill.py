import numpy as np

def fill_empty_frames(data):
    """
    通过上一帧填充空帧
    :param data: 形状为 (N, C, T, V, M) 的数据，其中 N 是样本数，C 是通道数（如坐标维度），
                 T 是帧数，V 是关节点数，M 是人数
    :return: 填充后的数据
    """
    N, C, T, V, M = data.shape

    for n in range(N):
        for m in range(M):
            previous_frame = None
            for t in range(T):
                current_frame = data[n, :, t, :, m]
                if np.all(current_frame == 0):  # 检查是否为空帧
                    if previous_frame is not None:
                        data[n, :, t, :, m] = previous_frame  # 用上一帧填充
                else:
                    previous_frame = current_frame  # 更新上一帧

    return data

if __name__ == '__main__':
    data = np.load('../data/train_joint.npy')
    data = fill_empty_frames(data)
    np.save('../data_prenormalized/1111', data)
