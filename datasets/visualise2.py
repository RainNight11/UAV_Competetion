import matplotlib.pyplot as plt
from datasets.dataset import *


def visualise(data, graph=None, is_3d=True):
    '''
    Visualise skeleton data using matplotlib

    Args:
        data: tensor of shape (B x C x V x T x M)
        graph: graph representation for skeleton
        is_3d: set true for 3d skeletons
    '''
    N, C, T, V, M = data.shape

    rows = 15  # 15 行
    cols = 20  # 20 列
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    G = graph
    edge = G.inward

    for t in range(T):  # 时间步

        ax = axes[t]  # 获取对应的子图
        ax.set_title(f'time:{t+1}s')
        if is_3d:
            ax = fig.add_subplot(rows, cols, t + 1, projection='3d')

        for m in range(M):  # 人数
            for i, (v1, v2) in enumerate(edge):
                x1 = data[0, :2, t, v1, m]
                x2 = data[0, :2, t, v2, m]  # v1与v2节点的x,y坐标
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    if is_3d:
                        ax.plot([data[0, 0, t, v1, m], data[0, 0, t, v2, m]],
                                [data[0, 1, t, v1, m], data[0, 1, t, v2, m]],
                                [data[0, 2, t, v1, m], data[0, 2, t, v2, m]], p_type[m])
                    else:
                        ax.plot(data[0, 0, t, [v1, v2], m],
                                data[0, 2, t, [v1, v2], m], p_type[m])
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        if is_3d:
            ax.set_zlim([-100, 100])
            # 关闭三维坐标轴的刻度
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        else:
            # 关闭二维坐标轴的刻度
            ax.set_xticks([])
            ax.set_yticks([])


    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示所有子图

if __name__ == '__main__':
    from Model_inference.Mix_Former.graph import uav

    dataset = UavDataset(data_path='../data_prenormalized/1111.npy',
                         label_path='../data/train_label.npy',
                         random_choose=False,
                         random_shift=False,
                         random_move=False,
                         window_size=-1,
                         normalization=False,
                         debug=False,
                         random_rot=False,
                         use_mmap=True)

    sample_index = 8640

    images, labels, _ = dataset[sample_index]
    images = torch.from_numpy(images)

    visualise(images.unsqueeze(0), graph=uav.Graph(), is_3d=False)
