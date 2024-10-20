import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import os

from datasets.tools import random_rot

sys.path.extend(['../'])
from . import tools
from . import preprocess

class UavDataset(Dataset): # 构建数据集
    def __init__(self, data_path, label_path=None,
                random_choose=False, random_shift=False, random_move=False,p_interval=1,
                window_size=-1, normalization=False, debug=False, use_mmap=True,is_test=False,random_rot=False,d=3):
        """
        :param data_path: 数据文件路径
        :param label_path: 标签文件路径
        :param random_choose: 如果为 True，则随机选择输入序列的一部分
        :param random_shift: 如果为 True，则在序列的开头或结尾随机填充零
        :param random_move: 如果为 True，则在数据中随机移动
        :param window_size: 输出序列的长度
        :param normalization: 如果为 True，则对输入序列进行归一化
        :param debug: 如果为 True，则只使用前 100 个样本
        :param use_mmap: 如果为 True，则使用 mmap 模式加载数据，以节省内存
        """

        self.is_test = is_test
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.d = d
        self.load_data()

        if normalization:
            self.get_mean_map()

    def fill_empty_frames(self,data):
        """
        通过上一帧填充空帧
        :param data: 形状为 (N, C, T, V, M) 的数据，其中 N 是样本数，C 是通道数（如坐标维度），
                     T 是帧数，V 是关节点数，M 是人数
        :return: 填充后的数据
        """
        C, T, V, M = data.shape

        for m in range(M):
            previous_frame = None
            for t in range(T):
                current_frame = data[:, t, :, m]
                if np.all(current_frame == 0):  # 检查是否为空帧
                    if previous_frame is not None:
                        data[ :, t, :, m] = previous_frame  # 用上一帧填充
                else:
                    previous_frame = current_frame  # 更新上一帧

        return data

    def load_data(self): # 加载数据
        # data: N C V T M

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.is_test:
            # 测试数据集不加载标签
            self.label = None
            # if self.debug:
            #     self.data = self.data[0:100]
        else:
            file_ext = os.path.splitext(self.label_path)[-1]

            data = np.load(self.label_path, allow_pickle=True)
            self.label = data

        if self.debug: # 使用前100个样本
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            # self.sample_name = self.sample_name[0:100]

    def get_mean_map(self): # 归一化操作
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        data_numpy = self.data[index]
        data_numpy = np.array(data_numpy)

        data_numpy = self.fill_empty_frames(data_numpy)

        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # # reshape Tx(MVC) to CTVM
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.d == 2:
            data_numpy = data_numpy[[0,2],:,:,:]
        else:
            data_numpy = data_numpy

        if self.random_rot:
            data_numpy = random_rot(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        if self.is_test:
            return data_numpy, index
        else:
            label = self.label[index]
        return data_numpy, label, index

    def top_k(self, score, top_k):  # 计算分类评分的Top-K准确率
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Dataset(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)

if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)