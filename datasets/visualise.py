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


    plt.ion() # 显示一个动态图
    fig = plt.figure()

    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

    if graph is None:
        p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
        pose = [
            ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
        ]
        if is_3d:
            ax.axis([-1, 1, -1, 1,-1,1])
        else:
            ax.axis([-1, 1, -1, 1])

        for t in range(T):
            for m in range(M):
                pose[m].set_xdata(data[0, 0, t, :, m])
                pose[m].set_ydata(data[0, 1, t, :, m])
            fig.canvas.draw()
            plt.pause(0.001)
    else:
        p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
        G = graph
        edge = G.inward
        pose = []  # 存储绘图对象
        for m in range(M):
            a = [] # 存储该个体骨架的连线
            for i in range(len(edge)):
                if is_3d:
                    a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                else:
                    a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
            pose.append(a)
        if is_3d:
            ax.axis([-300, 300, -250, 250,-420,420])
        else:
            ax.axis([-300, 300, -420, 420])


        for t in range(T): # 时间步
            for m in range(M): # 人数
                for i, (v1, v2) in enumerate(edge):
                    x1 = data[0, :2, t, v1, m]
                    x2 = data[0, :2, t, v2, m] # v1与v2节点的x,y坐标
                    if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                        if is_3d:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                        else:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 2, t, [v1, v2], m])
                            

            fig.canvas.draw()  # 更新绘图
            plt.waitforbuttonpress()
            # plt.pause(0.1)

if __name__ == '__main__':
    from Model_inference.Mix_Former.graph import uav

    dataset = UavDataset(data_path='../data/train_joint.npy',
                           label_path='../data/train_label.npy',
                           random_choose=False,
                           random_shift=False,
                           random_move=False,
                           window_size=-1,
                           normalization=False,
                           debug=False,
                           random_rot=False,
                           use_mmap=True)

    # dataloader = DataLoader(dataset, batch_size=1)
    #
    # for cnt,(images,labels,_) in enumerate(tqdm(dataloader)):
    #     visualise(images, graph=uav.Graph(), is_3d=True)
    #     if cnt >=10:
    #         break

    sample_index = 8640

    images, labels, _ = dataset[sample_index]
    images = torch.from_numpy(images)

    visualise(images.unsqueeze(0), graph=uav.Graph(), is_3d=False)