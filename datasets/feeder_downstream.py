# sys
import pickle

# torch
import torch

import numpy as np
np.set_printoptions(threshold=np.inf)

try:
    from datasets import tools
except:
    import tools


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.l_ratio = l_ratio

        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape

        print(self.data.shape, len(self.number_of_frames), len(self.label))
        print("l_ratio", self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)


    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]
        label = self.label[index]

        # crop a sub-sequnce 
        data_numpy = augmentations.crop_subsequence(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        if self.input_representation == "motion":
            # motion
            motion = np.zeros_like(data_numpy)
            motion[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]

            data_numpy = motion

        elif self.input_representation == "bone":
            # bone
            bone = np.zeros_like(data_numpy)
            for v1, v2 in self.Bone:
                bone[:, :, v1 - 1, :] = data_numpy[:, :, v1 - 1, :] - data_numpy[:, :, v2 - 1, :]

            data_numpy = bone

        return data_numpy, label