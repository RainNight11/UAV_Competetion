import time
import torch

import numpy as np
np.set_printoptions(threshold=np.inf)
import random

from datasets import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.crop_resize = True
        self.l_ratio = l_ratio

        self.load_data(mmap)

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V

        print(self.data.shape, len(self.number_of_frames))
        print("l_ratio", self.l_ratio)

    def load_data(self, mmap):
        # data: N C T V M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        if self.num_frame_path != None:
            self.number_of_frames = np.load(self.num_frame_path)
        else:
            self.number_of_frames = np.ones(self.data.shape[0], dtype=np.int32)*50

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
  
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        
        number_of_frames = self.number_of_frames[index]

        # temporal crop-resize
        data_numpy_v1 = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        data_numpy_v2 = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Rotate(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Flip(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.Shear(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.spatial_masking(data_numpy_v1)
        if random.random() < 0.5:
            data_numpy_v1 = augmentations.temporal_masking(data_numpy_v1)

        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Rotate(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Flip(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.Shear(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.spatial_masking(data_numpy_v2)
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.temporal_masking(data_numpy_v2)

        return data_numpy_v1, data_numpy_v2