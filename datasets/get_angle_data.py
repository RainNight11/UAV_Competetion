import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep

# bone

def gen_angle_data_one_num_worker(path_train,path_test):
    # if os.path.basename(path) == 'MMVRAC_CSv1.npz':
    # new_train_x = open_memmap('../data/train_joint.npy',dtype='float32',mode='w+',shape=(16432, 9, 300, 17, 2))
    # new_test_x = open_memmap('../data/test_A_joint.npy',dtype='float32',mode='w+',shape=(2000, 9, 300, 17, 2))
    new_train_x = open_memmap('../data/train_joint.npy', dtype='float32', mode='w+', shape=(16432, 9, 300, 17, 2))
    new_test_x = open_memmap('../data/test_A_joint.npy', dtype='float32', mode='w+', shape=(2000, 9, 300, 17, 2))
    # else:
    #     new_train_x = open_memmap('train_joint.npy',dtype='float32',mode='w+',shape=(16431, 9, 305, 17, 2))
    #     new_test_x = open_memmap('test_A_joint.npy',dtype='float32',mode='w+',shape=(6598, 9, 305, 17, 2))
    #
    # data = np.load(path,mmap_mode='r')
    # train_x = data[f'x_train']
    # train_y = data[f'y_train']
    # test_x = data[f'x_test']
    # test_y = data[f'y_test']
    train_data = np.load(path_train)
    test_data = np.load(path_test)

    N_train, _,T_train, _,_ = train_data.shape
    N_test,_, T_test, _ ,_= test_data.shape
    train_x = train_data.reshape((N_train, T_train, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    test_x = test_data.reshape((N_test, T_test, 2, 17, 3)).transpose(0, 4, 1, 3, 2)

    Parallel(n_jobs=6)(delayed(lambda i: new_train_x.__setitem__(i,getThridOrderRep(train_x[i])))(i) for i in tqdm(range(N_train)))
    Parallel(n_jobs=6)(delayed(lambda i: new_test_x.__setitem__(i,getThridOrderRep(test_x[i])))(i) for i in tqdm(range(N_test)))

    new_train_x = new_train_x.transpose(0, 2, 4, 3, 1).reshape(N_train, T_train, -1)
    new_test_x = new_test_x.transpose(0, 2, 4, 3, 1).reshape(N_test, T_test, -1)

    np.save('../data_prenormalized/train_joint.npy', new_train_x)
    np.save('../data_prenormalized/test_A_joint.npy', new_test_x)

    # np.savez(f'{path[:-4]}_angle.npz', x_train=new_train_x, y_train=train_y, x_test=new_test_x, y_test=test_y)

if __name__ == '__main__':
    # gen_angle_data_one_num_worker('../data/train_joint.npy','../data/test_A_joint.npy')
