#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import logging

TORCH_VERSION = torch.__version__
print('Torch Version',TORCH_VERSION)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Temporal Graph Convolution Network')
    parser.add_argument('--work-dir',default='./work_dir/temp',help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--pred_name',default="")
    parser.add_argument('--config',default='./config/mst_joint_test2d.yaml',help='path to the configuration file')
    # parser.add_argument('--config', nargs='+', help='List of config files', required=True)

    # device
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')

    # processor
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--save-score',type=str2bool,default=True,help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval',type=int,default=100,help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval',type=int,default=1,help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval',type=int,default=5,help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    parser.add_argument('--show-topk',type=int,default=[1, 5],nargs='+',help='which Top K accuracy will be shown')

    # load_data
    parser.add_argument('--dataset', default='datasets.dataset.UavDataset', help='data loader will be used')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--train_data_args',default=dict(),help='the arguments of data loader for training')
    parser.add_argument('--val_data_args',default=dict(),help='the arguments of data loader for val')
    parser.add_argument('--test_data_args',default=dict(),help='the arguments of data loader for test')
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',type=dict,default=dict(),help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')
    # 这三个参数暂时不懂
    parser.add_argument('--label_smoothing',type=float,default=0.0,help='label_smoothing')
    parser.add_argument('--ignore-weights',type=str,default=[],nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate') # 动量啥的，不太懂暂时
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=256, help='val batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='testing batch size')
    parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
    parser.add_argument('--num-epoch',type=int,default=80,help='stop training in which epoch')
    parser.add_argument('--lr-decay-rate',type=float,default=0.1,help='decay rate for learning rate')
    parser.add_argument('--weight-decay',type=float,default=0.0005,help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', type=int,default=0)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    return parser

def load_configurations(config_files):
    all_args = {}
    for config_file in config_files:
        with open(config_file, 'rb') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        all_args.update(default_arg)
    return all_args

class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_data_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    #answer = input('delete it? y/n:')
                    answer = 'y'
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        #input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_idx = 0

    def load_data(self):
        UavDataset = import_class(self.arg.dataset)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=UavDataset(**self.arg.train_data_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_workers,
                drop_last=True,
                worker_init_fn=init_seed)
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=UavDataset(**self.arg.val_data_args),
                batch_size=self.arg.val_batch_size,
                shuffle=False,
                num_workers=self.arg.num_workers,
                drop_last=False,
                worker_init_fn=init_seed)
        else:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=UavDataset(**self.arg.test_data_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_workers,
                drop_last=False,
                worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir) # 复制文件到work_dir这个文件夹下面
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights: # 没有初始权重则不执行
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights.items()])

            #self.arg.ignore_weights = ['fc.weight','fc.bias','mlp_head.4.fc','mlp_head.4.bias',"mlp_head.0.weight", "mlp_head.0.bias", "mlp_head.1.weight", "mlp_head.1.bias", "mlp_head.4.weight"]
            self.arg.ignore_weights = []

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))


            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list: # 如果不止一个GPU的话
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg 保存参数
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)


    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        for name, param in self.model.named_parameters():
            self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part: # 是否只训练部分
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)

            # forward
            output = self.model(data)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)


        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        
        if save_model:
            torch.save(self.model.state_dict(), f"{self.arg.model_saved_name}-{epoch}-{int(self.global_step)}.pt")

    def eval(self, epoch, save_score=False, loader_name=['val'], wrong_file=None, result_file=None):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_idx = epoch
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', accuracy, 'Best',self.best_acc,'ep',self.best_idx, ' model: ', self.arg.model_saved_name)

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # score_dict = dict(
            #     zip(self.data_loader[ln].dataset.sample_name, score))
            
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            # if save_score:
            #     with open('{}/epoch{}_{}_score.pkl'.format(
            #             self.arg.work_dir, epoch + 1, ln), 'wb') as f:
            #         pickle.dump(score_dict, f)
            if save_score:
                # Convert score_dict values to a numpy array
                scores_array = np.array(score)
                
                # Save the numpy array as .npy file
                np.save('{}/{}_score.npy'.format(self.arg.work_dir,ln), scores_array)

    def predict(self, loader_name=['test'], save_predictions=True):
        # 记录推理日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='predict.log', filemode='w')

        logging.info("Starting the inference process.")
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for ln in tqdm(loader_name):
                logging.info(f"Processing data loader: {ln}")
                process = tqdm(self.data_loader[ln])
                for batch_idx, (data,index) in enumerate(process):
                    logging.info(f"Processing batch {batch_idx} with data shape {data.shape} and index {index}.")
                    data = data.float().cuda(self.output_device)
                    output = self.model(data)
                    logging.info(f"Model output obtained with shape {output.shape}.")
                
                    # Assuming output is the prediction probabilities
                    predictions.append(output.data.cpu().numpy())
                    logging.info(f"Batch {batch_idx} predictions appended.")
        
        # Concatenate all predictions into a single numpy array
        predictions_array = np.concatenate(predictions, axis=0)
        logging.info(f"Predictions concatenated with final shape {predictions_array.shape}.")
        
        # Save predictions if required
        if save_predictions:
            np.save(self.arg.pred_name, predictions_array)
            logging.info(f"Predictions saved to 'pred.npy'.")
        logging.info("Inference process completed.")
        

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['val'])

            print('best accuracy: ', self.best_acc,'epoch: ',self.best_idx, 'model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.predict(loader_name=['test'])
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == '__main__':
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'rb') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # merged_args = load_configurations(p.config)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
