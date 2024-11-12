# 说明：

## Github项目链接

https://github.com/RainNight11/UAV_Competetion

## 环境要求：

项目根目录的environment.yml文件，表示在该比赛项目中所使用的conda环境配置

## 训练日志

我们将训练日志与相应的权重文件放置在log_and_weights下

## 数据位置

训练集的数据在data文件夹目录下，测试集的数据在data_test文件夹下，data和data_test的文件都需要通过data_process.ipynb获得（对于训练数据和测试集需要修改gen_modal的一些内容）

## 运行方法

运行main函数即可，修改config参数，分别对应config与config_mixformer文件夹内的不同train、test的yaml文件，最后得到不同的pred文件（文件列于下面，可使用网盘给定的权重进行检查）
**注意：对于ctr_loss，需要运行main_loss.py，因为对model与main函数有修改**

## 最优权重搜索

对于这里的ensemble.py使用验证集生成的置信度个标签，我们摒弃了高斯过程最小化操作，使用**遗传算法**进行最优权重的搜索，将最优权重在log_and_weights文件夹下的prediction中取出
**注意：我们在使用遗传算法时发现每次得到结果都不同，具有较大的不稳定性，故给出最优权重**

## 关于不同的GCN不同模态的pred.npy文件

将其放置在pred_npy_file下面，可以直接使用复核，也可以运行config的测试文件进行生成。

## 权重

https://drive.google.com/file/d/1GruB3k0Gppt73Mz4PiRPH3b2OvWWhWXA/view?usp=sharing

