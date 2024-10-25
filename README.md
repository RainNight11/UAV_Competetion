# 说明：

## 环境要求：

项目根目录的environment.yml文件，表示在该比赛项目中所使用的conda环境配置

## 训练日志

见ctr_joint、ctr_bone、mst_joint2d等文件夹的work_dir

## 数据位置

训练集的数据在data文件夹目录下，B测试集的数据在data_test文件夹下，data和data_test的文件都需要通过data_process.ipynb获得（对于训练数据和B测试集需要修改gen_modal的一些内容）

## 运行方法

运行main函数即可，修改config参数，分别对应config文件夹内的不同train、test的yaml文件，最后得到不同的pred文件进行ensemble.py合并

## 权重

见ctr_joint、ctr_bone、mst_joint2d等文件夹，为了减小文件大小删去了部分，仅保留最后使用预测的部分_*_joint.npy
```
