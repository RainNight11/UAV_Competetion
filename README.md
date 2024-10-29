  # 说明：

## 环境要求：

项目根目录的environment.yml文件，表示在该比赛项目中所使用的conda环境配置

## 训练日志

位于train_prediction_log文件夹下，其中包含一个prediction日志，用来保存预测时使用的高斯过程最小化方法对应的权重及其最优权重（用A测试集测试时发现每次会所差异，故保存下来）。

其余的均为不同GCN网络对应的不同模态的训练日志，但是并不是每一个GCN都用了所有模态。

## 数据位置

训练集的数据在data文件夹目录下，B测试集的数据在data_test文件夹下，data和data_test的文件都需要通过data_process.ipynb获得（对于训练数据和B测试集需要修改gen_modal的一些内容）

## 运行方法

运行main函数即可，修改config参数，分别对应config与config_mixformer文件夹内的不同train、test的yaml文件，最后得到不同的pred文件（文件列于下面，可使用网盘给定的权重进行检查），最后进行Mix_B.py合并。

对于这里的Mix_A.py使用A测试集生成的置信度个标签找到最优权重之后，将最优权重在train_prediction_log文件夹下的prediction中取出，然后输入Mix_B.py进行直接融合由B测试集得到的npy文件。所以这里的rate并不是手动调节，数值小数点位数较多。（之后手动删除了其中的tdgcn_jb和mstgcn模态，因为准确率相对较低，得出结果更优，且在高斯分布最小化之后采取手动微调的方式，准确率略微提高。）

对于项目中ctrgcn_joint、tegcn_joint这类的文件夹是为了在训练时将权重与日志保存于此。

## 关于不同的GCN不同模态的pred.npy文件

将其放置在pred_npy_file下面，共14个，可以直接使用复核，也可以运行config的测试文件进行生成。

## 权重
[https://drive.google.com/drive/folders/1TMTTBBcxCugXOq7Hfld2nrRs-RcIdbYn?usp=drive_link](https://drive.google.com/drive/folders/1TMTTBBcxCugXOq7Hfld2nrRs-RcIdbYn?usp=drive_link)
