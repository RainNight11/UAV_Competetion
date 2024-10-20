# 2024-全球校园人工智能算法精英大赛-算法挑战赛-基于无人机的人体行为识别

### 依赖库

python ：`numpy tqdm`

### 数据说明
1.数据的格式为（N，C，T，V，M），分别代表样本，通道（三维空间中的XYZ坐标通道），帧数（统一为300帧），节点（统一为17个节点），人数（统一为2，最多两个人时是两个非零值）  
2.标签的格式为（N，），每个样本对应一个标签值  
3.置信度文件格式为（N，155），每个样本对应155个类置信度值  
4.生成Bone模态原理是根据Graph，连通点相减  
5.生成Motion模态原理是前后帧相减  


注意：

1. 如有问题前往QQ群（849776886）提问
2. 完整流程可以直接运行子文件夹下的`ipynb`
3. 国内注意PIP换源，命令为：`pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple`

## 数据准备（UAV-human骨架数据预处理）

### 注意
1. 在验证中，我们发现Windows多线程处理存在一些问题（包括内存占用异常增加），或者系统盘与工作区在一个存储介质上，也会存在系统IO耗尽导致死机卡顿。因此默认启用单线程
2. 为加快处理速度，也可以启用多线程，通常可带来2-4倍的性能提升。

**多线程处理**：在以下提到的命令中，添加`--use_mp True`即可。

### 流程
1. 数据集处理出bone模态数据（可选）：运行`python gen_modal.py --modal bone`得到bone模态数据
2. 数据集处理出motion模态数据（可选）：运行`python gen_modal.py --modal motion`得到motion模态的数据
3. bone模态与joint模态合并（可选）：运行`python gen_modal.py --modal jmb`得到合并模态的数据
4. 最终你会得到如下所展示的目录结构与文件
```
└─data
    ├── train_label.npy
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── test_*_bone_motion.npy
    ├── test_*_bone.npy
    ├── test_*_joint_bone.npy
    ├── test_*_joint_motion.npy
    ├── test_*_joint.npy
```
