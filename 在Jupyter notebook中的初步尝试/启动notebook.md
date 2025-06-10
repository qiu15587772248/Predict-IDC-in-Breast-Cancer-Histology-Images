#由于Windows系统在有些环境下将num_workers值设置大于0时会出现报错（在notebook里运行会出现死锁现象！！！），但在Pycharm中可以用“if __name__ == '__main__':”保护起来。
所以在notebook中的尝试因不可抗拒因素停止在了模型运行的初始阶段（因为如果num_workers设置为0将会把训练时长极大延长（即使时50*50的输入尺寸，一轮也需要五十分钟左右）
但我在notebook中尝试了对数据集的划分以及其他预处理，并生成了相应的可视化结果，可以尝试以下指令打开notebook（假设在anaconda中创建了一个名为“pytorch”的虚拟环境，项目文件地址也需要相应调整）

    conda activate pytorch
    E:
    cd 深度学习课设
    jupyter notebook