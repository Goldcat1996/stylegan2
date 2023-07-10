#训练自己的数据
    修改参数：
        显卡：CUDA_VISIBLE_DEVICES
        端口：port 如果需要同时训练两个任务，保证port不一样
        batch
        size 生成器的输入大小，比如你的初始化模型是1024分辨率，则size=1024
        size_d 鉴别器的输入大小，比如你的初始化模型是1024分辨率，则size_d=1024
        path 自己的训练集路径
        ckpt 加载的预训练模型
        ckpt_save_dir 训练保存的模型位置
    训练脚本：
        bash scripts/train/train.sh

#训练剪枝的模型
    设置 config.py 文件中参数：
        debug_channels_size = True
        channels_ratio的选择表示需要网络会减小的通道数，比如channels_ratio = '12'表示通道数减小1/2
    训练脚本：
        bash scripts/train/train.sh

