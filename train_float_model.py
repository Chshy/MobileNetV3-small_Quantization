import os
# import sys
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# sys.path.insert(0, project_root)

from data_loader import get_dataset
# from models.MobileNetV3 import mobilenet_v3_small
from modules.MobileNetV3 import mobilenet_v3_small

from utils.trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 超参数
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001
# EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 256      # batchsize
# EPOCHS = 100            # 训练轮次
# INIT_LR = 1e-2        # Adam初始学习率
# WEIGHT_DECAY = 5e-3   # 权重衰减系数
# MIN_LR = 1e-3         # 最小学习率
# WARMUP_EPOCHS = 5    # 学习率热身轮次


# EPOCHS = 50
# INIT_LR = 1e-2
# WEIGHT_DECAY = 1e-2
# MIN_LR = 1e-3
# WARMUP_EPOCHS = 5

EPOCHS = 70
INIT_LR = 4e-4
WEIGHT_DECAY = 1e-2
MIN_LR = 5e-5
WARMUP_EPOCHS = 0

def main(load_weight_path = None):
    print("using device: ", DEVICE)

    # 获取数据集
    train_set, val_set, test_set = get_dataset("ImageNet1k_64")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = val_loader

    # 初始化模型和优化器
    model = mobilenet_v3_small(num_classes = 1000).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 如果有预训练权重，加载权重
    load_weight_path = "./weights/fp32.pth"
    if load_weight_path is not None:
        model.load_state_dict(torch.load(load_weight_path, map_location=DEVICE))

    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INIT_LR,
        weight_decay=WEIGHT_DECAY
    )

    # 混合调度策略
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            # 第一阶段：线性warmup
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=WARMUP_EPOCHS
            ),
            # 第二阶段：余弦退火
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - WARMUP_EPOCHS,
                eta_min=MIN_LR
            )
        ],
        [WARMUP_EPOCHS]
    )


    # 创建训练器实例
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        # scheduler=None,
        save_best=True,
        save_dir = "./runs",
        experiment_name = "TrainFP32"
    )

    # 开始训练
    trainer.train()

    # 最终评估
    trainer.evaluate()

if __name__ == '__main__':
    main()

