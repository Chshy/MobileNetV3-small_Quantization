import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import get_dataset
from modules.MobileNetV3 import MobileNetV3
from utils.trainer import Trainer


# 超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

BASE_BATCH_SIZE = 512
BATCH_SIZE = BASE_BATCH_SIZE * NUM_GPUS

BASE_LR = 1e-3
SCALED_LR = BASE_LR * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5
INIT_LR = SCALED_LR
WARMUP_EPOCHS = 5
MIN_LR = 1e-5

WEIGHT_DECAY = 5e-4

EPOCHS = 150

import json

def get_custom_mobilenet_model(num_classes=1000, json_file = None):

    if json_file is not None:
        with open(json_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'input_channels': 3,
            'init_conv': {'kernel': 3, 'out_channels': 16, 'use_se': False, 'use_hs': True, 'stride': 1, 'padding': 1},
            'blocks': [
                {'kernel':3, 'exp_size':16,  'out_channels':16, 'use_se':True,  'use_hs':False, 'stride':2},
                {'kernel':3, 'exp_size':72,  'out_channels':24, 'use_se':False, 'use_hs':False, 'stride':2},
                {'kernel':3, 'exp_size':88,  'out_channels':24, 'use_se':False, 'use_hs':False, 'stride':1},
                {'kernel':5, 'exp_size':96,  'out_channels':40, 'use_se':True,  'use_hs':True,  'stride':2},
                {'kernel':5, 'exp_size':240, 'out_channels':40, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':240, 'out_channels':40, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':120, 'out_channels':48, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':144, 'out_channels':48, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':288, 'out_channels':96, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':576, 'out_channels':96, 'use_se':True,  'use_hs':True,  'stride':1},
                {'kernel':5, 'exp_size':576, 'out_channels':96, 'use_se':True,  'use_hs':True,  'stride':1},
            ],
            'final_conv': { 'kernel': 1, 'out_channels': 512, 'use_se': True, 'use_hs': True, 'stride': 1},
            'classifier_hidden_dim': 512
        }
    return MobileNetV3(config, num_classes=num_classes)

def main(load_weight_path = None):
    print("Using device: ", DEVICE)
    print("Available GPUs:", torch.cuda.device_count())

    # 获取数据集
    train_set, val_set, test_set = get_dataset("ImageNet1k_64")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = val_loader

    # 初始化模型和损失函数
    criterion = nn.CrossEntropyLoss()
    model = get_custom_mobilenet_model(num_classes = 1000)
    
    # 如果有预训练权重，加载权重
    load_weight_path = "./weights/fp32.pth"
    if load_weight_path is not None:
        print(f"Loading weights from {load_weight_path}")
        state_dict = torch.load(load_weight_path, map_location=DEVICE) # 直接加载到DEVICE
        # 适配可能存在的DataParallel前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict = False)

    # 多卡并行处理, 并移动模型到DEVICE
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

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
                start_factor=1e-6,
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

