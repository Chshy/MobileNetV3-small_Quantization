import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

import math

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation模块"""

    def __init__(self, input_channels, squeeze_channels):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace = False)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid(inplace = False)

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """InvertedResidual模块"""

    def __init__(self, 
                 in_channels,  # 卷积1输入
                 out_channels, # 卷积3输出
                 kernel_size,  # 卷积2的kernel size
                 stride,       # 卷积2的stride

                 # 以下参数用于计算卷积2的输入/输出通道数
                 expansion_ratio = None,
                 hidden_dim = None, 

                 # 以下参数用于控制SqueezeExcitation模块中间的通道数
                 use_se = True,
                 se_ratio = None,
                 
                 activation = nn.ReLU # 激活函数
                ):
        
        super().__init__()

        # 计算 hidden dim
        if hidden_dim is not None and expansion_ratio is not None:
            raise ValueError("🚫Error: Only one of hidden_dim or expansion_ratio can be provided")
        if hidden_dim is None and expansion_ratio is None:
            raise ValueError("🚫Error: Either hidden_dim or expansion_ratio must be provided")
        if hidden_dim is None: # 使用 expansion_ratio 计算 hidden_dim
            hidden_dim = int(in_channels * expansion_ratio)

        # 当且仅当 *输入输出通道/尺寸匹配* 时, 使用残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        # Expansion phase
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = activation(inplace=False)
        
        # Depthwise convolution
        # 卷积2需要使用padding 使得输入和输出尺寸一致
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, 
            padding=kernel_size//2, groups=hidden_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = activation(inplace=False)
        
        # Squeeze-and-Excitation
        if use_se:
            if se_ratio is None:
                se_ratio = 0.25
                # SE中间层channel = SE输出层channel * se_ratio
                # 论文第5.3节, MobileNetV3将SE模块的中间通道数固定为扩展层通道数的1/4
            squeeze_channel = _make_divisible(int(hidden_dim * se_ratio), 8) # 计算SE中间层channel 确保squeeze_channel为8的倍数
            self.se = SqueezeExcitation(hidden_dim, squeeze_channel)
        else:
            self.se = None
        
        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        if self.se:
            x = self.se(x)
            
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x += residual
            
        return x

class MobileNetV3(nn.Module):
    """MobileNetV3 Base Class"""
    def __init__(self, config, num_classes=1000, dropout=0.8):
        super().__init__()
        layers = []
        
        # 创建卷积块的辅助函数
        def build_conv_block(conv_cfg, in_ch):

            # 创建基础卷积层
            block = [
                nn.Conv2d(
                    in_ch,
                    conv_cfg['out_channels'],
                    kernel_size=conv_cfg['kernel'],
                    stride=conv_cfg['stride'],
                    padding=conv_cfg['kernel'] // 2,
                    bias=False
                ),
                nn.BatchNorm2d(conv_cfg['out_channels']),
                nn.Hardswish(inplace=False) if conv_cfg.get('use_hs', False) else nn.ReLU(inplace=False)
            ]
            
            # 添加SE模块（如果需要）
            if conv_cfg.get('use_se', False):
                se_ratio = conv_cfg.get('se_ratio', 0.25)
                # 注意：SE的输入通道应该是当前层的输出通道
                squeeze_channels = _make_divisible(
                    int(conv_cfg['out_channels'] * se_ratio), 8
                )
                block.append(SqueezeExcitation(
                    conv_cfg['out_channels'],  # 修复原init_conv的通道错误
                    squeeze_channels
                ))
            
            # 返回通道信息用于后续处理
            return block, conv_cfg['out_channels']
        
        # 构建初始卷积层
        init_block, in_channels = build_conv_block(
            config['init_conv'],
            config['input_channels']
        )
        layers += init_block
        
        # 构建中间残差块
        for block_cfg in config['blocks']:
            layers.append(InvertedResidual(
                in_channels=in_channels,
                out_channels=block_cfg['out_channels'],
                kernel_size=block_cfg['kernel'],
                stride=block_cfg['stride'],
                expansion_ratio=block_cfg.get('expansion', None),
                hidden_dim=block_cfg.get('exp_size', None),
                use_se=block_cfg.get('use_se', False),
                se_ratio=block_cfg.get('se_ratio', 0.25),
                activation=nn.Hardswish if block_cfg.get('use_hs', False) else nn.ReLU
            ))
            in_channels = block_cfg['out_channels']
        
        # 构建最终卷积层
        final_block, final_out_ch = build_conv_block(
            config['final_conv'],
            in_channels
        )
        layers += final_block
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(final_out_ch, config.get('classifier_hidden_dim', 1280)),
            nn.Hardswish(inplace = False),
            nn.Dropout(p=dropout),
            nn.Linear(config.get('classifier_hidden_dim', 1280), num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# 预定义配置
def mobilenet_v3_large(num_classes=10):
    config = {
        'input_channels': 3, # 输入图片的通道数
        'init_conv': {'kernel': 3, 'out_channels': 16, 'use_se': False, 'use_hs': True, 'stride': 2},
        'blocks': [
            {'kernel': 3, 'exp_size':16, 'out_channels': 16, 'use_se': False, 'use_hs': False, 'stride': 1},
            {'kernel': 3, 'exp_size':64, 'out_channels': 24, 'use_se': False, 'use_hs': False, 'stride': 2},
            {'kernel': 3, 'exp_size':72, 'out_channels': 24, 'use_se': False, 'use_hs': False, 'stride': 1},
            {'kernel': 5, 'exp_size':72, 'out_channels': 40, 'use_se': True, 'use_hs': False, 'stride': 2},
            {'kernel': 5, 'exp_size':120, 'out_channels': 40, 'use_se': True, 'use_hs': False, 'stride': 1},
            {'kernel': 5, 'exp_size':120, 'out_channels': 40, 'use_se': True, 'use_hs': False, 'stride': 1},
            {'kernel': 3, 'exp_size':240, 'out_channels': 80, 'use_se': False, 'use_hs': True, 'stride': 2},
            {'kernel': 3, 'exp_size':200, 'out_channels': 80, 'use_se': False, 'use_hs': True, 'stride': 1},
            {'kernel': 3, 'exp_size':184, 'out_channels': 80, 'use_se': False, 'use_hs': True, 'stride': 1},
            {'kernel': 3, 'exp_size':184, 'out_channels': 80, 'use_se': False, 'use_hs': True, 'stride': 1},
            {'kernel': 3, 'exp_size':480, 'out_channels': 112, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 3, 'exp_size':672, 'out_channels': 112, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':672, 'out_channels': 160, 'use_se': True, 'use_hs': True, 'stride': 2},
            {'kernel': 5, 'exp_size':960, 'out_channels': 160, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':960, 'out_channels': 160, 'use_se': True, 'use_hs': True, 'stride': 1},
        ],
        'final_conv': {'kernel': 1, 'out_channels': 960, 'use_se': False, 'use_hs': True, 'stride': 1},
        'classifier_hidden_dim': 1280
    }
    return MobileNetV3(config, num_classes = num_classes)

def mobilenet_v3_small(num_classes=10):
    config = {
        'input_channels': 3,
        'init_conv': {'kernel': 3, 'out_channels': 16, 'use_se': False, 'use_hs': True, 'stride': 2},
        'blocks': [
            {'kernel': 3, 'exp_size':16, 'out_channels': 16, 'use_se': True, 'use_hs': False, 'stride': 2},
            {'kernel': 3, 'exp_size':72, 'out_channels': 24, 'use_se': False, 'use_hs': False, 'stride': 2},
            {'kernel': 3, 'exp_size':88, 'out_channels': 24, 'use_se': False, 'use_hs': False, 'stride': 1},
            {'kernel': 5, 'exp_size':96, 'out_channels': 40, 'use_se': True, 'use_hs': True, 'stride': 2},
            {'kernel': 5, 'exp_size':240, 'out_channels': 40, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':240, 'out_channels': 40, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':120, 'out_channels': 48, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':144, 'out_channels': 48, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':288, 'out_channels': 96, 'use_se': True, 'use_hs': True, 'stride': 2},
            {'kernel': 5, 'exp_size':576, 'out_channels': 96, 'use_se': True, 'use_hs': True, 'stride': 1},
            {'kernel': 5, 'exp_size':576, 'out_channels': 96, 'use_se': True, 'use_hs': True, 'stride': 1},
        ],
        'final_conv': {'kernel': 1, 'out_channels': 576, 'use_se': True, 'use_hs': True, 'stride': 1},
        'classifier_hidden_dim': 1024
    }
    return MobileNetV3(config, num_classes = num_classes)

# 示例用法
if __name__ == '__main__':
    model_large = mobilenet_v3_large()
    model_small = mobilenet_v3_small()
    
    # 自定义配置示例
    custom_config = {
        'input_channels': 3,
        'init_conv': {'kernel': 3, 'out_channels': 24, 'use_se': False, 'use_hs': False, 'stride': 2},
        'blocks': [
            {'kernel': 3, 'expansion':2, 'out_channels': 32, 'use_se': True, 'use_hs': False, 'stride': 1},
            {'kernel': 5, 'expansion':4, 'out_channels': 64, 'use_se': False, 'use_hs': False, 'stride': 2},
        ],
        'final_conv': {'kernel': 1, 'out_channels': 512, 'use_se': True, 'use_hs': True, 'stride': 1},
        'classifier_hidden_dim': 1024
    }
    model_custom = MobileNetV3(custom_config, num_classes=10)