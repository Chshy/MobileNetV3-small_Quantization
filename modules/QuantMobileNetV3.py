import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

from .quantized.quant_node import QuantNode
from .quantized.linear import QuantLinear, QuantLinearAct
from .quantized.conv import QuantConv2d, QuantConv2dAct, QuantConv2dBnAct
from .quantized.base_module import QuantBaseModule


import torch
import torch.nn as nn
from typing import Dict

class QuantMobileNetV3(QuantBaseModule):
    def __init__(self, config, num_classes=1000, quant_params=None):
        super().__init__()
        self.quant_params = quant_params or {}
        layers = []
        
        # 构建量化卷积块的辅助函数
        # def build_quant_conv_block(name, conv_cfg, in_ch):
        def build_quant_conv_block(conv_cfg, in_ch, quant_params):

            # qparams = self.quant_params.get(name, {})
            block = []
            
            # 量化卷积层
            conv = QuantConv2dBnAct(
                in_ch,
                conv_cfg['out_channels'],
                kernel_size=conv_cfg['kernel'],
                stride=conv_cfg['stride'],
                padding=conv_cfg['kernel'] // 2,
                bias=False,
                activation=nn.Hardswish() if conv_cfg.get('use_hs', False) else nn.ReLU(),
                **quant_params
            )
            block.append(conv)
            
            # SE模块量化
            if conv_cfg.get('use_se', False):
                se_qparams = self.quant_params.get(f"se", {})
                squeeze_channels = _make_divisible(
                    int(conv_cfg['out_channels'] * conv_cfg.get('se_ratio', 0.25)), 8
                )
                block.append(QuantSqueezeExcitation(
                    conv_cfg['out_channels'],
                    squeeze_channels,
                    **se_qparams
                ))
            
            return block, conv_cfg['out_channels']
        
        # 输入原始数据量化
        input_quant_params = self.quant_params.get("input_quant", {})
        self.input_quant = QuantNode(**input_quant_params)

        # 构建初始卷积层
        init_block, in_channels = build_quant_conv_block(
            config['init_conv'],
            config['input_channels'],
            self.quant_params.get("init_conv", {})
        )
        layers += init_block

        # 构建中间残差块

        # 目标长度（以 config['blocks'] 的长度为例）
        target_length = len(config['blocks'])
        # 获取现有列表（若不存在则初始化为空列表）
        existing_blocks = self.quant_params.get("blocks", [])
        # 动态调整长度：截断到目标长度，不足部分补空字典
        adjusted_blocks = existing_blocks[:target_length] + [{}] * max(target_length - len(existing_blocks), 0)

        # for block_cfg in config['blocks']:
        for idx, block_cfg in enumerate(config['blocks']):

            # 取得第 idx 个块的量化参数
            quant_params = adjusted_blocks[idx]

            layers.append(QuantInvertedResidual(
                in_channels=in_channels,
                out_channels=block_cfg['out_channels'],
                kernel_size=block_cfg['kernel'],
                stride=block_cfg['stride'],
                expansion_ratio=block_cfg.get('expansion', None),
                hidden_dim=block_cfg.get('exp_size', None),
                use_se=block_cfg.get('use_se', False),
                se_ratio=block_cfg.get('se_ratio', 0.25),
                activation=nn.Hardswish if block_cfg.get('use_hs', False) else nn.ReLU,
                **quant_params
            ))
            in_channels = block_cfg['out_channels']

        # 构建最终卷积层
        final_block, final_out_ch = build_quant_conv_block(
            config['final_conv'],
            in_channels,
            self.quant_params.get("final_conv", {})
        )
        layers += final_block
 

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 量化分类器

        classifier_qparams = self.quant_params.get("classifier", {})
        # self.classifier = nn.Sequential(
        #     QuantLinearAct(
        #         # config['final_conv']['out_channels'],
        #         final_out_ch,
        #         config.get('classifier_hidden_dim', 1280),
        #         activation=nn.Hardswish(),
        #         *classifier_qparams.get("fc1", {})
        #     ),
        #     nn.Dropout(p=0.8),
        #     QuantLinear(
        #         config.get('classifier_hidden_dim', 1280),
        #         num_classes,
        #         *classifier_qparams.get("fc2", {})
        #     )
        # )
        self.classifier = nn.Sequential(
            nn.Linear(final_out_ch, config.get('classifier_hidden_dim', 1280)),
            nn.Hardswish(inplace = False),
            nn.Dropout(p=0.8),
            nn.Linear(config.get('classifier_hidden_dim', 1280), num_classes)
        )
    
    def forward(self, x):
        # print(x)
        # print(f"from QuantMobileNetV3, type(x): {type(x)}")

        x = self.input_quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class QuantInvertedResidual(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_ratio=None,
                 hidden_dim=None,
                 use_se=True,
                 se_ratio=0.25,
                 activation=nn.ReLU,
                 quant_params=None):
        super().__init__()
        quant_params = quant_params or {}
        
        # 计算hidden_dim
        if hidden_dim is None:
            hidden_dim = int(in_channels * expansion_ratio)
            
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        # 量化扩展卷积
        self.conv1 = QuantConv2dBnAct(
            in_channels, hidden_dim, kernel_size=1, bias=False,
            activation=activation(),
            **quant_params.get("conv1", {})
        )
        
        # 量化深度卷积
        self.conv2 = QuantConv2dBnAct(
            hidden_dim, hidden_dim, kernel_size, bias = False,
            stride=stride,
            groups=hidden_dim,
            padding=kernel_size//2,
            activation=activation(),
            **quant_params.get("conv2", {})
        )
        
        # 量化SE模块
        if use_se:
            self.se = QuantSqueezeExcitation(
                hidden_dim,
                _make_divisible(int(hidden_dim * se_ratio), 8),
                quant_params=quant_params.get("se", {})  # 关键修改：移除了**解包
            )
        else:
            self.se = None
            
        # 量化投影卷积
        self.conv3 = QuantConv2dBnAct(
            hidden_dim, out_channels, kernel_size=1, bias=False,
            activation=None,
            **quant_params.get("conv3", {})
        )
        
        # # 残差连接量化
        # if self.use_residual:
        #     self.residual_quant = QuantNode(**quant_params.get("residual", {}))
        # TODO 这里残差连接量化需要重新考虑！！！

    def forward(self, x):
        residual = x

        # print(x)
        # print(f"from QuantInvertedResidual, type(x): {type(x)}")
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.se:
            x = self.se(x)
            
        x = self.conv3(x)
        
        if self.use_residual:
            # print(f"from QuantInvertedResidual, x.size()): {x.size()}")
            # print(f"from QuantInvertedResidual, residual.size()): {residual.size()}")
            x += residual
            
        return x

class QuantSqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels, quant_params=None):
        super().__init__()
        quant_params = quant_params or {}
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = QuantConv2dAct(
            input_channels, squeeze_channels, kernel_size=1,
            activation=nn.ReLU(),
            **quant_params.get("fc1", {})
        )
        self.fc2 = QuantConv2dAct(
            squeeze_channels, input_channels, kernel_size=1,
            activation=nn.Hardsigmoid(),
            **quant_params.get("fc2", {})
        )
        
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        return x * scale

# # 示例量化配置
# quant_config = {
#     "init_conv": {"weight_num_bits": 8, "act_num_bits": 8},
#     "blocks.0": {
#         "conv1": {"weight_num_bits": 4, "act_num_bits": 4},
#         "conv2": {"weight_num_bits": 4, "act_num_bits": 4},
#         "se": {"fc1": {"weight_num_bits": 4}, "fc2": {"weight_num_bits": 4}}
#     },
#     "classifier": {"weight_num_bits": 8, "act_num_bits": 8}
# }

# # 创建量化模型并加载权重
# def mobilenet_v3_large_quant(num_classes=10, quant_params=None):
#     fp_config = mobilenet_v3_large().config  # 假设可以获取原始配置
#     model = QuantMobileNetV3(fp_config, num_classes, quant_params)
#     return model

# # 使用示例
# if __name__ == '__main__':
#     # 加载浮点模型
#     fp_model = mobilenet_v3_large()
    
#     # 创建量化模型
#     quant_model = mobilenet_v3_large_quant(quant_params=quant_config)
    
#     # 加载权重
#     quant_model.load_from_fp_model(fp_model)
    
#     # 设置量化状态
#     quant_model.set_quant_state(quant_on=False, calib_on=True)
    
#     # 校准模型
#     quant_model.calibrate(calib_loader)
    
#     # 启用推理量化
#     quant_model.set_quant_state(quant_on=True, calib_on=False)

# # # 预配置函数改造
# # def quant_mobilenet_v3_large(num_classes=10):
# #     config = mobilenet_v3_large(num_classes).config  # 复用原始配置
# #     return QuantMobileNetV3(config, num_classes=num_classes)

# # def quant_mobilenet_v3_small(num_classes=10):
# #     config = mobilenet_v3_small(num_classes).config  # 复用原始配置
# #     return QuantMobileNetV3(config, num_classes=num_classes)