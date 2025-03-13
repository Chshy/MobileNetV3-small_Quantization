import os
import sys
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# sys.path.insert(0, project_root)

# # from quant_modules.basic.quantize import quantize, dequantize
# # from quant_modules.basic.pseudo_quantize import pseudo_quantize
# from quant_modules.basic.calib import QuantCalibrator

from .calib import QuantCalibrator

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings


# 校准模式: 浮点数运算，统计输入和输出
#           权重是固定的 不需要数据就能统计
# QAT模式: 伪量化生效，统计输入和输出
#          权重不使用ema统计

# class QuantNode(nn.Module):
#     def __init__(self, act_num_bits=8, act_symmetric=True, act_signed=True):
#         super().__init__()

#         self.act_quant = QuantCalibrator(num_bits=act_num_bits, symmetric=act_symmetric, signed=act_signed, enable_ema=True)
        
#         self.enable_quant = True
#         self.enable_calib = True
    
#     def process_result(self, result):
#         if self.enable_calib:
#             self.act_quant.calibrate(result)
#         if self.enable_quant:
#             proc_result = self.act_quant.quantize(result)
#         else:
#             proc_result = result
#         return proc_result
    
#     def forward(self, x):
#         return self.process_result(x)



class BaseQuantConv2d(nn.Conv2d):
    def __init__(self, *args, 
                 weight_num_bits=8, 
                 weight_symmetric=True,
                 weight_signed=True,
                 bias_num_bits=8, 
                 bias_symmetric=True,
                 bias_signed=True,
                 act_num_bits=8, 
                 act_symmetric=False,
                 act_signed=True,
                 init_conv=None, 
                 **kwargs):
        
        if init_conv is not None:

            # 检查是否传入了其他参数，如果有则报错
            if len(args) > 0 or len(kwargs) > 0:
                raise ValueError("🚫Error: 当使用 'init_conv' 初始化时, 不可传入其他参数")
            
            params = {
                'in_channels': init_conv.in_channels,
                'out_channels': init_conv.out_channels,
                'kernel_size': init_conv.kernel_size,
                'stride': init_conv.stride,
                'padding': init_conv.padding,
                'dilation': init_conv.dilation,
                'groups': init_conv.groups,
                'bias': init_conv.bias is not None,
                'padding_mode': init_conv.padding_mode
            }
            super().__init__(**params)
            # self.weight.data.copy_(init_conv.weight.data)
            # if params['bias']:
            #     self.bias.data.copy_(init_conv.bias.data)
            self.load_state_dict(init_conv.state_dict())
        else:
            # 若无 init_conv 则使用默认方法初始化
            super().__init__(*args, **kwargs)
        
        # 初始化量化器
        self.weight_quant = QuantCalibrator(num_bits=weight_num_bits, symmetric=weight_symmetric, signed=weight_signed, enable_ema=False)
        if self.bias is not None:
            self.bias_quant = QuantCalibrator(num_bits=bias_num_bits, symmetric=bias_symmetric, signed=bias_signed, enable_ema=False)
        self.act_quant = QuantCalibrator(num_bits=act_num_bits, symmetric=act_symmetric, signed=act_signed, enable_ema=True)
        
        self.enable_quant = True
        self.enable_calib = True

    def process_weight(self):
        # 校准并量化权重和偏置
        if self.enable_calib:
            self.weight_quant.calibrate(self.weight)
            if self.bias is not None:
                self.bias_quant.calibrate(self.bias)

        if self.enable_quant:
            proc_weight = self.weight_quant.quantize(self.weight)
            if self.bias is not None:
                proc_bias = self.bias_quant.quantize(self.bias)
            else:
                proc_bias = None
        else:
            proc_weight = self.weight
            proc_bias = self.bias

        return proc_weight, proc_bias
       
    def process_result(self, result):
        # 校准并量化输出
        if self.enable_calib:
            self.act_quant.calibrate(result)
        if self.enable_quant:
            proc_result = self.act_quant.quantize(result)
        else:
            proc_result = result
        return proc_result

class QuantConv2d(BaseQuantConv2d):
    def __init__(self, *args, 
                 act_num_bits=8, 
                 act_symmetric=True,  # 默认输出对称量化
                 **kwargs):
        """
        初始化方法：
        - args/kwargs: 卷积层参数
        - init_conv: 用于初始化的现有卷积层（可选）
        """
        super().__init__(*args, act_num_bits=act_num_bits, act_symmetric=act_symmetric, **kwargs)

    def forward(self, x):
        proc_weight, proc_bias = self.process_weight()
        result = F.conv2d(x, proc_weight, proc_bias, self.stride, self.padding, self.dilation, self.groups)
        return self.process_result(result)

class QuantConv2dAct(BaseQuantConv2d):
    def __init__(self, *args, 
                 activation=nn.ReLU(), 
                 act_num_bits=8,
                 act_symmetric=True,  # 默认输出对称量化
                 **kwargs):
        """
        初始化方法：
        - args/kwargs: 卷积层参数
        - init_conv: 用于初始化的现有卷积层（可选）
        - activation: 激活函数（默认ReLU）
        """
        super().__init__(*args, act_num_bits=act_num_bits, act_symmetric=act_symmetric, **kwargs)
        self.activation = activation

    def forward(self, x):
        proc_weight, proc_bias = self.process_weight()
        result = F.conv2d(x, proc_weight, proc_bias, self.stride, self.padding, self.dilation, self.groups)
        if self.activation is not None:
            result = self.activation(result)
        return self.process_result(result)
    
class QuantConv2dBnAct(BaseQuantConv2d):
    def __init__(self, *args, 
                 init_conv=None, 
                 init_bn=None, 
                 activation=nn.ReLU(),
                 act_num_bits=8,
                 act_symmetric=False,  # 默认输出非对称量化
                 **kwargs):
        """
        初始化方法：
        - args/kwargs: 卷积层参数
        - init_conv: 用于初始化的现有卷积层（可选）
        - init_bn: 用于初始化的现有批归一化层（可选）
        - activation: 激活函数（默认ReLU）
        """
        super().__init__(*args, act_num_bits=act_num_bits, act_symmetric=act_symmetric, init_conv=init_conv, **kwargs)
        
        if self.bias is not None:
            warnings.warn("⚠️Warning: 当使用 BatchNorm 时, 卷积层不应包含偏置, BatchNorm 会抵消偏置的作用", UserWarning)
        
        # 初始化BatchNorm层
        self.bn = nn.BatchNorm2d(self.out_channels)
        if init_bn is not None:
            if init_bn.num_features != self.out_channels: # 验证卷积的输出通道数和 bn 的通道数匹配
                raise ValueError(f"🚫Error: BatchNorm通道数({init_bn.num_features})与卷积输出通道({self.out_channels})不匹配")
            self.bn.load_state_dict(init_bn.state_dict())
        
        self.activation = activation

        # debuf
        # print(f"BaseQuantConv2d: in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, bn={self.bn is not None}, activation={self.activation is not None}")
        
        bias = getattr(self, "bias_quant", None)
        bias_bits = bias.num_bits if bias else "N/A"
        bias_sym = "T" if bias and bias.symmetric else "N"
        bias_sign = "S" if bias and bias.signed else "N"

        print(
            f"BaseQuantConv2d: W: {self.weight_quant.num_bits} "
            f"{'T' if self.weight_quant.symmetric else 'F'} "
            f"{'S' if self.weight_quant.signed else 'U'}, "
            
            f"B: {bias_bits} {bias_sym} {bias_sign}, "
            
            f"A: {self.act_quant.num_bits} "
            f"{'T' if self.act_quant.symmetric else 'F'} "
            f"{'S' if self.act_quant.signed else 'U'}"
        )



    def forward(self, x):
        proc_weight, proc_bias = self.process_weight()
        result = F.conv2d(x, proc_weight, proc_bias, self.stride, self.padding, self.dilation, self.groups)
        result = self.bn(result)
        if self.activation is not None:
            result = self.activation(result)
        return self.process_result(result)
