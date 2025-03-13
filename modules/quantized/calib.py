# 校准函数(统计量化参数时使用)

import os
import sys
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# sys.path.insert(0, project_root)

# from quant_modules.basic.quantize import quantize, dequantize
# from quant_modules.basic.pseudo_quantize import pseudo_quantize

import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantCalibrator:

    def __init__(self, num_bits=8, symmetric=True, signed = True, custom_quant_range = None, enable_ema=True, ema_beta=0.9):
        r"""初始化量化校准模块

        Args:
            num_bits:           量化的目标比特数
            symmetric:          是否使用对称量化
            signed:             输出是有符号数还是无符号数
            custom_quant_range: 自定义量化范围，格式为 (min_val, max_val)，如果为 None 则使用自动计算的默认范围
            enable_ema:         统计min/max时是否启用EMA
            ema_beta:           EMA算法的衰减系数
        """
        self.num_bits = num_bits
        self.signed = signed
        self.symmetric = symmetric

        self.enable_ema = enable_ema
        self.ema_beta = ema_beta
        
        self.min_val = None
        self.max_val = None
        self.scale = 1.0
        self.zero_point = 0.0

        # 计算输出dtype、量化范围(输出值截断的范围)
        # self.output_dtype, self.Q_MIN, self.Q_MAX = self.calculate_dtype_qmin_qmax(self.num_bits, self.signed, custom_quant_range)
        self.Q_MIN, self.Q_MAX = self.calculate_dtype_qmin_qmax(self.num_bits, self.signed, custom_quant_range)

    def calculate_dtype_qmin_qmax(self, num_bits, signed, custom_quant_range): # TODO
        '''根据输入的参数计算输出的dtype和量化范围'''

        # 计算Q_MIN和Q_MAX

        # 先计算若没有自定义范围的话，最大能够覆盖的范围
        if signed:
            if self.symmetric:
                Q_MIN = -(2 ** (num_bits - 1) - 1)
                Q_MAX = 2 ** (num_bits - 1) - 1
            else:
                Q_MIN = -(2 ** (num_bits - 1))
                Q_MAX = 2 ** (num_bits - 1) - 1
        else:
            Q_MIN = 0
            Q_MAX = 2 ** num_bits - 1

        # 处理自定义范围的情况
        if custom_quant_range is not None:
            custom_qmin, custom_qmax = custom_quant_range

            # 首先检查自定义输出范围是否合法
            # 1. 范围必须包含0
            if 0 not in (custom_qmin, custom_qmax):
                raise ValueError("Custom quantization range must contain 0")
            # 2. 范围左端点必须比右端点小
            if custom_qmin >= custom_qmax:
                raise ValueError("Custom quantization range must be in ascending order")
            # 3. 范围必须在允许的范围内
            # if custom_qmax - custom_qmin > Q_MAX - Q_MIN:
            if not Q_MIN <= custom_qmin <= custom_qmax <= Q_MAX:
                raise ValueError("Custom quantization range out of allowed range")
            # 4. 范围必须是整数
            if not (isinstance(custom_qmin, int) and isinstance(custom_qmax, int)):
                raise ValueError("Custom quantization range must be integers")

            Q_MIN, Q_MAX = custom_qmin, custom_qmax
            
        # return output_dtype, Q_MIN, Q_MAX
        return Q_MIN, Q_MAX
        
    def ema_update(self, ema, value):
        """Exponential Moving Average更新"""
        if ema is None:
            return value
        else:
            return self.ema_beta * ema + (1 - self.ema_beta) * value
        
    def calculate_scale_and_zero_point(self):
        """根据min/max计算量化参数"""

        # 首先让统计范围包括0
        self.min_val = min(self.min_val, 0)
        self.max_val = max(self.max_val, 0)

        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            if self.signed:
                # 有符号对称
                # 0 → 0
                # -max_abs → Q_MIN, max_abs → Q_MAX
                self.scale = max_abs / self.Q_MAX
                self.zero_point = 0.0
            else:
                # 无符号对称
                # 0 → middle(Q_MIN, Q_MAX)
                # -max_abs → Q_MIN, max_abs → Q_MAX
                self.scale = (2 * max_abs) / (self.Q_MAX - self.Q_MIN)
                self.zero_point = round((self.Q_MAX + self.Q_MIN) / 2)
            
        else:
            # 非对称
            # min_val → Q_MIN, max_val → Q_MAX
            self.scale = (self.max_val - self.min_val) / (self.Q_MAX - self.Q_MIN)
            self.zero_point = self.Q_MIN - round(self.min_val / self.scale)

        return

    def calibrate(self, tensor):
        """校准量化参数"""
        current_min = tensor.min().item()
        current_max = tensor.max().item()
        
        if self.enable_ema:
            self.min_val = self.ema_update(self.min_val, current_min)
            self.max_val = self.ema_update(self.max_val, current_max)
            
        else:
            self.min_val = current_min
            self.max_val = current_max
            

        # 计算并更新量化参数
        self.calculate_scale_and_zero_point()

    def get_params(self):
        """获取量化参数"""
        return self.scale, self.zero_point
    
    def quantize(self, tensor):
        """量化张量"""
        quantized = torch.clamp(RoundSTE.apply(tensor / self.scale + self.zero_point).to(torch.int), self.Q_MIN, self.Q_MAX)
        recovered = (quantized - self.zero_point) * self.scale
        return recovered

