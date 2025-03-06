import os
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.insert(0, project_root)

from quant_modules.basic.calib import QuantCalibrator

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

class BaseQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, 
                 weight_num_bits=8, 
                 weight_symmetric=True,
                 weight_signed=True,
                 bias_num_bits=8, 
                 bias_symmetric=True,
                 bias_signed=True,
                 act_num_bits=8, 
                 act_symmetric=False,
                 act_signed=True,
                 init_linear=None,
                 **kwargs):
        if init_linear is not None:
            super().__init__(init_linear.in_features, init_linear.out_features, bias=init_linear.bias is not None)
            self.load_state_dict(init_linear.state_dict())
        else:
            super().__init__(in_features, out_features, **kwargs)

        self.weight_quant = QuantCalibrator(num_bits=weight_num_bits, symmetric=weight_symmetric, signed=weight_signed, enable_ema=False)
        self.bias_quant = QuantCalibrator(num_bits=bias_num_bits, symmetric=bias_symmetric, signed=bias_signed, enable_ema=False) if self.bias is not None else None
        self.act_quant = QuantCalibrator(num_bits=act_num_bits, symmetric=act_symmetric, signed=act_signed, enable_ema=True)
        self.enable_quant = True
        self.enable_calib = True

    def process_weight(self):
        if self.enable_calib:
            self.weight_quant.calibrate(self.weight)
            if self.bias is not None and self.bias_quant is not None:
                self.bias_quant.calibrate(self.bias)
        if self.enable_quant:
            proc_weight = self.weight_quant.quantize(self.weight)
            proc_bias = self.bias_quant.quantize(self.bias) if self.bias is not None else None
        else:
            proc_weight = self.weight
            proc_bias = self.bias
        return proc_weight, proc_bias

    def process_result(self, result):
        if self.enable_calib:
            self.act_quant.calibrate(result)
        return self.act_quant.quantize(result) if self.enable_quant else result

class QuantLinear(BaseQuantLinear):
    def forward(self, x):
        proc_weight, proc_bias = self.process_weight()
        result = F.linear(x, proc_weight, proc_bias)
        return self.process_result(result)

class QuantLinearAct(BaseQuantLinear):
    def __init__(self, *args, activation=nn.ReLU(), **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation

    def forward(self, x):
        proc_weight, proc_bias = self.process_weight()
        result = F.linear(x, proc_weight, proc_bias)
        if self.activation is not None:
            result = self.activation(result)
        return self.process_result(result)

