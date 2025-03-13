import torch.nn as nn
from .calib import QuantCalibrator

class QuantNode(nn.Module):
    def __init__(self, act_num_bits=8, act_symmetric=True, act_signed=True):
        super().__init__()

        self.act_quant = QuantCalibrator(num_bits=act_num_bits, symmetric=act_symmetric, signed=act_signed, enable_ema=True)
        
        self.enable_quant = True
        self.enable_calib = True
    
    def process_result(self, result):
        if self.enable_calib:
            self.act_quant.calibrate(result)
        if self.enable_quant:
            proc_result = self.act_quant.quantize(result)
        else:
            proc_result = result
        return proc_result
    
    def forward(self, x):
        return self.process_result(x)