import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Callable

class QuantBaseModule(nn.Module):
    """
    量化模型基类，提供量化相关通用功能
    """
    
    def __init__(self):
        super().__init__()
    
    def set_quant_state(self, quant_on: bool = True, calib_on: bool = False) -> None:
        """
        设置全模型量化/校准状态
        Args:
            quant_on: 是否启用量化
            calib_on: 是否启用校准
        """
        for module in self.modules():
            # print(f"Setting attribute: {module.__class__.__name__} enable_quant: {quant_on}, enable_calib: {calib_on}")
            if hasattr(module, 'enable_quant') and hasattr(module, 'enable_calib'):
                module.enable_quant = quant_on
                module.enable_calib = calib_on

    def load_from_fp_model(
        self,
        fp_model: nn.Module,
        custom_mapping: Optional[Union[Dict[str, str], Callable[[str], str]]] = None
    ) -> None:
        """
        从浮点模型加载权重
        Args:
            fp_model: 原始浮点模型
            custom_mapping: 层名称映射规则，支持字典或函数形式
        """
        state_dict = fp_model.state_dict()
        mapped_dict = {}
        
        # 创建目标模型的状态字典副本
        target_dict = self.state_dict()
        
        for name, param in state_dict.items():
            # 处理自定义名称映射
            target_name = name
            if custom_mapping:
                if isinstance(custom_mapping, dict):
                    target_name = custom_mapping.get(name, name)
                elif callable(custom_mapping):
                    target_name = custom_mapping(name)
                else:
                    raise TypeError("custom_mapping should be dict or callable")
            
            # 验证参数匹配性
            if target_name in target_dict:
                if param.shape == target_dict[target_name].shape:
                    mapped_dict[target_name] = param
                else:
                    print(f"Ignoring {name} -> {target_name}: shape mismatch")
            else:
                print(f"Ignoring {name}: no corresponding layer found")
        
        # 加载处理后的权重
        self.load_state_dict(mapped_dict, strict=False)

        print(f"Loaded {len(mapped_dict)}/{len(state_dict)} params from fp_model")
    
    def print_quant_params(self, verbose: bool = False) -> None:
        """
        打印所有量化参数
        Args:
            verbose: 是否显示详细信息
        """
        print("\nQuantization Parameters:")
        for name, module in self.named_modules():
            # print(f"{name}")
            # print(getattr(module, f"weight_quant", None))
            # print(getattr(module, f"bias_quant", None))
            # print(getattr(module, f"act_quant", None))
            params = []
            
            # 收集各量化器参数
            for qtype in ['weight', 'bias', 'act']:
                quantizer = getattr(module, f"{qtype}_quant", None)
                if quantizer:
                    scale, zp = quantizer.get_params()
                    params.append(f"{qtype.upper()}: scale={scale:.6f} zp={zp:.2f}")
            
            if params:
                # print(f"[{name}]")
                print(f"[{name}]", end="")
                if verbose:
                    config = []
                    if hasattr(module, 'num_bits'):
                        config.append(f"Bits: {module.num_bits}")
                    if hasattr(module, 'symmetric'):
                        config.append(f"Symmetric: {module.symmetric}")
                    if config:
                        print("  Config: " + ", ".join(config))
                print("  " + " | ".join(params))
    
    def calibrate(
        self,
        calib_loader,
        device: str = "cuda",
        num_batches: int = 100,
        verbose: bool = True
    ) -> None:
        """
        执行模型校准
        Args:
            calib_loader: 校准数据加载器
            device: 计算设备
            num_batches: 使用的批次数量
            verbose: 是否显示进度
        """
        self.to(device)
        self.train()  # 保持BN的运行统计
        
        # 设置校准模式
        self.set_quant_state(quant_on=False, calib_on=True)
        
        # 运行校准
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calib_loader):
                if i >= num_batches:
                    break
                if verbose:
                    print(f"\rCalibrating batch {i+1}/{num_batches}", end="")
                inputs = inputs.to(device)
                self(inputs)
        
        # 恢复量化状态
        self.set_quant_state(quant_on=True, calib_on=False)
        if verbose:
            print("\nCalibration completed.")

    def _quant_layers(self):
        """遍历所有量化层"""
        return [m for m in self.modules() if hasattr(m, 'enable_quant')]

    def extra_repr(self) -> str:
        """显示模型量化状态"""
        quant_layers = self._quant_layers()
        return f"Quant status: {len(quant_layers)} layers\n" + \
               f"  - Quant enabled: {quant_layers[0].enable_quant}\n" + \
               f"  - Calib enabled: {quant_layers[0].enable_calib}"