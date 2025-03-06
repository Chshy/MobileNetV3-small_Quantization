import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import os
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.insert(0, project_root)
from quant_modules.basic.calib import QuantCalibrator
from quant_modules.computations.conv import QuantConv2d, QuantConv2dAct, QuantConv2dBnAct
from quant_modules.computations.linear import QuantLinear, QuantLinearAct
from quant_modules.computations.conv import QuantNode

class Observer:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.data = defaultdict(lambda: defaultdict(list))

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (QuantNode, QuantConv2d, QuantConv2dAct, QuantConv2dBnAct, QuantLinear, QuantLinearAct)):
                # 注册前向pre_hook捕获输入
                hook_pre = module.register_forward_pre_hook(self._create_pre_hook(name))
                self.hooks.append(hook_pre)
                # 注册forward_hook捕获输出和权重/bias
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _create_pre_hook(self, name):
        def hook(module, inputs):
            if isinstance(inputs, tuple) and len(inputs) > 0:
                input_data = inputs[0].detach().cpu()
                self.data[name]['input'].append(input_data)
        return hook

    def _create_hook(self, name):
        def hook(module, inputs, output):
            # 捕获输出
            output_data = output.detach().cpu()
            self.data[name]['output'].append(output_data)
            
            # 处理权重和bias
            if hasattr(module, 'process_weight'):
                proc_weight, proc_bias = module.process_weight()
                self.data[name]['weight'].append(proc_weight.detach().cpu())
                if proc_bias is not None:
                    self.data[name]['bias'].append(proc_bias.detach().cpu())
        return hook

    def clear_data(self):
        self.data.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def plot_distributions(self, max_samples=1000, save_dir='./observer_plots'):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        for layer_name in self.data:
            for key in ['input', 'output', 'weight', 'bias']:
                if key not in self.data[layer_name]:
                    continue
                
                # 合并数据并采样防止内存不足
                all_data = torch.cat(self.data[layer_name][key])
                if len(all_data) > max_samples:
                    indices = torch.randperm(len(all_data))[:max_samples]
                    all_data = all_data[indices]
                
                plt.figure(figsize=(10, 6))
                # sns.histplot(all_data.numpy().flatten(), kde=True, bins=50)
                sns.histplot(
                    all_data.numpy().flatten(),
                    kde=True,
                    bins=50,
                    stat='density',  # 关键修改：启用密度归一化
                    element='bars',  # 可选：使用柱状图而非step填充
                    fill=False       # 可选：关闭填充以更清晰显示KDE曲线
                )
                plt.title(f"{layer_name} {key} Distribution")
                plt.xlabel("Value")
                plt.ylabel("Density")
                
                # 生成文件名并保存
                filename = f"{layer_name.replace('.', '_')}_{key}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()  # 关闭当前图像防止内存泄漏

        print(f"All plots saved to {os.path.abspath(save_dir)}")
