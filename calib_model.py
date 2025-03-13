import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import get_dataset
from modules.MobileNetV3 import MobileNetV3
from utils.trainer import Trainer

import json

from modules.QuantMobileNetV3 import QuantMobileNetV3




def load_config_from_json(json_file):
    if json_file is not None:
        with open(json_file, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("json_file must be provided")
    return config






def model_key_mapping(fp_dict, quant_dict, print_info = True):
    
    # 获取按顺序排列的键列表
    fp_keys = list(fp_dict.keys())
    quant_keys = list(quant_dict.keys())
    # 确保两个列表长度相同
    assert len(fp_keys) == len(quant_keys), "Key Numbers Not Match."
    # 生成映射字典
    key_mapping = {fp_key: quant_key for fp_key, quant_key in zip(fp_keys, quant_keys)}

    if print_info:
        # 获取fp_keys最长字符串长度(为了打印美观)
        max_length = max(len(key) for key in fp_keys)

        # 打印权重映射关系
        print("权重映射关系:")
        print(f"{'Float Model Layers':<{max_length}} -> {'Quantized Model Layers':<{max_length}}")
        for k, v in key_mapping.items():
            print(f"{k:<{max_length}} -> {v:<{max_length}}")

    return key_mapping

def evaluate_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # print(type(images), type(labels))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

from collections import OrderedDict

def compare_models_layerwise(fp_model, quant_model, key_mapping, device='cuda', input_shape=(1, 3, 64, 64)):
    # 确保模型处于评估模式
    fp_model.eval()
    quant_model.eval()
    fp_model.to(device)
    quant_model.to(device)

    
    # 生成固定输入数据
    torch.manual_seed(42)
    input_data = torch.randn(*input_shape).to(device)
    
    # 存储各层输出
    fp_outputs = OrderedDict()
    quant_outputs = OrderedDict()

    # 获取模型的所有模块
    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quant_model.named_modules())

    # 构建模块映射关系（参数名 -> 模块实例）
    module_mapping = {}
    for fp_param, quant_param in key_mapping.items():
        # 提取模块名称（去掉参数类型后缀）
        fp_module_name = '.'.join(fp_param.split('.')[:-1])
        quant_module_name = '.'.join(quant_param.split('.')[:-1])
        
        # 获取对应模块实例
        if fp_module_name in fp_modules and quant_module_name in quant_modules:
            module_mapping[fp_modules[fp_module_name]] = quant_modules[quant_module_name]

    # 注册钩子的函数
    def register_hooks(model, output_dict, model_type):
        hooks = []
        for name, module in model.named_modules():
            def closure(m, name=name):
                def hook(module, input, output):
                    output_dict[f"{model_type}_{name}"] = output.detach()
                return hook
            hooks.append(module.register_forward_hook(closure(module)))
        return hooks

    # 为两个模型注册钩子
    fp_hooks = register_hooks(fp_model, fp_outputs, "fp")
    quant_hooks = register_hooks(quant_model, quant_outputs, "quant")

    # 前向传播
    with torch.no_grad():
        _ = fp_model(input_data)
        _ = quant_model(input_data)

    # 移除钩子
    for hook in fp_hooks + quant_hooks:
        hook.remove()

    # 构建反向映射：quant模块名 -> fp模块名
    reverse_mapping = {v: k for k, v in module_mapping.items()}

    # 逐层对比输出
    mismatch_found = False
    for quant_name, quant_out in quant_outputs.items():
        # 跳过没有对应的层
        if quant_name.split('_', 1)[1] not in ['.'.join(k.split('.')[:-1]) for k in key_mapping.values()]:
            continue

        # 找到对应的fp模块输出
        fp_module = reverse_mapping.get(quant_modules[quant_name.split('_', 1)[1]], None)
        if not fp_module:
            continue

        fp_name = f"fp_{list(fp_modules.keys())[list(fp_modules.values()).index(fp_module)]}"
        fp_out = fp_outputs.get(fp_name)

        if fp_out is None:
            print(f"⚠️ 未找到对应输出：{fp_name} -> {quant_name}")
            continue

        # 对比张量
        try:
            if not torch.allclose(fp_out, quant_out, atol=1e-5):
                print("\n" + "=" * 80)
                print(f"❌ 输出不匹配：\nFP: {fp_name}\nQuant: {quant_name}")
                print(f"FP shape: {fp_out.shape} | Quant shape: {quant_out.shape}")
                print(f"最大绝对误差: {torch.max(torch.abs(fp_out - quant_out)).item():.5e}")
                print(f"平均绝对误差: {torch.mean(torch.abs(fp_out - quant_out)).item():.5e}")
                print("=" * 80 + "\n")
                mismatch_found = True
                # break  # 发现第一个差异即停止
            else:
                print(f"✅ 匹配成功：{fp_name.ljust(50)} -> {quant_name}")
        except RuntimeError as e:
            print(f"⛔ 形状不匹配：{fp_name} ({fp_out.shape}) vs {quant_name} ({quant_out.shape})")
            mismatch_found = True
            break

    if not mismatch_found:
        print("\n🎉 所有对应层输出完全匹配！")

# DEVICE = 'cpu'
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # 从JSON加载配置文件
    model_json_file = "./config/model.json"
    config = load_config_from_json(model_json_file)
    quant_json_file = "./config/quantize.json"
    quant_params = load_config_from_json(quant_json_file)

    # conv_weight_qparams = {"weight_num_bits": 8, "weight_symmetric":True, "weight_signed": True}
    # conv_relu_qparams = {
    #     **conv_weight_qparams, 
    #     "act_num_bits": 8, "act_symmetric":False, "act_signed": False
    # }
    # conv_hswish_qparams = {
    #     **conv_weight_qparams, 
    #     "act_num_bits": 8, "act_symmetric":False, "act_signed": True
    # }
    # conv_hsigmoid_qparams = {
    #     **conv_weight_qparams, 
    #     "act_num_bits": 8, "act_symmetric":True, "act_signed": True
    # }
    


    # quant_params = {

    # }

    # 加载浮点数模型和权重
    fp_model = MobileNetV3(config = config, num_classes = 1000)
    load_weight_path = "./weights/fp32.pth"
    if load_weight_path is not None:
        print(f"Loading weights from {load_weight_path}")
        state_dict = torch.load(load_weight_path, map_location=DEVICE)
        # 适配可能存在的DataParallel前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        fp_model.load_state_dict(state_dict, strict = False)

    # 加载量化模型
    quant_model = QuantMobileNetV3(config = config, num_classes = 1000, quant_params = quant_params)


    # 生成从浮点数模型到量化模型的权重映射
    key_mapping = model_key_mapping(fp_model.state_dict(), quant_model.state_dict(), print_info = True)

    # 让量化模型加载浮点数模型的权重 作为初始权重
    quant_model.load_from_fp_model(fp_model, custom_mapping = key_mapping)

    fp_model.eval()
    quant_model.eval()

    # compare_models_layerwise(
    #     fp_model=fp_model,
    #     quant_model=quant_model,
    #     key_mapping=key_mapping,
    #     device=DEVICE,
    #     input_shape=(1, 3, 64, 64)  # 根据你的输入尺寸调整
    # )

    # 加载数据集
    train_set, val_set, test_set = get_dataset("ImageNet1k_64")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = val_loader
    

    # 测试浮点数精度
    print("开始测试原始模型精度(fp_model)...")
    fp32_accuracy = evaluate_model(fp_model, test_loader, DEVICE)
    print(f"原始模型的测试集准确率: {fp32_accuracy:.2f}%")


    quant_model.set_quant_state(quant_on = False, calib_on = False)
    print("开始测试量化前的模型精度(quant_model, enable_quant = False)...")
    fp32_accuracy = evaluate_model(quant_model, test_loader, DEVICE)
    print(f"量化前模型的测试集准确率: {fp32_accuracy:.2f}%")

    # 测试未校准精度
    quant_model.set_quant_state(quant_on = True, calib_on = False)
    print("开始测试量化后（未校准）的模型精度...")
    quant_accuracy = evaluate_model(quant_model, test_loader, DEVICE)
    print(f"量化后（未校准）模型的测试集准确率: {quant_accuracy:.2f}%")

    # 校准模型,并测试校准后的精度
    quant_model.set_quant_state(quant_on = False, calib_on = True)
    quant_model.calibrate(train_loader, DEVICE, num_batches=1000, verbose=True)
    quant_model.set_quant_state(quant_on = True, calib_on = False)

    print("开始测试量化后（已校准）的模型精度...")
    quant_accuracy = evaluate_model(quant_model, test_loader, DEVICE)
    print(f"量化后（已校准）模型的测试集准确率: {quant_accuracy:.2f}%")




if __name__ == "__main__":
    main()
