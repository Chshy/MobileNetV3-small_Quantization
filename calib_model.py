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

from tqdm import tqdm
def evaluate_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


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
    quant_model.print_quant_params()

    # 测试未校准精度
    quant_model.set_quant_state(quant_on = True, calib_on = False)
    print("开始测试量化后（未校准）的模型精度...")
    quant_accuracy = evaluate_model(quant_model, test_loader, DEVICE)
    print(f"量化后（未校准）模型的测试集准确率: {quant_accuracy:.2f}%")
    quant_model.print_quant_params()

    # 校准模型,并测试校准后的精度
    quant_model.set_quant_state(quant_on = False, calib_on = True)
    quant_model.calibrate(train_loader, DEVICE, num_batches=782, verbose=True)
    quant_model.set_quant_state(quant_on = True, calib_on = False)

    print("开始测试量化后（已校准）的模型精度...")
    quant_accuracy = evaluate_model(quant_model, test_loader, DEVICE)
    print(f"量化后（已校准）模型的测试集准确率: {quant_accuracy:.2f}%")
    quant_model.print_quant_params()




if __name__ == "__main__":
    main()
