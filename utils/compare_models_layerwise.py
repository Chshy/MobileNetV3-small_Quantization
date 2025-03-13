import torch
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