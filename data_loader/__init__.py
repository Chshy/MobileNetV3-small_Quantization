import importlib

def get_dataset(dataset_name, **kwargs):
    """统一入口：根据名称加载数据集"""
    try:
        module = importlib.import_module(f'.load_{dataset_name}', package=__name__)
        return module.load_dataset(**kwargs)
    except ModuleNotFoundError:
        raise ValueError(f"未找到数据集 {dataset_name} 的加载脚本！")


'''

from data_loader import get_dataset

# 加载 ImageNet（自动调用 data_loader.load_imagenet.load_dataset()）
train_set, val_set, _ = get_dataset("imagenet", data_dir="./my_data")

# 创建 DataLoader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)



创建新脚本: 仿照 load_imagenet.py 创建 load_coco.py。
实现字段映射: 在 __getitem__ 中将原始数据字段转换为 (image, label)。
定义默认预处理: 在 get_default_transforms() 中配置数据增强。
注册到工厂: 无需额外步骤，命名符合 load_{name}.py 即可。

接口一致性: 所有脚本必须提供 load_dataset() 并返回相同格式。
命名规范: 脚本名称为 load_{dataset_name}.py，使用小写和下划线。
灵活配置: 允许用户覆盖 data_dir 和 transform 参数。
异常处理: 明确提示缺失字段或加载失败问题。

在Python中，文件路径的当前目录（即用./表示的目录）通常是相对于当前工作目录（CWD，Current Working Directory）的，而不是脚本所在的目录。

'''
