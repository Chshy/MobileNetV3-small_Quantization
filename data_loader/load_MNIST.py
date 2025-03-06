from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from data_loader.base_dataset import BaseDataset

def get_default_transforms():
    """此数据集的默认预处理"""
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机放射变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST 的均值和标准差
    ])
    # 若对训练集使用随机放射变换进行数据增强，则训练集准确度可能会低于测试集/验证集准确度
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    return train_transform, val_transform

class MNISTDataset(BaseDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(transform)
        self.dataset = dataset  # 可以是原始数据集或 Subset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(self.dataset, datasets.MNIST):
            # 如果是原始数据集，直接获取数据
            image, label = self.dataset[idx]
        else:
            # 如果是 Subset，通过 indices 获取数据
            image, label = self.dataset.dataset[self.dataset.indices[idx]]
        image = self.apply_transform(image)
        return image, label

def load_dataset(data_dir="./data", train_transform=None, val_transform=None, test_transform=None, val_split_ratio=None):
    """
    返回 (train_dataset, val_dataset, test_dataset)
    
    参数:
    - val_split_ratio: 验证集划分比例，默认为 None（不划分）。
                       如果提供值（如 0.2），则将训练集按比例划分为训练集和验证集。
    """
    # 设置默认预处理（允许覆盖）
    default_train_tf, default_val_tf = get_default_transforms()
    train_transform = train_transform if train_transform else default_train_tf
    val_transform = val_transform if val_transform else default_val_tf
    test_transform = test_transform if test_transform else default_val_tf  # 测试集使用验证集的预处理

    # 加载原始数据
    train_val_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
    test_dataset_raw = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)

    # 划分训练集和验证集
    if val_split_ratio is not None:
        if not (0 < val_split_ratio < 1):
            raise ValueError("val_split_ratio 必须在 (0, 1) 范围内")
        train_size = int((1 - val_split_ratio) * len(train_val_dataset))  # 训练集大小
        val_size = len(train_val_dataset) - train_size  # 验证集大小
        train_dataset_raw, val_dataset_raw = random_split(train_val_dataset, [train_size, val_size]) # 【注意 这里是随机划分的】

        # 创建 Dataset 对象
        train_dataset = MNISTDataset(train_dataset_raw, train_transform)
        val_dataset = MNISTDataset(val_dataset_raw, val_transform)
    else:
        # 不划分时，验证集为 None
        train_dataset = MNISTDataset(train_val_dataset, train_transform)
        val_dataset = None

    # 测试集
    test_dataset = MNISTDataset(test_dataset_raw, test_transform)

    return train_dataset, val_dataset, test_dataset

# 示例用法（用户测试用）
if __name__ == "__main__":
    # 默认不划分验证集
    train_set, val_set, test_set = load_dataset()
    print(f"训练集样本数: {len(train_set)}, 验证集样本数: {len(val_set) if val_set else 'None'}, 测试集样本数: {len(test_set)}")

    # 按 80% 训练集、20% 验证集划分
    train_set, val_set, test_set = load_dataset(val_split_ratio=0.2)
    print(f"训练集样本数: {len(train_set)}, 验证集样本数: {len(val_set)}, 测试集样本数: {len(test_set)}")