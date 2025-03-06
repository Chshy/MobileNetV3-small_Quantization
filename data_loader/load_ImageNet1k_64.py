from datasets import load_dataset as load_dataset_hf

from torch.utils.data import DataLoader
from data_loader.base_dataset import BaseDataset
import torchvision.transforms as transforms

'''
https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 1281167
    })
    validation: Dataset({
        features: ['image', 'label'],
        num_rows: 50000
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 100000
    })
})
'''

def get_default_transforms():
    """此数据集的默认预处理"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class ImageNetDataset(BaseDataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__(transform)
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']      # 原始字段名 'image'
        label = item['label']      # 原始字段名 'label'
        image = self.apply_transform(image)
        return image, label

def load_dataset(data_dir="./data", train_transform=None, val_transform=None, test_transform=None):
    """
    返回 (train_dataset, val_dataset, test_dataset)
    """
    # 加载原始数据
    dataset = load_dataset_hf("benjamin-paine/imagenet-1k-64x64", cache_dir=data_dir)
    
    # 设置默认预处理（允许覆盖）
    default_train_tf, default_val_tf = get_default_transforms()
    train_transform = train_transform if train_transform else default_train_tf
    val_transform = val_transform if val_transform else default_val_tf
    test_transform = test_transform if test_transform else default_val_tf  # 测试集使用验证集的预处理
    
    # 创建 Dataset 对象
    train_dataset = ImageNetDataset(dataset['train'], train_transform)
    val_dataset = ImageNetDataset(dataset['validation'], val_transform)
    test_dataset = ImageNetDataset(dataset['test'], test_transform)
    
    return train_dataset, val_dataset, test_dataset

# 示例用法（用户测试用）
if __name__ == "__main__":
    train_set, val_set, test_set = load_dataset()
    print(f"训练集样本数: {len(train_set)}, 验证集样本数: {len(val_set)}, 测试集样本数: {len(test_set)}")