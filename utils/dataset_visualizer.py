import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

from data_loader import get_dataset

class DatasetVisualizer:
    """
    通用数据集可视化工具类。
    """
    def __init__(self, dataset, image_key='image', label_key='label', label_mapping=None):
        """
        初始化可视化工具。
        
        参数:
            dataset: 数据集对象 (如 PyTorch Dataset 或 Hugging Face Dataset)。
            image_key: 数据集中存储图像的字段名，默认为 'image'。
            label_key: 数据集中存储标签的字段名，默认为 'label'。
            label_mapping: 标签映射字典，用于将数字标签转换为字符串标签。
        """
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.label_mapping = label_mapping

    def show_images_and_labels(self, num_images=5, random_select=False):
        """
        显示数据集中指定数量的图像及其标签。
        
        参数:
            num_images: 要显示的图像数量，默认为5。
            random_select: 是否随机选取图像，默认为False（从头开始选取）。
        """
        # 确定要显示的图像索引
        if random_select:
            indices = random.sample(range(len(self.dataset)), num_images)  # 随机选取索引
        else:
            indices = range(num_images)  # 从头开始选取索引

        # 创建子图
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

        for i, idx in enumerate(indices):
            # 获取图像和标签
            item = self.dataset[idx]
            image = item[self.image_key]  # 获取图像
            label = item[self.label_key]  # 获取标签

            # 如果有标签映射，则将数字标签转换为字符串标签
            if self.label_mapping:
                label = self.label_mapping.get(label, label)

            # 显示图像
            if num_images == 1:
                ax = axes  # 单张图像时，axes 不是数组
            else:
                ax = axes[i]

            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')  # 关闭坐标轴

        plt.tight_layout()
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 加载数据集
    train_set, val_set, test_set = get_dataset("ImageNet1k_64")

    # # 定义标签映射（如果需要）
    # label_mapping = {
    #     0: "cat",
    #     1: "dog",
    #     2: "bird",
    #     # 添加更多标签映射...
    # }

    # 创建可视化工具实例
    # visualizer = DatasetVisualizer(train_set, image_key='image', label_key='label', label_mapping=label_mapping)
    visualizer = DatasetVisualizer(train_set, image_key='image', label_key='label')

    # 显示前5张图像
    print("显示前5张图像：")
    visualizer.show_images_and_labels(num_images=5, random_select=False)

    # 随机显示5张图像
    print("随机显示5张图像：")
    visualizer.show_images_and_labels(num_images=5, random_select=True)