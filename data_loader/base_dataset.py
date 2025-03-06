from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    """所有数据集类的基类，实现接口一致性"""
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method.")

    def apply_transform(self, image):
        """应用数据增强"""
        if self.transform:
            return self.transform(image)
        return image
