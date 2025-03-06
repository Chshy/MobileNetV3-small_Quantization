import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_training_curves(
    csv_path: str,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
    show_plot: bool = True,
    verbose: bool = True
) -> None:
    """
    绘制训练过程中的损失和准确率曲线，并保存为图片
    
    Args:
        csv_path: 训练日志CSV文件路径
        save_path: 图片保存路径（默认与CSV同路径，文件名自动替换为.png）
        figsize: 图像尺寸（宽, 高），默认(14,6)
        show_plot: 是否显示图形（默认True）
        verbose: 是否输出调试信息（默认True）
    """
    
    # 验证文件存在性
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件 {csv_path} 不存在")
        
    # 读取数据
    data = pd.read_csv(csv_path)
    
    # 验证必要列存在
    required_columns = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"CSV文件缺少必要列: {col}")
            
    if verbose:
        print("数据前5行预览:")
        print(data.head())

    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(data['epoch'], data['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 处理保存路径
    if save_path is None:
        save_path = os.path.splitext(csv_path)[0] + '.png'
    
    # 保存图片
    plt.savefig(save_path)
    
    if verbose:
        print(f"图片已保存至: {save_path}")
    
    # 控制是否显示图形
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # 命令行调用示例
    import argparse
    parser = argparse.ArgumentParser(description='训练曲线可视化工具')
    parser.add_argument('csv_path', help='训练日志CSV文件路径')
    parser.add_argument('--no-show', action='store_false', dest='show_plot',
                       help='不显示图形窗口')
    args = parser.parse_args()
    
    plot_training_curves(
        csv_path=args.csv_path,
        show_plot=args.show_plot
    )



# # 作为模块导入使用
# from training_visualizer import plot_training_curves

# plot_training_curves(
#     csv_path='./training_log.csv',
#     save_path='./custom_name.png',
#     figsize=(16, 8),
#     show_plot=False
# )

# # 或者通过命令行直接运行
# python training_visualizer.py ./training_log.csv --no-show