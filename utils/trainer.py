import os
import sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.insert(0, project_root)

from utils.training_visualizer import plot_training_curves

import torch
# import os
import time
import json
import csv
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epochs,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        save_dir=None,
        save_best=False,
        experiment_name=None
    ):
        """
        初始化训练器
        :param model: 要训练的模型
        :param train_loader: 训练数据加载器
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param device: 训练设备 (cpu/cuda)
        :param epochs: 训练总轮次
        :param val_loader: 验证数据加载器 (可选)
        :param test_loader: 测试数据加载器 (可选)
        :param scheduler: 学习率调度器 (可选)
        :param save_dir: 结果保存根目录 (默认: runs)
        :param save_best: 是否保存最佳模型 (根据验证准确率)
        :param experiment_name: 实验名称 (默认: 时间戳)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.save_best = save_best
        self.best_val_acc = 0.0

        # 创建保存目录
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if experiment_name is None:
            experiment_name = timestamp
        else:
            experiment_name = f"{timestamp}_{experiment_name}"
        
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "runs")
        
        self.save_path = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_path, exist_ok=True)

        # 初始化日志和保存超参数
        self._init_logger()
        self._save_hyperparameters()

    def _init_logger(self):
        """初始化训练日志文件"""
        self.log_file = os.path.join(self.save_path, "training_log.csv")
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "train_acc", "val_acc", "lr"
            ])

    def _save_hyperparameters(self):
        """保存训练超参数"""
        hyperparams = {
            "model": type(self.model).__name__,
            "batch_size": self.train_loader.batch_size,
            "epochs": self.epochs,
            "optimizer": type(self.optimizer).__name__,
            "lr": self.optimizer.param_groups[0]["lr"],
            "criterion": type(self.criterion).__name__,
            "device": str(self.device),
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None
        }

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                hyperparams.update({
                    "scheduler_step_size": self.scheduler.step_size,
                    "scheduler_gamma": self.scheduler.gamma
                })

        with open(os.path.join(self.save_path, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=4)

    def _log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """记录单个epoch的结果"""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}" if val_loss is not None else "N/A",
                f"{train_acc:.2f}%",
                f"{val_acc:.2f}%" if val_acc is not None else "N/A",
                f"{lr:.6f}"
            ])

    def _train_epoch(self, epoch):
        """单个epoch的训练逻辑"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}/{self.epochs}",
            # leave=False
            leave=True
        )
        
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{total_loss/(batch_idx+1):.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })
        
        return total_loss/len(self.train_loader), 100.*correct/total

    def _validate(self, data_loader):
        """验证/测试逻辑"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Validating",
            leave=True
        )
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_total = targets.size(0)
                total += batch_total
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条显示
                avg_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{current_acc:.2f}%"
                })
        
        return total_loss/len(data_loader), 100.*correct/total

    def train(self):
        """完整的训练流程"""
        for epoch in range(self.epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 验证阶段
            val_loss, val_acc = None, None
            if self.val_loader:
                val_loss, val_acc = self._validate(self.val_loader)
                
                # 保存最佳模型
                if self.save_best and val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_path, "best_model.pth")
                    )
            
            # 学习率调整
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_loss:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录日志
            self._log_epoch(epoch+1, train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # 打印信息
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            if val_loss:
                print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")

        # 保存最终模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_path, "final_model.pth")
        )
        print(f"\nTraining complete. Models saved to {self.save_path}")

        plot_training_curves(
            csv_path=self.log_file,
            save_path=os.path.join(self.save_path, "training_log.png"),
            figsize=(16, 8),
            show_plot=True
        )

    def evaluate(self, data_loader=None):
        """评估模型性能"""
        if data_loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided")
            data_loader = self.test_loader
        
        loss, acc = self._validate(data_loader)
        print(f"\nEvaluation Results:")
        print(f"Loss: {loss:.4f} | Accuracy: {acc:.2f}%")
        return loss, acc