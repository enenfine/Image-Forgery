import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils import seed_everything
from src.metrics import dice_coeff, evaluate_model

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        
        # 损失函数（BCE+Dice）
        self.criterion = nn.BCELoss()
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay']
        )
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['epochs']
        )
        
        # 训练记录
        self.best_f1 = 0.0
        self.save_dir = cfg['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self, epoch):
        """单轮训练"""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']}")
        
        for batch in pbar:
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            preds = self.model(imgs)
            loss = self.criterion(preds, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            avg_loss = total_loss / (len(pbar) * self.cfg['batch_size'])
            pbar.set_postfix({'loss': avg_loss, 'lr': self.optimizer.param_groups[0]['lr']})
        
        return avg_loss

    def train(self):
        """完整训练流程"""
        seed_everything(42)
        for epoch in range(self.cfg['epochs']):
            # 训练
            train_loss = self.train_one_epoch(epoch)
            # 验证
            val_dice, val_f1 = evaluate_model(self.model, self.val_loader, self.device)
            
            # 学习率更新
            self.scheduler.step()
            
            # 日志打印
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f} | Val F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1
                }, os.path.join(self.save_dir, 'best_model.pth'))
                print(f"Best model saved (F1: {self.best_f1:.4f})")
        
        # 保存最终模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, 'final_model.pth')
        )
        print(f"Training finished. Best F1: {self.best_f1:.4f}")