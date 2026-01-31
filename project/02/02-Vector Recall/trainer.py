"""
训练器 - 模型训练流程
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from typing import Dict, Optional

from models.two_tower import TwoTowerModel, InfoNCELoss, BPRLoss
from index.hnsw_index import HNSWIndex


class Trainer:
    """
    双塔模型训练器
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = InfoNCELoss(temperature=config.temperature)
        
        # 检查点目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # 移动数据到设备
            user_data = {k: v.to(self.device) for k, v in batch['user'].items()}
            pos_item_data = {k: v.to(self.device) for k, v in batch['pos_item'].items()}
            neg_item_data = {k: v.to(self.device) for k, v in batch['neg_items'].items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            
            output = self.model(user_data, pos_item_data, neg_item_data)
            
            # 计算损失
            loss = self.criterion(
                output['user_emb'],
                output['pos_item_emb'],
                output['neg_item_emb']
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """评估模型"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            user_data = {k: v.to(self.device) for k, v in batch['user'].items()}
            pos_item_data = {k: v.to(self.device) for k, v in batch['pos_item'].items()}
            neg_item_data = {k: v.to(self.device) for k, v in batch['neg_items'].items()}
            
            output = self.model(user_data, pos_item_data, neg_item_data)
            
            loss = self.criterion(
                output['user_emb'],
                output['pos_item_emb'],
                output['neg_item_emb']
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {self.config.num_epochs}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # 评估
            val_loss = self.evaluate()
            self.history['val_loss'].append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"lr={self.scheduler.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


class IndexBuilder:
    """
    索引构建器 - 使用训练好的模型构建HNSW索引
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        hnsw_config,
        device: str = 'cuda'
    ):
        self.model = model
        self.hnsw_config = hnsw_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def build_index(
        self,
        item_ids: np.ndarray,
        item_categories: np.ndarray,
        item_tags: np.ndarray,
        item_features: np.ndarray,
        batch_size: int = 1024
    ) -> HNSWIndex:
        """
        构建物品索引
        
        Args:
            item_ids: 物品ID数组
            item_categories: 物品类别
            item_tags: 物品标签
            item_features: 物品特征
            batch_size: 批处理大小
            
        Returns:
            HNSWIndex实例
        """
        print("Building HNSW index...")
        start_time = time.time()
        
        # 创建索引
        index = HNSWIndex(
            dim=self.hnsw_config.dim,
            max_elements=self.hnsw_config.max_elements,
            M=self.hnsw_config.M,
            ef_construction=self.hnsw_config.ef_construction,
            ef_search=self.hnsw_config.ef_search,
            space=self.hnsw_config.space
        )
        index.init_index()
        
        # 批量编码物品
        num_items = len(item_ids)
        all_embeddings = []
        
        for i in tqdm(range(0, num_items, batch_size), desc="Encoding items"):
            end_idx = min(i + batch_size, num_items)
            
            item_data = {
                'item_id': torch.tensor(item_ids[i:end_idx], dtype=torch.long).to(self.device),
                'category_id': torch.tensor(item_categories[i:end_idx], dtype=torch.long).to(self.device),
                'tag_ids': torch.tensor(item_tags[i:end_idx], dtype=torch.long).to(self.device),
                'item_features': torch.tensor(item_features[i:end_idx], dtype=torch.float32).to(self.device)
            }
            
            embeddings = self.model.encode_item(item_data)
            all_embeddings.append(embeddings.cpu().numpy())
        
        # 合并所有embedding
        all_embeddings = np.vstack(all_embeddings)
        
        # 添加到索引
        index.add_items(
            vectors=all_embeddings,
            ids=item_ids,
            metadata=[{'item_id': int(iid)} for iid in item_ids]
        )
        
        elapsed = time.time() - start_time
        print(f"Index built in {elapsed:.2f}s, total items: {index.get_current_count()}")
        
        return index
