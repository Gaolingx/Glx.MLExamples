"""
数据集 - 训练和评估数据处理
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class RecommendationDataset(Dataset):
    """
    推荐数据集
    
    每个样本包含：用户、正样本物品、负样本物品
    """
    
    def __init__(
        self,
        user_item_pairs: List[Tuple[int, int]],  # (user_id, item_id) 正样本对
        user_features: np.ndarray,  # [num_users, feature_dim]
        item_features: np.ndarray,  # [num_items, feature_dim]
        item_categories: np.ndarray,  # [num_items]
        item_tags: np.ndarray,  # [num_items, max_tags]
        num_items: int,
        num_negatives: int = 10,
        user_history: Optional[Dict[int, set]] = None  # 用于避免采样用户已交互物品
    ):
        self.user_item_pairs = user_item_pairs
        self.user_features = user_features
        self.item_features = item_features
        self.item_categories = item_categories
        self.item_tags = item_tags
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_history = user_history or {}
    
    def __len__(self):
        return len(self.user_item_pairs)
    
    def _sample_negatives(self, user_id: int, pos_item: int) -> List[int]:
        """采样负样本"""
        history = self.user_history.get(user_id, set())
        history = history | {pos_item}
        
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.num_items - 1)
            if neg not in history:
                negatives.append(neg)
        
        return negatives
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, pos_item_id = self.user_item_pairs[idx]
        neg_item_ids = self._sample_negatives(user_id, pos_item_id)
        
        # 用户数据
        user_data = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'user_features': torch.tensor(self.user_features[user_id], dtype=torch.float32)
        }
        
        # 正样本物品数据
        pos_item_data = {
            'item_id': torch.tensor(pos_item_id, dtype=torch.long),
            'category_id': torch.tensor(self.item_categories[pos_item_id], dtype=torch.long),
            'tag_ids': torch.tensor(self.item_tags[pos_item_id], dtype=torch.long),
            'item_features': torch.tensor(self.item_features[pos_item_id], dtype=torch.float32)
        }
        
        # 负样本物品数据
        neg_item_data = {
            'item_id': torch.tensor(neg_item_ids, dtype=torch.long),
            'category_id': torch.tensor(self.item_categories[neg_item_ids], dtype=torch.long),
            'tag_ids': torch.tensor(self.item_tags[neg_item_ids], dtype=torch.long),
            'item_features': torch.tensor(self.item_features[neg_item_ids], dtype=torch.float32)
        }
        
        return {
            'user': user_data,
            'pos_item': pos_item_data,
            'neg_items': neg_item_data
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    自定义collate函数
    """
    user_data = {
        'user_id': torch.stack([b['user']['user_id'] for b in batch]),
        'user_features': torch.stack([b['user']['user_features'] for b in batch])
    }
    
    pos_item_data = {
        'item_id': torch.stack([b['pos_item']['item_id'] for b in batch]),
        'category_id': torch.stack([b['pos_item']['category_id'] for b in batch]),
        'tag_ids': torch.stack([b['pos_item']['tag_ids'] for b in batch]),
        'item_features': torch.stack([b['pos_item']['item_features'] for b in batch])
    }
    
    # 负样本展平
    neg_item_ids = torch.cat([b['neg_items']['item_id'] for b in batch])
    neg_category_ids = torch.cat([b['neg_items']['category_id'] for b in batch])
    neg_tag_ids = torch.cat([b['neg_items']['tag_ids'] for b in batch])
    neg_item_features = torch.cat([b['neg_items']['item_features'] for b in batch])
    
    neg_item_data = {
        'item_id': neg_item_ids,
        'category_id': neg_category_ids,
        'tag_ids': neg_tag_ids,
        'item_features': neg_item_features
    }
    
    return {
        'user': user_data,
        'pos_item': pos_item_data,
        'neg_items': neg_item_data
    }


def generate_synthetic_data(
    num_users: int = 10000,
    num_items: int = 100000,
    num_categories: int = 100,
    num_tags: int = 500,
    max_tags_per_item: int = 10,
    user_feature_dim: int = 64,
    item_feature_dim: int = 64,
    num_interactions: int = 500000
) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    生成合成数据用于测试
    """
    print("Generating synthetic data...")
    
    # 用户特征
    user_features = np.random.randn(num_users, user_feature_dim).astype(np.float32)
    
    # 物品特征
    item_features = np.random.randn(num_items, item_feature_dim).astype(np.float32)
    
    # 物品类别
    item_categories = np.random.randint(0, num_categories, size=num_items)
    
    # 物品标签（每个物品有1-10个标签）
    item_tags = np.zeros((num_items, max_tags_per_item), dtype=np.int64)
    for i in range(num_items):
        n_tags = np.random.randint(1, max_tags_per_item + 1)
        item_tags[i, :n_tags] = np.random.choice(num_tags, size=n_tags, replace=False) + 1  # 0是padding
    
    # 生成用户-物品交互
    user_item_pairs = []
    user_history = {i: set() for i in range(num_users)}
    
    for _ in range(num_interactions):
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        
        user_item_pairs.append((user_id, item_id))
        user_history[user_id].add(item_id)
    
    print(f"Generated {len(user_item_pairs)} interactions")
    
    return (
        user_item_pairs,
        user_features,
        item_features,
        item_categories,
        item_tags,
        user_history
    )
