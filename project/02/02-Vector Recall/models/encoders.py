"""
用户和物品编码器 - 将特征映射到统一的向量空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLPBlock(nn.Module):
    """MLP基础块：Linear -> BatchNorm -> ReLU -> Dropout"""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout: float = 0.1,
        use_bn: bool = True
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UserEncoder(nn.Module):
    """
    用户编码器（User Tower）
    
    将用户的各种特征编码为固定维度的向量表示
    """
    
    def __init__(
        self,
        num_users: int,
        user_embedding_dim: int = 64,
        user_feature_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 用户ID Embedding
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # 用户静态特征编码（年龄、性别等）
        self.user_feature_encoder = nn.Linear(user_feature_dim, user_embedding_dim)
        
        # 用户行为序列编码（可选，这里用简化版本）
        self.behavior_encoder = nn.Sequential(
            nn.Linear(user_embedding_dim, user_embedding_dim),
            nn.ReLU()
        )
        
        # 特征融合层
        total_input_dim = user_embedding_dim * 3  # ID + 特征 + 行为
        
        # MLP层
        layers = []
        in_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        
        # 输出层（不使用激活函数，方便做归一化）
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        user_behavior: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            user_ids: 用户ID [batch_size]
            user_features: 用户特征 [batch_size, user_feature_dim]
            user_behavior: 用户行为序列的聚合表示 [batch_size, embedding_dim]
            normalize: 是否L2归一化输出
            
        Returns:
            用户向量表示 [batch_size, output_dim]
        """
        # 用户ID Embedding
        user_emb = self.user_embedding(user_ids)  # [B, E]
        
        # 用户特征编码
        feature_emb = self.user_feature_encoder(user_features)  # [B, E]
        
        # 行为序列编码
        if user_behavior is None:
            behavior_emb = torch.zeros_like(user_emb)
        else:
            behavior_emb = self.behavior_encoder(user_behavior)  # [B, E]
        
        # 特征拼接
        combined = torch.cat([user_emb, feature_emb, behavior_emb], dim=-1)
        
        # MLP
        output = self.mlp(combined)
        
        # L2归一化
        if normalize:
            output = F.normalize(output, p=2, dim=-1)
        
        return output


class ItemEncoder(nn.Module):
    """
    物品编码器（Item Tower）
    
    将物品的各种特征编码为固定维度的向量表示
    """
    
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        num_tags: int,
        item_embedding_dim: int = 64,
        item_feature_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 物品ID Embedding
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        
        # 类别Embedding
        self.category_embedding = nn.Embedding(num_categories, item_embedding_dim // 2)
        
        # 标签Embedding（使用平均池化）
        self.tag_embedding = nn.Embedding(num_tags, item_embedding_dim // 2, padding_idx=0)
        
        # 物品静态特征编码（价格、热度等）
        self.item_feature_encoder = nn.Linear(item_feature_dim, item_embedding_dim)
        
        # 特征融合
        total_input_dim = item_embedding_dim * 2 + item_embedding_dim  # ID + cat/tag + 特征
        
        # MLP层
        layers = []
        in_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        item_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            item_ids: 物品ID [batch_size]
            category_ids: 类别ID [batch_size]
            tag_ids: 标签ID [batch_size, max_tags]
            item_features: 物品特征 [batch_size, item_feature_dim]
            normalize: 是否L2归一化
            
        Returns:
            物品向量表示 [batch_size, output_dim]
        """
        # 物品ID Embedding
        item_emb = self.item_embedding(item_ids)  # [B, E]
        
        # 类别Embedding
        cat_emb = self.category_embedding(category_ids)  # [B, E/2]
        
        # 标签Embedding（平均池化）
        tag_emb = self.tag_embedding(tag_ids)  # [B, max_tags, E/2]
        tag_mask = (tag_ids != 0).float().unsqueeze(-1)  # [B, max_tags, 1]
        tag_emb = (tag_emb * tag_mask).sum(dim=1) / (tag_mask.sum(dim=1) + 1e-8)  # [B, E/2]
        
        # 物品特征编码
        feature_emb = self.item_feature_encoder(item_features)  # [B, E]
        
        # 拼接
        combined = torch.cat([item_emb, cat_emb, tag_emb, feature_emb], dim=-1)
        
        # MLP
        output = self.mlp(combined)
        
        if normalize:
            output = F.normalize(output, p=2, dim=-1)
        
        return output
