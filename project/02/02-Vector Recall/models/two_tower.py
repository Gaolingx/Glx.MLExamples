"""
双塔模型 - 结合用户塔和物品塔进行训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders import UserEncoder, ItemEncoder


class TwoTowerModel(nn.Module):
    """
    双塔召回模型
    
    使用对比学习训练用户和物品的向量表示，
    使得正样本对的相似度高于负样本对。
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 用户塔
        self.user_encoder = UserEncoder(
            num_users=config.num_users,
            user_embedding_dim=config.user_feature_dim,
            user_feature_dim=config.user_feature_dim,
            hidden_dims=config.user_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout
        )
        
        # 物品塔
        self.item_encoder = ItemEncoder(
            num_items=config.num_items,
            num_categories=config.num_categories,
            num_tags=config.num_tags,
            item_embedding_dim=config.item_feature_dim,
            item_feature_dim=config.item_feature_dim,
            hidden_dims=config.item_hidden_dims,
            output_dim=config.embedding_dim,
            dropout=config.dropout
        )
    
    def encode_user(self, user_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码用户"""
        return self.user_encoder(
            user_ids=user_data['user_id'],
            user_features=user_data['user_features'],
            user_behavior=user_data.get('user_behavior'),
            normalize=True
        )
    
    def encode_item(self, item_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码物品"""
        return self.item_encoder(
            item_ids=item_data['item_id'],
            category_ids=item_data['category_id'],
            tag_ids=item_data['tag_ids'],
            item_features=item_data['item_features'],
            normalize=True
        )
    
    def forward(
        self,
        user_data: Dict[str, torch.Tensor],
        pos_item_data: Dict[str, torch.Tensor],
        neg_item_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_data: 用户数据字典
            pos_item_data: 正样本物品数据
            neg_item_data: 负样本物品数据（可选）
            
        Returns:
            包含向量和相似度的字典
        """
        # 编码用户
        user_emb = self.encode_user(user_data)  # [B, D]
        
        # 编码正样本物品
        pos_item_emb = self.encode_item(pos_item_data)  # [B, D]
        
        # 计算正样本相似度
        pos_sim = torch.sum(user_emb * pos_item_emb, dim=-1)  # [B]
        
        output = {
            'user_emb': user_emb,
            'pos_item_emb': pos_item_emb,
            'pos_sim': pos_sim
        }
        
        # 如果有负样本
        if neg_item_data is not None:
            neg_item_emb = self.encode_item(neg_item_data)  # [B * num_neg, D]
            output['neg_item_emb'] = neg_item_emb
            
            # 计算负样本相似度
            batch_size = user_emb.size(0)
            num_neg = neg_item_emb.size(0) // batch_size
            
            neg_item_emb_reshaped = neg_item_emb.view(batch_size, num_neg, -1)  # [B, N, D]
            neg_sim = torch.bmm(
                neg_item_emb_reshaped, 
                user_emb.unsqueeze(-1)
            ).squeeze(-1)  # [B, N]
            
            output['neg_sim'] = neg_sim
        
        return output


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比学习损失
    
    最大化正样本对相似度，最小化负样本对相似度
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_emb: [batch_size, dim]
            pos_item_emb: [batch_size, dim]
            neg_item_emb: [batch_size * num_neg, dim] 或 [batch_size, num_neg, dim]
            
        Returns:
            标量损失
        """
        batch_size = user_emb.size(0)
        
        # 正样本相似度
        pos_sim = torch.sum(user_emb * pos_item_emb, dim=-1) / self.temperature  # [B]
        
        # 负样本相似度
        if neg_item_emb.dim() == 2:
            num_neg = neg_item_emb.size(0) // batch_size
            neg_item_emb = neg_item_emb.view(batch_size, num_neg, -1)
        
        neg_sim = torch.bmm(
            neg_item_emb, 
            user_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # [B, num_neg]
        
        # 拼接：正样本在第一列
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1 + num_neg]
        
        # 标签：正样本索引为0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class BPRLoss(nn.Module):
    """
    BPR (Bayesian Personalized Ranking) 损失
    
    使正样本得分高于负样本得分
    """
    
    def forward(
        self,
        pos_sim: torch.Tensor,
        neg_sim: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_sim: 正样本相似度 [batch_size]
            neg_sim: 负样本相似度 [batch_size, num_neg]
            
        Returns:
            标量损失
        """
        # 扩展pos_sim维度
        pos_sim = pos_sim.unsqueeze(-1)  # [B, 1]
        
        # BPR损失
        loss = -F.logsigmoid(pos_sim - neg_sim).mean()
        
        return loss
