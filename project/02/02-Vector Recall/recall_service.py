"""
召回服务 - 在线推荐接口
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass

from models.two_tower import TwoTowerModel
from index.hnsw_index import HNSWIndex


@dataclass
class RecallResult:
    """召回结果"""
    item_ids: List[int]
    scores: List[float]
    latency_ms: float
    metadata: Optional[List[dict]] = None


class VectorRecallService:
    """
    矢量召回服务
    
    提供高效的用户-物品匹配服务
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        index: HNSWIndex,
        device: str = 'cuda'
    ):
        self.model = model
        self.index = index
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_files(
        cls,
        model_path: str,
        index_path: str,
        model_config,
        device: str = 'cuda'
    ) -> 'VectorRecallService':
        """
        从文件加载服务
        """
        # 加载模型
        model = TwoTowerModel(model_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载索引
        index = HNSWIndex.load(index_path)
        
        return cls(model, index, device)
    
    @torch.no_grad()
    def encode_user(
        self,
        user_id: int,
        user_features: np.ndarray,
        user_behavior: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        编码单个用户
        """
        user_data = {
            'user_id': torch.tensor([user_id], dtype=torch.long).to(self.device),
            'user_features': torch.tensor([user_features], dtype=torch.float32).to(self.device)
        }
        
        if user_behavior is not None:
            user_data['user_behavior'] = torch.tensor(
                [user_behavior], dtype=torch.float32
            ).to(self.device)
        
        user_emb = self.model.encode_user(user_data)
        return user_emb.cpu().numpy()[0]
    
    @torch.no_grad()
    def encode_users_batch(
        self,
        user_ids: np.ndarray,
        user_features: np.ndarray
    ) -> np.ndarray:
        """
        批量编码用户
        """
        user_data = {
            'user_id': torch.tensor(user_ids, dtype=torch.long).to(self.device),
            'user_features': torch.tensor(user_features, dtype=torch.float32).to(self.device)
        }
        
        user_emb = self.model.encode_user(user_data)
        return user_emb.cpu().numpy()
    
    def recall(
        self,
        user_id: int,
        user_features: np.ndarray,
        top_k: int = 500,
        user_behavior: Optional[np.ndarray] = None,
        ef_search: Optional[int] = None
    ) -> RecallResult:
        """
        为单个用户召回候选物品
        
        Args:
            user_id: 用户ID
            user_features: 用户特征向量
            top_k: 返回的候选数量
            user_behavior: 用户行为序列embedding（可选）
            ef_search: 搜索宽度（可选）
            
        Returns:
            RecallResult对象
        """
        start_time = time.time()
        
        # 1. 编码用户
        user_emb = self.encode_user(user_id, user_features, user_behavior)
        
        # 2. HNSW搜索
        item_ids, distances, metadata = self.index.search_single(
            user_emb, k=top_k
        )
        
        # 3. 转换距离为相似度分数
        # 对于cosine距离，距离越小相似度越高
        if self.index.space == 'cosine':
            scores = 1 - distances  # cosine similarity
        elif self.index.space == 'ip':
            scores = distances  # inner product
        else:
            scores = -distances  # L2 distance
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecallResult(
            item_ids=item_ids.tolist(),
            scores=scores.tolist(),
            latency_ms=latency_ms,
            metadata=metadata
        )
    
    def batch_recall(
        self,
        user_ids: np.ndarray,
        user_features: np.ndarray,
        top_k: int = 500
    ) -> List[RecallResult]:
        """
        批量用户召回
        """
        start_time = time.time()
        
        # 批量编码用户
        user_embs = self.encode_users_batch(user_ids, user_features)
        
        # 批量搜索
        item_ids, distances = self.index.search(user_embs, k=top_k)
        
        # 转换结果
        if self.index.space == 'cosine':
            scores = 1 - distances
        elif self.index.space == 'ip':
            scores = distances
        else:
            scores = -distances
        
        total_latency = (time.time() - start_time) * 1000
        avg_latency = total_latency / len(user_ids)
        
        results = []
        for i in range(len(user_ids)):
            results.append(RecallResult(
                item_ids=item_ids[i].tolist(),
                scores=scores[i].tolist(),
                latency_ms=avg_latency
            ))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            'index_stats': self.index.get_stats(),
            'device': str(self.device),
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
