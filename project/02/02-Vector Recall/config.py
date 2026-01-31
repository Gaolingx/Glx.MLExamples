"""
配置文件 - 定义系统超参数
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """模型配置"""
    # Embedding维度
    embedding_dim: int = 128
    
    # 用户特征配置
    num_users: int = 100000
    user_feature_dim: int = 64
    user_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # 物品特征配置
    num_items: int = 1000000
    item_feature_dim: int = 64
    item_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # 类别特征
    num_categories: int = 1000
    num_tags: int = 5000
    max_tags_per_item: int = 10
    
    # Dropout
    dropout: float = 0.1


@dataclass
class HNSWConfig:
    """HNSW索引配置"""
    # 向量维度
    dim: int = 128
    
    # HNSW参数
    M: int = 16                    # 每个节点的最大连接数
    ef_construction: int = 200     # 构建时的搜索宽度
    ef_search: int = 100           # 查询时的搜索宽度
    
    # 距离度量: 'cosine', 'l2', 'ip'(内积)
    space: str = 'cosine'
    
    # 索引容量
    max_elements: int = 1000000


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 10
    
    # 负采样数量
    num_negatives: int = 10
    
    # 温度参数（用于对比学习）
    temperature: float = 0.07
    
    # 设备
    device: str = 'cuda'
    
    # 检查点
    checkpoint_dir: str = './checkpoints'
    save_every: int = 2


@dataclass
class RecallConfig:
    """召回服务配置"""
    # 召回数量
    top_k: int = 500
    
    # 索引路径
    index_path: str = './index_files/hnsw.bin'
    model_path: str = './checkpoints/best_model.pt'
