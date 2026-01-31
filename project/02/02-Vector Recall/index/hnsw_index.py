"""
HNSW索引 - 高效的近似最近邻搜索
"""
import numpy as np
import hnswlib
from typing import Tuple, List, Optional
import pickle
import os
from pathlib import Path


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) 索引
    
    用于高效的向量相似度搜索，支持百万级别向量的毫秒级检索
    """
    
    def __init__(
        self,
        dim: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        space: str = 'cosine',
        num_threads: int = -1
    ):
        """
        Args:
            dim: 向量维度
            max_elements: 最大元素数量
            M: 每个节点的最大连接数（越大精度越高，但内存和构建时间增加）
            ef_construction: 构建时的搜索宽度（越大精度越高，构建越慢）
            ef_search: 查询时的搜索宽度（越大精度越高，查询越慢）
            space: 距离度量，'cosine'、'l2' 或 'ip'（内积）
            num_threads: 线程数，-1表示使用所有可用核心
        """
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        self.num_threads = num_threads if num_threads > 0 else os.cpu_count()
        
        # 初始化HNSW索引
        self.index = hnswlib.Index(space=space, dim=dim)
        
        # ID到元数据的映射
        self.id_to_metadata = {}
        
        # 是否已初始化
        self._initialized = False
    
    def init_index(self):
        """初始化索引（首次添加元素前调用）"""
        if not self._initialized:
            self.index.init_index(
                max_elements=self.max_elements,
                M=self.M,
                ef_construction=self.ef_construction,
                random_seed=42
            )
            self.index.set_ef(self.ef_search)
            self.index.set_num_threads(self.num_threads)
            self._initialized = True
    
    def add_items(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        metadata: Optional[List[dict]] = None
    ):
        """
        批量添加向量到索引
        
        Args:
            vectors: 向量矩阵 [num_items, dim]
            ids: 向量ID [num_items]
            metadata: 元数据列表（可选）
        """
        if not self._initialized:
            self.init_index()
        
        # 确保向量是float32
        vectors = vectors.astype(np.float32)
        ids = ids.astype(np.int64)
        
        # L2归一化（对于cosine距离）
        if self.space == 'cosine':
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # 添加到索引
        self.index.add_items(vectors, ids, num_threads=self.num_threads)
        
        # 保存元数据
        if metadata is not None:
            for i, item_id in enumerate(ids):
                self.id_to_metadata[int(item_id)] = metadata[i]
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 100,
        ef_search: Optional[int] = None,
        filter_fn: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量搜索最近邻
        
        Args:
            query_vectors: 查询向量 [num_queries, dim]
            k: 返回的最近邻数量
            ef_search: 搜索宽度（可选，覆盖默认值）
            filter_fn: 过滤函数（可选）
            
        Returns:
            ids: 最近邻ID [num_queries, k]
            distances: 距离 [num_queries, k]
        """
        query_vectors = query_vectors.astype(np.float32)
        
        # L2归一化
        if self.space == 'cosine':
            query_vectors = query_vectors / (
                np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8
            )
        
        # 设置搜索宽度
        if ef_search is not None:
            self.index.set_ef(ef_search)
        else:
            self.index.set_ef(self.ef_search)
        
        # 搜索
        if filter_fn is not None:
            # 带过滤的搜索
            ids, distances = self.index.knn_query(
                query_vectors, 
                k=k,
                filter=filter_fn,
                num_threads=self.num_threads
            )
        else:
            ids, distances = self.index.knn_query(
                query_vectors, 
                k=k,
                num_threads=self.num_threads
            )
        
        return ids, distances
    
    def search_single(
        self,
        query_vector: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        单个查询搜索，返回带元数据的结果
        
        Args:
            query_vector: 查询向量 [dim]
            k: 返回数量
            
        Returns:
            ids, distances, metadata_list
        """
        ids, distances = self.search(query_vector.reshape(1, -1), k=k)
        ids = ids[0]
        distances = distances[0]
        
        # 获取元数据
        metadata_list = [
            self.id_to_metadata.get(int(item_id), {})
            for item_id in ids
        ]
        
        return ids, distances, metadata_list
    
    def get_current_count(self) -> int:
        """获取当前索引中的元素数量"""
        return self.index.get_current_count() if self._initialized else 0
    
    def save(self, path: str):
        """
        保存索引到文件
        
        Args:
            path: 保存路径（不含扩展名）
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存HNSW索引
        self.index.save_index(f"{path}.bin")
        
        # 保存元数据和配置
        meta = {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'space': self.space,
            'id_to_metadata': self.id_to_metadata
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(meta, f)
    
    @classmethod
    def load(cls, path: str, num_threads: int = -1) -> 'HNSWIndex':
        """
        从文件加载索引
        
        Args:
            path: 索引路径（不含扩展名）
            num_threads: 线程数
            
        Returns:
            HNSWIndex实例
        """
        # 加载元数据
        with open(f"{path}.meta", 'rb') as f:
            meta = pickle.load(f)
        
        # 创建索引实例
        index = cls(
            dim=meta['dim'],
            max_elements=meta['max_elements'],
            M=meta['M'],
            ef_construction=meta['ef_construction'],
            ef_search=meta['ef_search'],
            space=meta['space'],
            num_threads=num_threads
        )
        
        # 加载HNSW索引
        index.index = hnswlib.Index(space=meta['space'], dim=meta['dim'])
        index.index.load_index(f"{path}.bin", max_elements=meta['max_elements'])
        index.index.set_ef(meta['ef_search'])
        index._initialized = True
        
        # 恢复元数据
        index.id_to_metadata = meta['id_to_metadata']
        
        return index
    
    def get_stats(self) -> dict:
        """获取索引统计信息"""
        return {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'current_count': self.get_current_count(),
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'space': self.space
        }
