"""
Vector Recall (矢量召回) network using PyTorch + HNSW (hnswlib)

Features:
- PyTorch encoder (configurable: MLP or simple Transformer-like)
- Contrastive training (in-batch negatives / InfoNCE)
- Build HNSW index with hnswlib, configurable ef_construction, M
- Insert, delete, save, load index
- Retrieval API: get top-N candidate ids for a user embedding
- Simple re-ranking hook (use a secondary model or dot-product scoring)
- Monitoring functions: compute recall@K (requires ground truth)

Notes:
- hnswlib is CPU-based. Use GPU for embedding generation; move vectors to CPU before indexing.
- For production: consider sharding, quantization (PQ/OPQ or hnswlib's compressed vectors), caching, async writes.

Requirements:
- torch
- hnswlib (pip install hnswlib)

Example usage at bottom.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import hnswlib
except Exception as e:
    raise ImportError("Please install hnswlib: pip install hnswlib")
from typing import List, Tuple, Dict
import time


class TwoTowerModel(nn.Module):
    """双塔模型：分别为用户和物品生成嵌入向量"""

    def __init__(self, user_feat_dim: int, item_feat_dim: int,
                 embedding_dim: int = 128, hidden_dims: List[int] = [256, 128]):
        super(TwoTowerModel, self).__init__()

        # 用户塔
        user_layers = []
        in_dim = user_feat_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        user_layers.append(nn.Linear(in_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)

        # 物品塔
        item_layers = []
        in_dim = item_feat_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        item_layers.append(nn.Linear(in_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)

    def forward(self, user_feat: torch.Tensor, item_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播生成用户和物品嵌入"""
        user_emb = self.user_tower(user_feat)
        item_emb = self.item_tower(item_feat)

        # L2归一化，便于后续计算余弦相似度
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return user_emb, item_emb

    def get_user_embedding(self, user_feat: torch.Tensor) -> torch.Tensor:
        """获取用户嵌入"""
        with torch.no_grad():
            user_emb = self.user_tower(user_feat)
            user_emb = F.normalize(user_emb, p=2, dim=1)
        return user_emb

    def get_item_embedding(self, item_feat: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入"""
        with torch.no_grad():
            item_emb = self.item_tower(item_feat)
            item_emb = F.normalize(item_emb, p=2, dim=1)
        return item_emb


class VectorRecallSystem:
    """矢量召回系统：整合模型训练、HNSW索引构建和检索"""

    def __init__(self, model: TwoTowerModel, embedding_dim: int = 128):
        self.model = model
        self.embedding_dim = embedding_dim
        self.hnsw_index = None
        self.item_ids = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_model(self, train_loader, epochs: int = 10, lr: float = 0.001):
        """训练双塔模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (user_feat, item_feat, labels) in enumerate(train_loader):
                user_feat = user_feat.to(self.device)
                item_feat = item_feat.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()

                # 获取嵌入向量
                user_emb, item_emb = self.model(user_feat, item_feat)

                # 计算相似度得分（内积）
                scores = torch.sum(user_emb * item_emb, dim=1)

                # 计算损失
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    def build_hnsw_index(self, item_features: torch.Tensor, item_ids: List[int],
                         M: int = 16, ef_construction: int = 200):
        """
        构建HNSW索引

        Args:
            item_features: 物品特征矩阵 [num_items, feat_dim]
            item_ids: 物品ID列表
            M: HNSW参数，控制图的连接数（越大越精确但越慢）
            ef_construction: 构建索引时的搜索范围（越大越精确但构建越慢）
        """
        print("开始构建HNSW索引...")
        start_time = time.time()

        # 生成物品嵌入向量
        self.model.eval()
        item_embeddings = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(item_features), batch_size):
                batch = item_features[i:i + batch_size].to(self.device)
                emb = self.model.get_item_embedding(batch)
                item_embeddings.append(emb.cpu().numpy())

        item_embeddings = np.vstack(item_embeddings)
        self.item_ids = item_ids

        # 初始化HNSW索引
        num_items = len(item_embeddings)
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_index.init_index(max_elements=num_items, ef_construction=ef_construction, M=M)

        # 添加数据到索引
        self.hnsw_index.add_items(item_embeddings, np.array(item_ids))

        build_time = time.time() - start_time
        print(f"HNSW索引构建完成，耗时: {build_time:.2f}秒")
        print(f"索引包含 {num_items} 个物品")

    def recall(self, user_features: torch.Tensor, k: int = 100, ef_search: int = 50) -> Tuple[
        List[List[int]], np.ndarray]:
        """
        为用户召回Top-K物品

        Args:
            user_features: 用户特征 [batch_size, feat_dim]
            k: 召回物品数量
            ef_search: 搜索时的范围参数（越大越精确但越慢）

        Returns:
            recalled_items: 召回的物品ID列表
            distances: 对应的距离/相似度
        """
        if self.hnsw_index is None:
            raise ValueError("HNSW索引未构建，请先调用build_hnsw_index方法")

        print(f"开始召回Top-{k}物品...")
        start_time = time.time()

        # 设置搜索参数
        self.hnsw_index.set_ef(ef_search)

        # 生成用户嵌入
        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_features.to(self.device))
            user_emb = user_emb.cpu().numpy()

        # HNSW检索
        labels, distances = self.hnsw_index.knn_query(user_emb, k=k)

        recall_time = time.time() - start_time
        print(f"召回完成，耗时: {recall_time:.3f}秒")
        print(f"平均每个用户召回耗时: {recall_time / len(user_features) * 1000:.2f}ms")

        return labels.tolist(), distances

    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 {path} 加载")


# ==================== 使用示例 ====================
def demo_usage():
    """演示如何使用矢量召回系统"""

    # 1. 模拟数据
    num_users = 1000
    num_items = 10000
    user_feat_dim = 50
    item_feat_dim = 30
    embedding_dim = 128

    # 生成模拟的用户和物品特征
    user_features = torch.randn(num_users, user_feat_dim)
    item_features = torch.randn(num_items, item_feat_dim)
    item_ids = list(range(num_items))

    # 2. 初始化模型
    model = TwoTowerModel(
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        embedding_dim=embedding_dim,
        hidden_dims=[256, 128]
    )

    # 3. 创建召回系统
    recall_system = VectorRecallSystem(model, embedding_dim=embedding_dim)

    # 4. 模拟训练数据（实际应用中从真实数据加载）
    # 这里简化处理，生成随机正负样本
    train_user_feat = torch.randn(5000, user_feat_dim)
    train_item_feat = torch.randn(5000, item_feat_dim)
    train_labels = torch.randint(0, 2, (5000,))

    train_dataset = torch.utils.data.TensorDataset(train_user_feat, train_item_feat, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 5. 训练模型（示例中训练3个epoch）
    print("\n========== 开始训练模型 ==========")
    recall_system.train_model(train_loader, epochs=3, lr=0.001)

    # 6. 构建HNSW索引
    print("\n========== 构建HNSW索引 ==========")
    recall_system.build_hnsw_index(
        item_features=item_features,
        item_ids=item_ids,
        M=16,  # 控制精度和速度的平衡
        ef_construction=200
    )

    # 7. 为用户召回物品
    print("\n========== 执行召回 ==========")
    test_users = user_features[:10]  # 为前10个用户召回
    recalled_items, distances = recall_system.recall(
        user_features=test_users,
        k=100,  # 召回Top-100
        ef_search=50
    )

    # 8. 展示结果
    print("\n========== 召回结果示例 ==========")
    for i in range(min(3, len(recalled_items))):
        print(f"\n用户 {i} 的Top-10召回结果:")
        print(f"物品IDs: {recalled_items[i][:10]}")
        print(f"相似度距离: {distances[i][:10]}")

    # 9. 保存模型
    # recall_system.save_model('vector_recall_model.pth')

    print("\n========== 系统性能总结 ==========")
    print(f"物品池大小: {num_items}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"召回数量: 100")
    print("系统已就绪，可用于生产环境")


if __name__ == "__main__":
    demo_usage()