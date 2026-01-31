"""
主程序 - 训练和测试矢量召回系统
"""
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import time

from config import ModelConfig, HNSWConfig, TrainingConfig, RecallConfig
from models.two_tower import TwoTowerModel
from data.dataset import (
    RecommendationDataset,
    collate_fn,
    generate_synthetic_data
)
from trainer import Trainer, IndexBuilder
from recall_service import VectorRecallService


def train(args):
    """训练模型"""
    print("=" * 60)
    print("Vector Recall System - Training")
    print("=" * 60)
    
    # 配置
    model_config = ModelConfig(
        num_users=args.num_users,
        num_items=args.num_items,
        embedding_dim=args.embedding_dim
    )
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_negatives=args.num_negatives
    )
    
    # 生成数据
    (user_item_pairs, user_features, item_features, 
     item_categories, item_tags, user_history) = generate_synthetic_data(
        num_users=args.num_users,
        num_items=args.num_items,
        num_interactions=args.num_interactions
    )
    
    # 创建数据集
    dataset = RecommendationDataset(
        user_item_pairs=user_item_pairs,
        user_features=user_features,
        item_features=item_features,
        item_categories=item_categories,
        item_tags=item_tags,
        num_items=args.num_items,
        num_negatives=train_config.num_negatives,
        user_history=user_history
    )
    
    # 分割训练/验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 创建模型
    model = TwoTowerModel(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train()
    
    # 构建索引
    print("\nBuilding HNSW index...")
    hnsw_config = HNSWConfig(dim=args.embedding_dim, max_elements=args.num_items)
    index_builder = IndexBuilder(model, hnsw_config)
    
    index = index_builder.build_index(
        item_ids=np.arange(args.num_items),
        item_categories=item_categories,
        item_tags=item_tags,
        item_features=item_features
    )
    
    # 保存索引
    index.save('./index/hnsw')
    print("Index saved to ./index/hnsw")
    
    return model, index, user_features


def benchmark(model, index, user_features, num_queries=1000):
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    # 创建召回服务
    service = VectorRecallService(model, index)
    
    # 预热
    for _ in range(10):
        user_id = np.random.randint(0, len(user_features))
        _ = service.recall(user_id, user_features[user_id], top_k=500)
    
    # 单次查询延迟测试
    latencies = []
    for _ in range(num_queries):
        user_id = np.random.randint(0, len(user_features))
        result = service.recall(user_id, user_features[user_id], top_k=500)
        latencies.append(result.latency_ms)
    
    latencies = np.array(latencies)
    
    print(f"\nSingle Query Latency (Top-500):")
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f} ms")
    print(f"  QPS:  {1000 / latencies.mean():.1f}")
    
    # 批量查询测试
    batch_sizes = [16, 32, 64, 128]
    print(f"\nBatch Query Throughput:")
    
    for batch_size in batch_sizes:
        user_ids = np.random.randint(0, len(user_features), size=batch_size)
        batch_features = user_features[user_ids]
        
        start = time.time()
        _ = service.batch_recall(user_ids, batch_features, top_k=500)
        elapsed = time.time() - start
        
        qps = batch_size / elapsed
        print(f"  Batch={batch_size}: {elapsed*1000:.2f}ms, QPS={qps:.1f}")
    
    # 索引统计
    print(f"\nIndex Statistics:")
    stats = service.get_stats()
    for key, value in stats['index_stats'].items():
        print(f"  {key}: {value}")


def demo(model, index, user_features):
    """演示召回"""
    print("\n" + "=" * 60)
    print("Recall Demo")
    print("=" * 60)
    
    service = VectorRecallService(model, index)
    
    # 随机选择一个用户
    user_id = np.random.randint(0, len(user_features))
    
    print(f"\nRecalling for User {user_id}...")
    result = service.recall(user_id, user_features[user_id], top_k=20)
    
    print(f"Latency: {result.latency_ms:.2f} ms")
    print(f"\nTop-20 Recommended Items:")
    print("-" * 40)
    print(f"{'Rank':<6} {'Item ID':<12} {'Score':<12}")
    print("-" * 40)
    
    for i, (item_id, score) in enumerate(zip(result.item_ids[:20], result.scores[:20])):
        print(f"{i+1:<6} {item_id:<12} {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Vector Recall System')
    parser.add_argument('--num_users', type=int, default=10000)
    parser.add_argument('--num_items', type=int, default=100000)
    parser.add_argument('--num_interactions', type=int, default=500000)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_negatives', type=int, default=10)
    parser.add_argument('--skip_benchmark', action='store_true')
    
    args = parser.parse_args()
    
    # 训练
    model, index, user_features = train(args)
    
    # 基准测试
    if not args.skip_benchmark:
        benchmark(model, index, user_features)
    
    # 演示
    demo(model, index, user_features)


if __name__ == '__main__':
    main()
