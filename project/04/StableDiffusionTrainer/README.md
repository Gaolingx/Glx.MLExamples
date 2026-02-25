# StableDiffusionTrainer

基于 **PyTorch Lightning + Diffusers** 的 Stable Diffusion 1.5 训练项目。当前实现以 **UNet 去噪器训练** 为核心，冻结 VAE 与 Text Encoder，支持自动断点续训、TensorBoard 可视化、训练健康指标监控与推理脚本。

---

## 1. 项目概览

- 训练入口：`scripts/train.py`
- 推理入口：`scripts/inference.py`
- 训练模块：`src/lightningmodule/sd15_module.py`
- 数据模块：`src/data/sd_datamodule.py`
- 训练工具与回调：`src/utils/training.py`
- 配置文件：`configs/train_config.json`

当前默认配置：

- 预训练基座：`runwayml/stable-diffusion-v1-5`
- CLIP：`CompVis/stable-diffusion-v1-4`
- VAE：`stabilityai/sd-vae-ft-mse`
- 数据集：`reach-vb/pokemon-blip-captions`
- 默认图像分辨率：`256`

---

## 2. 安装（Install）

### 2.1 环境要求

- Python 3.10+（推荐）
- CUDA 可选（有 GPU 时自动使用）
- Windows / Linux 均可

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要依赖包括：

- `torch`
- `pytorch-lightning`
- `diffusers`
- `transformers`
- `datasets`
- `tensorboard`

### 2.3（可选）Hugging Face 登录

首次下载模型或数据集时，若遇到权限/速率问题，可先登录：

```bash
huggingface-cli login
```

---

## 3. 项目结构（Project Structure）

```text
StableDiffusionTrainer/
├─ configs/
│  └─ train_config.json              # 训练/验证/日志/推理主配置
├─ models/
│  └─ unet/
│     └─ config.json                 # 自定义 UNet 结构配置
├─ scripts/
│  ├─ train.py                       # 训练入口
│  └─ inference.py                   # 推理入口
├─ src/
│  ├─ data/
│  │  └─ sd_datamodule.py            # DataModule 与 transforms/collate
│  ├─ lightningmodule/
│  │  └─ sd15_module.py              # LightningModule（损失/优化/验证采样/导出）
│  └─ utils/
│     ├─ config.py                   # JSON 配置加载
│     └─ training.py                 # 回调、logger、恢复训练、seed
├─ outputs/
│  └─ sd15-lightning/
│     ├─ checkpoints/                # Lightning ckpt 与 HF 导出
│     ├─ tb_logs/                    # TensorBoard 日志
│     └─ final_model/                # 最终 Diffusers 格式模型
└─ requirements.txt
```

---

## 4. 功能介绍

### 4.1 训练能力

- 基于 latent diffusion 的 UNet 去噪训练（MSE loss）
- 支持 `epsilon` / `v_prediction` 目标
- 支持 mixed precision（配置项 `training.precision`）
- 支持梯度检查点（`gradient_checkpointing`）
- 支持自动/手动优化流程切换（`manual_optimization`）

### 4.2 工程能力

- 自动种子设置（可复现）
- ModelCheckpoint + top-k + last ckpt
- 自动恢复训练（`--resume`）
- 恢复后可重置 LR / Scheduler（`resume_override`）
- 导出 HF 目录格式 checkpoint（`hf_checkpoint`）
- TensorBoard 日志与样本图可视化

### 4.3 推理能力

- 从 Lightning `.ckpt` 加载并构建 `StableDiffusionPipeline`
- 支持 prompt / negative prompt
- 支持多图批量保存

---

## 5. 配置说明（`configs/train_config.json`）

常用字段：

- `seed`：全局随机种子
- `output_dir`：训练输出目录
- `pretrained_model_name_or_path`：基础 SD1.5 路径
- `clip_model_name_or_path`：Tokenizer/TextEncoder 路径
- `vae_model_name_or_path`：VAE 路径
- `unet_config_path`：UNet 结构配置（可选）

### 5.1 dataset

- `name`：HF 数据集名称
- `split`：数据集分片（train）
- `image_column` / `caption_column`：图文字段
- `max_train_samples`：最大训练样本数
- `resolution`、`center_crop`、`random_flip`
- `num_workers`

### 5.2 training

- `batch_size`、`max_epochs`、`max_steps`
- `accumulate_grad_batches`
- `precision`
- `learning_rate`、`weight_decay`、Adam 参数
- `max_grad_norm`
- `lr_scheduler`、`lr_warmup_steps`
- `gradient_checkpointing`
- `enable_xformers_memory_efficient_attention`

### 5.3 validation

- `enabled`
- `every_n_steps`
- `num_inference_steps`
- `guidance_scale`
- `prompts`（周期采样可视化）

### 5.4 checkpoint / logging / inference

- checkpoint：保存目录、监控指标、保存周期
- logging：TensorBoard 路径、日志频率
- inference：推理步数、CFG、输出分辨率、输出目录

---

## 6. 运行示例（Run Examples）

### 6.1 启动训练

```bash
python scripts/train.py --config ./configs/train_config.json
```

### 6.2 从最新 checkpoint 自动续训

```bash
python scripts/train.py --config ./configs/train_config.json --resume
```

续训逻辑：优先使用 `checkpoint.dirpath/last.ckpt`，不存在则选最近修改的 `.ckpt`。

### 6.3 启动 TensorBoard

```bash
tensorboard --logdir outputs/sd15-lightning/tb_logs
```

### 6.4 推理示例

```bash
python scripts/inference.py ^
  --config ./configs/train_config.json ^
  --ckpt_path outputs/sd15-lightning/checkpoints/last.ckpt ^
  --prompt "Astronaut in a jungle, detailed, cinematic" ^
  --num_images 2
```

> 在 Windows `cmd` 中可使用 `^` 换行；也可写成单行命令。

---

## 7. 模型结构（Model Architecture）

训练时加载/构建的核心模块：

1. **Tokenizer / Text Encoder（CLIP）**
   - 文本编码为 `encoder_hidden_states`
   - 默认冻结参数

2. **VAE（AutoencoderKL）**
   - 输入图像编码到 latent 空间
   - latent 乘以 `vae.config.scaling_factor`
   - 默认冻结参数

3. **UNet2DConditionModel**
   - 在噪声 latent 与时间步条件下预测噪声/速度
   - 是主要可训练模块
   - 可从 `models/unet/config.json` 初始化自定义结构

4. **Noise Scheduler（DDPM）**
   - 训练时负责 `add_noise` 与 target 类型逻辑

5. **验证/推理 Scheduler（DPM-Solver++ 多步）**
   - 采样阶段使用 `DPMSolverMultistepScheduler`

---

## 8. 训练流程（Training Pipeline）

1. 读取 JSON 配置并设置随机种子
2. 构建 LightningModule（加载 tokenizer/text_encoder/vae/unet/scheduler）
3. 构建 DataModule（HF dataset + 图像增强 + tokenize + dataloader）
4. 构建 logger 与 callbacks
5. 初始化 `pl.Trainer`
6. 可选加载续训 ckpt（`--resume`）
7. 执行 `trainer.fit`
8. 训练结束后导出 `final_model`（Diffusers 格式）

### 单步损失计算逻辑

- `pixel_values -> VAE.encode -> latents`
- 采样 `noise` 与随机 `timesteps`
- `noisy_latents = scheduler.add_noise(latents, noise, timesteps)`
- UNet 前向预测 `model_pred`
- 根据 `prediction_type` 生成 target
- 计算 `MSE(model_pred, target)`

---

## 9. Metrics 与日志说明

项目会记录以下训练指标（部分由 callbacks 产生）：

### 9.1 核心训练指标

- `train/loss`
- `train/lr`

### 9.2 参数与梯度范数

- `train/grad_norm`
- `train/param_norm`
- `train/grad_norm_clip`
- `train/param_norm_clip`

### 9.3 训练健康度指标

- `train/step_time_ms`
- `train/samples_per_sec`
- `train/timestep_mean`
- `train/timestep_std`
- `train/latent_std`
- `train/noise_std`

### 9.4 可视化验证结果

当 `validation.enabled=true` 且达到 `every_n_steps`，将按照 `validation.prompts` 生成图像并写入 TensorBoard：

- `validation/sample_0`
- `validation/sample_1`
- ...

---

## 10. 输出产物说明

### 10.1 checkpoints

目录：`outputs/sd15-lightning/checkpoints/`

- Lightning `.ckpt`
- 对应的 `hf_checkpoint/`（包含 `unet/` 与 `training_config.json`）

### 10.2 final_model

目录：`outputs/sd15-lightning/final_model/`

为可直接被 Diffusers 加载的完整模型目录，一般包含：

- `unet/`
- `vae/`
- `text_encoder/`
- `tokenizer/`
- `scheduler/`
- `model_index.json`
- `training_config.json`

---

## 11. 常见问题（FAQ）

### Q1: 首次运行下载很慢/失败怎么办？

- 检查网络与代理配置
- 使用 `huggingface-cli login`
- 重试并确保 `cache_dir` 可写

### Q2: 显存不足（OOM）怎么办？

可组合调整：

- 降低 `training.batch_size`
- 提高 `accumulate_grad_batches` 以维持等效 batch
- 开启 `gradient_checkpointing`
- 将分辨率从 `256/512` 进一步降低
- 关闭不必要的并行进程

### Q3: 如何从断点继续训练？

使用：

```bash
python scripts/train.py --config ./configs/train_config.json --resume
```

并确认 `checkpoint.dirpath` 下存在 `last.ckpt` 或其他 `.ckpt`。

### Q4: 训练后如何导出可推理模型？

训练脚本会在 rank-0 自动导出：

- `outputs/sd15-lightning/final_model`

该目录可直接用于 Diffusers 推理。

### Q5: 为什么只有 UNet 在训练？

当前设计是冻结 VAE 与 CLIP Text Encoder，只优化 UNet。这是常见的 SD 微调策略，可降低训练成本并提升稳定性。

### Q6: 如何换成自己的数据集？

修改 `configs/train_config.json` 的 `dataset`：

- `name`：你的 HF 数据集名称或本地可加载源
- `image_column` / `caption_column`
- `max_train_samples`、`resolution` 等

并确保数据格式与字段一致。

---

## 12. 后续优化建议

- 增加验证集定量指标（如 FID/CLIPScore）
- 支持 LoRA/ControlNet 等轻量微调范式
- 增加多卡策略与更精细的性能 profiling
- 完善实验管理（配置版本化与结果追踪）

---

## 13. License

本仓库未单独声明 License 时，请按你的团队/组织规范补充许可证文件后再对外发布。
