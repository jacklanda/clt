# 代码库指标记录分析报告

## 概述
这是一个稀疏自动编码器（SAE/Transcoder）训练框架，包含完整的度量计算和Wandb集成。最近的主要变化是添加了L1稀疏性指标和其他评估指标。

---

## 1. 当前已记录到 Wandb 的指标列表

### 1.1 损失函数相关指标
- `ce_loss`: 交叉熵损失（仅当 loss_fn="ce" 时）
- `kl_loss`: KL散度（仅当 loss_fn 为 "kl" 或 "kl-fvu" 时）
- `acc_top1`: Top-1准确率（仅当 loss_fn 包含 "kl" 时）
- `loss/{name}`: 每个hookpoint的损失值（loss_fn="fvu" 时）
- `fvu/{name}`: 未解释方差分数（Fraction of Variance Unexplained）
- `fve/{name}`: 解释方差分数（Fraction of Variance Explained）

### 1.2 L0稀疏性指标（最新）
- `l0(per_token)/{name}`: 每个token的平均激活特征数
- `l0(per_sequence)/{name}`: 每个序列（128个token）的平均激活特征数
- `l0(per_batch)/{name}`: 整个batch中激活的特征总数
- `l0(per_feature)/{name}`: 每个特征的平均使用次数

### 1.3 L1稀疏性指标（新增 - 当前修改中）
- `l1(per_token)/{name}`: 每个token的激活值绝对和
- `l1(per_sequence)/{name}`: 每个序列的激活值绝对和
- `l1(per_batch)/{name}`: 整个batch的激活值绝对和
- `l1(per_feature)/{name}`: 每个特征的激活值绝对和

### 1.4 重构质量指标（新增 - 当前修改中）
- `cossim/{name}`: 目标与重建向量之间的余弦相似度
- `l2_ratio/{name}`: 重构与目标的L2范数比率
- `mse(l2)/{name}`: 均方误差（MSE）损失
- `relative_reconstruction_bias/{name}`: 相对重构偏差

### 1.5 特征健康指标
- `dead_pct/{name}`: 死特征比例（token阈值后未激活的特征百分比）
- `frac_dead/{name}`: 死特征的分数（0-1）

### 1.6 训练动力学指标（新增 - 当前修改中）
- `train/k`: 当前top-k值（可能随时间衰减）
- `train/lr`: 当前学习率

---

## 2. 代码中计算但可能未记录的指标

### 2.1 在ForwardOutput中计算但利用不足的指标
```python
# 以下指标由MidDecoder.__call__()计算，位于sparse_coder.py
```

1. **per_feature_l0 和 per_feature_l1**
   - 计算位置：sparse_coder.py 第312-321行
   - 当前记录：仅记录 `.mean()` 值
   - 潜在改进：可记录per-feature分布（直方图、分位数等）

2. **per_batch_l0 和 per_batch_l1**
   - 计算位置：sparse_coder.py 第311, 320行
   - 当前记录：仅记录 `.sum()` 值（累加）
   - 潜在改进：可记录平均值而非总和、标准差等

### 2.2 在训练循环中计算但未利用的统计信息

1. **latent_indices分布**
   - 计算位置：trainer.py 第806行
   - 使用场景：用于dead_mask更新
   - 未记录：哪些特征被激活频率最高、特征活跃度分布

2. **num_tokens_since_fired**
   - 计算位置：trainer.py 第274-277, 1010-1013行
   - 含义：自上次激活以来的token数
   - 未记录：特征生命周期统计、激活率时序趋势

3. **did_fire掩码**
   - 计算位置：trainer.py 第806-811行
   - 含义：每个训练步中哪些特征被激活
   - 未记录：激活动态、特征激活聚类

### 2.3 梯度相关指标
```python
# trainer.py 第1000行：优化器步骤，但未记录梯度统计
optimizer.step()
```
- 未计算：梯度范数、梯度稀疏性、参数更新幅度

### 2.4 损失曲线平滑化
- `best_loss`: 跟踪最佳损失（trainer.py 第283-287行），但仅在save_best时使用
- 未记录：best_loss历史、改进速度

---

## 3. 可从现有数据派生的新指标

### 3.1 从L0/L1组合派生
1. **Sparsity Trade-off比率**
   ```
   l1_per_token / (l0_per_token + epsilon)
   = 平均每个激活的权重
   ```
   
2. **特征活跃度熵**
   ```
   由per_feature_l0计算，表示特征使用的均衡性
   ```

3. **Reconstruction Efficiency**
   ```
   fve / l0_per_token
   = 每个特征的方差解释能力
   ```

### 3.2 从死特征统计派生
1. **特征复苏率**
   - 比较相邻步骤的死特征集合
   - 衡量特征是否在学习中恢复

2. **平均特征寿命**
   ```
   num_tokens_since_fired的分布统计
   ```

### 3.3 从重构指标派生
1. **重构稳定性**
   ```
   cossim的标准差（多个layers）
   ```

2. **偏差补偿效率**
   ```
   relative_reconstruction_bias的倒数
   ```

3. **L2范数扩展**
   ```
   l2_ratio > 1 表示能量扩展，< 1表示能量衰减
   ```

### 3.4 来自KL损失的派生指标
1. **模型行为偏差**
   ```
   当loss_fn="kl"时：
   - KL散度时间序列分析
   - accuracy改变率
   ```

2. **end-to-end影响**
   ```
   当loss_fn="kl-fvu"时：
   - kl_loss / fvu_loss比率
   - kl_coeff动态调整的有效性
   ```

---

## 4. 与稀疏编码相关的常见指标（对标行业标准）

### 4.1 标准SAE/自动编码器指标
| 指标 | 含义 | 实现状态 | 位置 |
|------|------|--------|------|
| Reconstruction MSE | 平均平方误差 | ✅ 已实现 | sparse_coder.py:267 |
| Fraction of Variance Explained | 解释方差比例 | ✅ 已实现 | sparse_coder.py:273 |
| Sparsity (L0) | 激活特征数 | ✅ 已实现 | sparse_coder.py:296-314 |
| Sparsity (L1) | 激活权重和 | ✅ 已实现 | sparse_coder.py:316-321 |
| Dead Features | 未激活特征比例 | ✅ 已实现 | sparse_coder.py:323-329 |
| Cosine Similarity | 方向相似度 | ✅ 已实现 | sparse_coder.py:290-292 |

### 4.2 已实现但记录策略可优化的指标
| 指标 | 当前记录方式 | 建议改进 |
|------|-----------|--------|
| Per-feature L0 | 仅记录均值 | 记录分布（直方图、分位数） |
| Per-feature L1 | 仅记录均值 | 同上，并加top-k活跃特征 |
| Per-batch L0/L1 | 仅记录总和 | 记录平均值、方差、分布 |
| L2 ratio | 仅全局值 | 加per-layer粒度 |

### 4.3 缺失的关键指标

#### A. 特征质量指标
- **特征相关性**：encoder权重之间的相关系数
- **特征正交性**：decoder列向量之间的夹角
- **特征区分度**：特征激活模式的多样性

#### B. 编码器/解码器指标
- **编码器稳定性**：权重范数、权重更新速率
- **解码器单位范数约束**：是否满足normalize_decoder
- **编码器-解码器对齐**：(W_enc @ W_dec.T)与单位矩阵的接近度

#### C. 训练动力学指标
- **收敛速度**：损失下降率
- **特征学习曲线**：新特征被激活的时间点
- **优化器健康**：梯度分布、学习率有效性

#### D. 泛化性指标
- **验证集表现**（如果有验证集）
- **层间特征重用**：跨层特征共享分析
- **cross-layer影响**（当cross_layer>0时）

---

## 5. 配置文件中的参数和功能

### 5.1 SparseCoderConfig（sparse_coder参数）
```python
# 架构参数
dtype: "none" | "float32" | "float16" | "bfloat16"
activation: "groupmax" | "topk" | "batchtopk"
expansion_factor: int = 32              # latents = input_width * expansion_factor
num_latents: int = 0                    # 0 = 使用expansion_factor
d_out: int = 0                          # transcoder输出维度���0 = 同输入维度

# 稀疏性参数
k: int = 32                             # top-k特征数
skip_connection: bool = False           # 添加跳跃连接

# 归一化参数
normalize_decoder: bool = True
normalize_io: bool = False              # 输入/输出归一化

# Transcoder特定参数
transcode: bool = False
tp_output: bool = True
n_targets: int = 0                      # 预测目标数（transcoder）
n_sources: int = 0                      # 跨层源数
per_source_tied: bool = False
secondary_target_tied: bool = False
coalesce_topk: "none" | "concat" | "per-layer" | "group"

# 其他
train_post_encoder: bool = True
post_encoder_scale: bool = False
use_fp8: bool = False
divide_cross_layer: bool = False
```

### 5.2 TrainConfig（训练参数）
```python
# 数据和优化参数
batch_size: int = 32
grad_acc_steps: int = 1
micro_acc_steps: int = 1

# 损失函数
loss_fn: "ce" | "fvu" | "kl" | "kl-fvu" = "fvu"
kl_coeff: float = 1.0
filter_bos: bool = False
remove_first_token: bool = False
remove_transcoded_modules: bool = False

# 优化器配置
optimizer: "adam" | "adam8" | "muon" | "signum" = "signum"
lr: float | None = None
lr_warmup_steps: int = 1000
b1: float = 0.9
b2: float = 0.999
force_lr_warmup: bool = False

# 特征生命周期
k_decay_steps: int = 0                  # k值衰减步数
k_anneal_mul: int = 10                  # 初始k倍数
dead_feature_threshold: int = 10_000_000
dead_latent_penalty: float = 0.0

# 层配置
hookpoints: list[str]
layers: list[int]
layer_stride: int = 1
cross_layer: int = 0
per_layer_k: list[int]

# 检查点和日志
save_every: int = 1000
save_best: bool = False
finetune: str | None = None
log_to_wandb: bool = True
wandb_log_frequency: int = 1
run_id: str | None = None（支持恢复运行）
```

---

## 6. 评估、验证和测试相关代码

### 6.1 测试代码
**位置**: `/sparsify/tests/`

1. **test_encode.py**
   - 测试融合编码器的前向/后向传播正确性
   - 与标准PyTorch实现的数值一致性验证
   - 性能对比（naive vs fused）

2. **test_decode.py**
   - 测试解码操作的正确性
   - Eager解码 vs Triton优化解码的一致性

### 6.2 数据评估脚本
**位置**: `/scripts/eval_data.py`

```python
# 功能：生成难度评估数据集
# - 找到模型预测错误的低频词token序列
# - 计算困惑度和损失
# - 用于评估SAE在困难样本上的表现
```

### 6.3 评估指标缺失
当前代码库中**缺少专门的评估模块**：
- ❌ 没有分布外(OOD)测试集评估
- ❌ 没有特征可解释��评估
- ❌ 没有模型性能影响评估（end-to-end任务）
- ❌ 没有特征聚类/相似性分析
- ❌ 没有梯度流分析

---

## 7. 最近的主要修改（当前yang/dev分支）

### 7.1 新增指标（commit c3fc3ed）
```python
# sparse_coder.py中的ForwardOutput类新增字段：
+ per_token_l1: float
+ per_sequence_l1: float
+ per_batch_l1: float
+ per_feature_l1: float

# trainer.py中新增Wandb日志：
+ info[f"l1(per_token)/{name}"]
+ info[f"l1(per_sequence)/{name}"]
+ info[f"l1(per_batch)/{name}"]
+ info[f"l1(per_feature)/{name}"]
+ info[f"cossim/{name}"]
+ info[f"relative_reconstruction_bias/{name}"]
+ info["train/k"]
+ info["train/lr"]
+ info[f"loss/{name}"]（结构化，per-hookpoint）
```

### 7.2 功能改进
1. **Transcoder维度支持**
   - d_out参数用于不同输入/输出维度
   - resolve_widths()同时处理input和output维度

2. **Wandb恢复功能**
   - 支持run_id用于继续现有运行
   - 环境变量支持：WANDB_PROJECT、WANDB_ENTITY

3. **L2比率计算改进**
   - 避免除零错误（注释掉的：`l2_norm_in_for_div[...] = 1`）

---

## 8. 建议的改进方向

### 8.1 立即可实施的改进
1. **特征活跃度排名**：记录top-10和bottom-10特征
2. **梯度诊断**：记录梯度范数和更新幅度
3. **学习曲线**：记录best_loss历史
4. **特征恢复率**：追踪死特征的复苏

### 8.2 中期改进
1. **完整的评估���架**
   - 在验证集上计算指标
   - End-to-end性能影响度量
   
2. **可视化增强**
   - 特征激活热力图
   - 特征相似性矩阵
   - 学习动力学曲线

### 8.3 长期改进
1. **自动化诊断系统**
   - 检测训练问题（崩溃、收敛停滞等）
   - 推荐超参数调整
   
2. **跨运行分析**
   - 比较不同配置的效果
   - 元学习最优配置
