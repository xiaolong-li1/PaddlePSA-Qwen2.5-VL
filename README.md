# PaddlePSA-Qwen2.5-VL

将 [Pyramid Sparse Attention (PSA)](https://github.com/ziplab/Pyramid-Sparse-Attention) 移植到 PaddlePaddle，并应用于 Qwen2.5-VL 多模态大模型，实现高效的图像/视频理解推理加速。

> **原项目**: [ziplab/Pyramid-Sparse-Attention](https://github.com/ziplab/Pyramid-Sparse-Attention)
> **论文**: [Pyramid Sparse Attention (arXiv:2512.04025)](https://arxiv.org/abs/2512.04025)
> **官网**: [http://ziplab.co/PSA](http://ziplab.co/PSA)

---

## 项目简介

本项目完成了以下工作：

1. **PSA 算法移植** - 将 PyTorch 版本的 Pyramid Sparse Attention 完整移植到 PaddlePaddle 框架
2. **Triton 内核适配** - 保留原有 Triton 高性能内核，适配 PaddlePaddle 张量操作
3. **Qwen2.5-VL 集成** - 实现注意力层即插即用替换，支持图像和视频多模态理解任务

---

## 什么是 PSA？

PSA (Pyramid Sparse Attention) 是一种金字塔式稀疏注意力机制，核心思想是：**不是所有的注意力都同等重要**。

PSA 通过自适应地为不同 query-key 块分配不同的注意力精度，根据重要性分数将注意力块分配到不同的金字塔层级，实现约 **90% 的计算稀疏度**：

| 层级 | 池化倍率 | 说明 |
|:----:|:--------:|------|
| Level 1 | 1x | 全分辨率，最重要的区域 |
| Level 2 | 2x | 2倍池化 |
| Level 4 | 4x | 4倍池化 |
| Level 8 | 8x | 8倍池化，次要区域 |
| Level 0 | - | 完全跳过，不重要的区域 |

---

## 安装

```bash
# 创建虚拟环境
uv venv --python 3.11
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt

# 安装 PSA 模块
uv pip install -e .

# 安装 PaddleMIX（项目已包含）
cd qwen2.5-vl/PaddleMIX
sh build_env.sh
cd ../..
```

---

## 快速开始

支持的模型：
- `Qwen/Qwen2.5-VL-3B-Instruct` (3B 参数，推荐显存 < 16GB)
- `Qwen/Qwen2.5-VL-7B-Instruct` (7B 参数，默认)

### 图像理解

```bash
# 使用 3B 模型
python qwen2.5-vl/infer_qwen2_5_vl_psa.py --type image \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --input "./image.jpg" \
    --prompt "请描述这张图片" \
    --use-psa

# 使用 7B 模型（默认）
python qwen2.5-vl/infer_qwen2_5_vl_psa.py --type image \
    --input "./image.jpg" \
    --prompt "请描述这张图片" \
    --use-psa
```

### 视频理解

```bash
# 视频需要本地文件（decord 不支持 HTTPS）
python qwen2.5-vl/infer_qwen2_5_vl_psa.py --type video \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --input "./video.mp4" \
    --prompt "请描述这个视频" \
    --fps 1.0 --use-psa
```

### Python API

```python
from infer_qwen2_5_vl_psa import QwenVLInference

# 初始化 3B 模型（启用 PSA）
model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    use_psa=True
)

# 图像推理
result = model.inference_image("./image.jpg", "请描述这张图片")

# 视频推理
result = model.inference_video("./video.mp4", "请描述这个视频", fps=1.0)

# 查看统计信息（包括稀疏度）
model.print_stats()
```

---

## 项目结构

```
PaddlePSA-Qwen2.5-VL/
├── psa_paddle/                             # PSA 核心模块
│   ├── PyramidAdaptiveBlockSparseAttn.py   # 主入口
│   ├── kernels/                            # Triton 内核
│   │   ├── psa_kernel_causal.py
│   │   ├── block_importance_kernels.py
│   │   └── attn_pooling_kernel.py
│   └── utils/
│       ├── block_importance.py
│       └── psa_logger.py
├── qwen2.5-vl/                             # Qwen2.5-VL 集成
│   ├── infer_qwen2_5_vl_psa.py             # 推理入口
│   ├── qwen2_5_vl_psa_attention.py         # 注意力替换
│   ├── PaddleMIX/                          # PaddleMIX 框架
│   └── README.md
├── tests/
├── requirements.txt
└── README.md
```

---

## PSA 配置

PSA 采用 [X-Attention](https://github.com/mit-han-lab/x-attention) 的反对角线采样方法进行块重要性估计，通过稀疏采样快速估算每个 query-key 块对的注意力重要性。

```python
from psa_paddle import AttentionConfig

config = AttentionConfig(
    # 基础配置
    text_length=512,           # 文本 token 长度（文本区域始终保持全精度）
    query_block=128,           # Query/Key 块大小（当前 Triton kernel 固定为 128）
    warmup_steps=0,            # 预热步数（预热期间使用标准注意力）

    # 块重要性估计参数 (X-Attention)
    xattn_stride=16,           # 采样步长：每 stride 个位置采样一个
    xattn_chunk_size=4096,     # 分块处理大小，用于控制显存

    # 掩码配置
    mask_mode="topk",          # 掩码模式: topk | energybound
    mask_ratios={              # 金字塔层级分配比例
        1: (0.0, 0.05),        # 重要性 Top 0~5% → 全分辨率 (1x)
        2: (0.05, 0.15),       # 重要性 5~15% → 2x 池化
        4: (0.15, 0.25),       # 重要性 15~25% → 4x 池化
        8: (0.25, 0.5),        # 重要性 25~50% → 8x 池化
        0: (0.5, 1.0),         # 重要性 50~100% → 跳过计算
    },

    # Key 相似度阈值（自适应池化级别选择）
    sim_2x_threshold=0.75,     # 相邻 Key 相似度 > 0.75 时允许 2x 池化
    sim_4x_threshold=0.7,      # Key 相似度 > 0.7 时允许 4x 池化
    sim_8x_threshold=0.7,      # Key 相似度 > 0.7 时允许 8x 池化
)
```

### 块重要性估计 (X-Attention)

块重要性估计采用 [X-Attention](https://github.com/mit-han-lab/x-attention) 的反对角线采样方法：

- **原理**：对完整注意力矩阵进行稀疏采样，每隔 `xattn_stride` 个位置采样一个 query-key 对
- **优势**：将 O(n²) 的重要性估计降低到 O(n²/stride²)，显著减少计算开销
- **参数**：`xattn_stride=16` 表示采样密度为 1/16，在保持精度的同时大幅提升效率

### 掩码模式

| 模式 | 说明 |
|------|------|
| `topk` | 基于 Top-K 排序选择，极端稀疏度下更稳定（默认） |
| `energybound` | 基于累积能量阈值，相似度指标更好 |

---

## 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | >= 3.10 |
| PaddlePaddle GPU | >= 3.2.2 |
| PaddleNLP | >= 3.0.0b4 |
| Triton | >= 3.5 |
| NVIDIA GPU | 必需 |

---

## 注意事项

1. **Triton 初始化** - 可能需要 `import torch` 来初始化 CUDA 驱动（后续版本会提供更优雅的 Triton-Paddle 适配方案）
2. **仅推理** - 当前版本仅支持推理，不支持训练
3. **视频输入** - decord 后端不支持 HTTPS，需下载到本地

---

## 引用

```bibtex
@misc{li2025psapyramidsparseattention,
  title={PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation},
  author={Xiaolong Li and Youping Gu and Xi Lin and Weijie Wang and Bohan Zhuang},
  year={2025},
  eprint={2512.04025},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.04025},
}
```

---

## 致谢

- [ziplab/Pyramid-Sparse-Attention](https://github.com/ziplab/Pyramid-Sparse-Attention) - 原始 PSA 实现
- [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX) - PaddlePaddle 多模态框架
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - 通义千问视觉语言模型
