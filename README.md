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

### 图像理解

```bash
python qwen2.5-vl/infer_qwen2_5_vl_psa.py --type image \
    --input "./image.jpg" \
    --prompt "请描述这张图片" \
    --use-psa
```

### 视频理解

```bash
# 视频需要本地文件（decord 不支持 HTTPS）
python qwen2.5-vl/infer_qwen2_5_vl_psa.py --type video \
    --input "./video.mp4" \
    --prompt "请描述这个视频" \
    --fps 1.0 --use-psa
```

### Python API

```python
from infer_qwen2_5_vl_psa import QwenVLInference

# 初始化（启用 PSA）
model = QwenVLInference(use_psa=True)

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

```python
from psa_paddle import AttentionConfig

config = AttentionConfig(
    query_block=128,           # Query 块大小
    key_block=32,              # Key 块大小
    mask_mode="energybound",   # 掩码模式: energybound | topk
    mask_ratios={
        1: (0.0, 0.6),         # 重要性 0~60% → 全分辨率
        2: (0.6, 0.8),         # 重要性 60~80% → 2x池化
        4: (0.8, 0.9),         # 重要性 80~90% → 4x池化
        8: (0.9, 0.9),         # 重要性 90% → 8x池化
        0: (0.9, 1.0),         # 重要性 90~100% → 跳过
    },
    xattn_stride=16,           # 交叉注意力步长
)
```

### 掩码模式

| 模式 | 说明 |
|------|------|
| `energybound` | 基于能量阈值，相似度指标更好 |
| `topk` | 基于 Top-K 选择，极端稀疏度下更稳定 |

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
@article{li2025pyramid,
  title={Pyramid Sparse Attention},
  author={Li, Xiaolong and Gu, Youping and Lin, Xi and Wang, Weijie and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2512.04025},
  year={2025}
}
```

---

## 致谢

- [ziplab/Pyramid-Sparse-Attention](https://github.com/ziplab/Pyramid-Sparse-Attention) - 原始 PSA 实现
- [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX) - PaddlePaddle 多模态框架
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - 通义千问视觉语言模型
