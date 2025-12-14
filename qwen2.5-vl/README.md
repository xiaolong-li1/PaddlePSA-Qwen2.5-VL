# Qwen2.5-VL 推理 (PaddlePaddle)

基于 PaddlePaddle 的 Qwen2.5-VL 多模态模型推理，支持 PSA 稀疏注意力加速。

## 支持的模型

| 模型 | 参数量 | 推荐显存 |
|------|--------|----------|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | < 16GB |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | >= 24GB |

## 快速开始

### 命令行推理

```bash
# 使用 3B 模型（推荐，显存占用更少）
python infer_qwen2_5_vl_psa.py --type image \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "请描述这张图片"

# 使用 7B 模型（默认）
python infer_qwen2_5_vl_psa.py --type image \
    --input "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "请描述这张图片" --use-psa

# 视频推理 (使用本地文件)
python infer_qwen2_5_vl_psa.py --type video \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --input "./video.mp4" \
    --prompt "请描述这个视频" --fps 0.5 --use-psa
```

### Python 代码调用

```python
from infer_qwen2_5_vl_psa import QwenVLInference

# 初始化 3B 模型
model = QwenVLInference(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    dtype="bfloat16",
    use_psa=True,
)

# 图像推理
result = model.inference_image("./test.jpg", "请描述这张图片")
print(result)

# 视频推理
result = model.inference_video("./video.mp4", "请描述这个视频", fps=1.0)
print(result)

# 打印统计信息
model.print_stats()
```

### 自定义 PSA 配置

```python
from psa_paddle import AttentionConfig
from infer_qwen2_5_vl_psa import QwenVLInference

psa_config = AttentionConfig(
    query_block=128,
    mask_mode="energybound",
    mask_ratios={
        1: (0.0, 0.6),
        2: (0.6, 0.8),
        4: (0.8, 0.9),
        8: (0.9, 0.9),
        0: (0.9, 1.0),
    },
    xattn_stride=16,
)

model = QwenVLInference(use_psa=True, psa_config=psa_config)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型名称或路径 | Qwen/Qwen2.5-VL-7B-Instruct |
| --type | 输入类型 (image/video) | image |
| --input | 输入文件路径或URL | 必填 |
| --prompt | 提示词 | 请详细描述这张图片 |
| --max-tokens | 最大生成token数 | 512 |
| --dtype | 数据类型 | bfloat16 |
| --fps | 视频采样帧率 | 1.0 |
| --max-pixels | 视频最大像素数 | 151200 |
| --use-psa | 启用 PSA 稀疏注意力 | False |
| --query-block | PSA query block size | 128 |
| --xattn-stride | PSA xattn stride | 16 |
| --cache-dir | 模型缓存目录 | ./model_cache |

## 视频输入注意事项

当前 PaddleMIX 使用 `decord` 作为视频读取后端，**不支持 HTTPS 链接**。

| 输入类型 | 支持情况 |
|----------|----------|
| 本地文件 | ✅ 支持 |
| HTTP URL | ✅ 支持 |
| HTTPS URL | ❌ 不支持 |

如果视频是 HTTPS 链接，请先下载到本地：

```bash
# 下载视频
wget -O ./video.mp4 "https://example.com/video.mp4"

# 使用本地路径推理
python infer_qwen2_5_vl_psa.py --type video --input "./video.mp4" --prompt "请描述这个视频"
```

## 视频参数详解

### fps (帧率采样)

控制每秒从视频中采样多少帧：

| fps值 | 说明 | 示例 (10秒视频) |
|-------|------|-----------------|
| 0.5 | 每2秒采样1帧 | 5帧 |
| 1.0 | 每秒采样1帧 | 10帧 |
| 2.0 | 每秒采样2帧 | 20帧 |
| 4.0 | 每秒采样4帧 | 40帧 |

更高fps = 更多帧 = 更详细但更慢、更耗显存

### max_pixels (最大像素数)

控制每帧图像的分辨率上限：

| max_pixels | 等效分辨率 | 说明 |
|------------|-----------|------|
| 151200 | 360x420 | 默认，省显存 |
| 360000 | 600x600 | 中等质量 |
| 518400 | 720x720 | 高质量 |

### 使用建议

| 场景 | fps | max_pixels | 说明 |
|------|-----|------------|------|
| 快速预览 | 0.5 | 151200 | 省显存，快速 |
| 标准推理 | 1.0 | 151200 | 默认配置 |
| 详细分析 | 2.0 | 360000 | 更多细节 |
| 动作识别 | 4.0 | 151200 | 捕捉快速动作 |
| 长视频+PSA | 1.0 | 151200 | 配合PSA处理长序列 |

## PSA 稀疏注意力

PSA (Pyramid Adaptive Block Sparse Attention) 根据注意力分数自动选择不同的池化级别：

| 池化级别 | 说明 |
|----------|------|
| 1x | 原始分辨率注意力 |
| 2x | 2倍池化 |
| 4x | 4倍池化 |
| 8x | 8倍池化 |
| 0 | 跳过 (边界区域) |

## 目录结构

```
qwen2.5-vl/
├── infer_qwen2_5_vl_psa.py      # 推理脚本
├── qwen2_5_vl_psa_attention.py  # PSA 注意力替换
├── PaddleMIX/                    # PaddleMIX 依赖
└── README.md
```

## 依赖

- PaddlePaddle GPU 3.2.2+
- PaddleNLP 3.0.0b4+
- triton (PSA 需要)
- decord (视频处理)
