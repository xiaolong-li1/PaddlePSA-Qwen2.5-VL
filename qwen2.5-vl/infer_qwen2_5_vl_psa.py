"""
Qwen2.5-VL 推理脚本 (PaddlePaddle)
支持图像和视频输入，可选 PSA 稀疏注意力加速
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PADDLE_DISABLE_WARNINGS"] = "1"

import logging
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("paddlenlp").setLevel(logging.ERROR)
logging.getLogger("paddlemix").setLevel(logging.ERROR)

import argparse
import sys
import os

# 相对路径导入
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_current_dir, 'PaddleMIX'))

import paddle

from psa_paddle import AttentionConfig
from qwen2_5_vl_psa_attention import replace_attention_with_psa, Qwen2_5_VLPSAAttention

from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)




def get_psa_stats(model):
    """获取所有PSA层的稀疏度统计"""
    total_sparsity = 0.0
    layer_count = 0
    last_seq_len = 0

    for name, module in model.named_sublayers():
        if isinstance(module, Qwen2_5_VLPSAAttention):
            if module.sparse_fn.sparsity_counter > 0:
                avg_sparsity = module.sparse_fn.sparsity_acc / module.sparse_fn.sparsity_counter
                total_sparsity += avg_sparsity
                layer_count += 1
                last_seq_len = module.sparse_fn.last_seq_len

    if layer_count > 0:
        return total_sparsity / layer_count, layer_count, last_seq_len
    return 0.0, 0, 0


class QwenVLInference:
    """Qwen2.5-VL 推理类 (PaddlePaddle)"""

    DEFAULT_CACHE_DIR = os.path.join(_project_root, "model_cache")
    DEFAULT_LOG_DIR = os.path.join(_project_root, "PSA_Log")

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: str = "bfloat16",
        use_psa: bool = False,
        psa_config: AttentionConfig = None,
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.use_psa = use_psa
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR

        os.makedirs(self.cache_dir, exist_ok=True)

        # 数据类型
        if dtype == "bfloat16":
            if paddle.amp.is_bfloat16_supported():
                self.dtype = "bfloat16"
            else:
                # V100 等不支持 bfloat16 的 GPU，使用 float16
                print("[Warning] bfloat16 not supported, falling back to float16")
                self.dtype = "float16"
        else:
            self.dtype = dtype

        print(f"[Config] model: {model_name}, dtype: {self.dtype}, PSA: {use_psa}")
        paddle.set_default_dtype(self.dtype)

        # 加载模型
        # 使用 sdpa 让 Vision Encoder 高效运行，PSA 只替换语言模型的注意力层
        print("[Loading] Model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=self.dtype,
            attn_implementation="sdpa",  # Vision Encoder 用 sdpa，语言模型后续被 PSA 替换
            cache_dir=self.cache_dir,
        )

        # 应用 PSA
        if use_psa:
            print("[PSA] Replacing attention layers...")
            self.psa_config = psa_config or AttentionConfig()
            self.model = replace_attention_with_psa(self.model, self.psa_config)

        # 加载处理器
        image_processor = Qwen2_5_VLImageProcessor()
        tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.processor = Qwen2_5_VLProcessor(image_processor, tokenizer)
        print("[Ready]")

    def inference_image(self, image_path: str, prompt: str, max_tokens: int = 512) -> str:
        """图像推理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, max_tokens)

    def inference_video(
        self,
        video_path: str,
        prompt: str,
        max_tokens: int = 512,
        fps: float = 1.0,
        max_pixels: int = 360 * 420,
    ) -> str:
        """视频推理"""
        # 构建视频内容
        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
            "fps": fps,
        }

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, max_tokens, is_video=True)

    def inference_multi_image(self, image_paths: list, prompt: str, max_tokens: int = 512) -> str:
        """多图推理"""
        content = [{"type": "image", "image": path} for path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self._generate(messages, max_tokens)

    def _generate(self, messages: list, max_tokens: int, is_video: bool = False) -> str:
        """内部生成方法"""
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # PaddleMIX 的 process_vision_info 不支持 return_video_kwargs 参数
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )

        print("[Inference] Generating...")
        with paddle.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        output_text = self.processor.batch_decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def print_stats(self):
        """打印统计信息"""
        print(f"\n[Memory] Allocated: {paddle.device.memory_allocated() / 1024 ** 3:.2f} GB, "
              f"Max: {paddle.device.max_memory_allocated() / 1024 ** 3:.2f} GB")

        if self.use_psa:
            avg_sparsity, layer_count, seq_len = get_psa_stats(self.model)
            print(f"[PSA] Avg Sparsity: {avg_sparsity:.4f} ({layer_count} layers), Seq Length: {seq_len}")
            print(f"[PSA] Log dir: {self.DEFAULT_LOG_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 推理 (PaddlePaddle)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型名称或路径")
    parser.add_argument("--type", type=str, default="image", choices=["image", "video"],
                        help="输入类型: image 或 video")
    parser.add_argument("--input", type=str,
                        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        help="输入文件路径或URL")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片",
                        help="提示词")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成token数")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="数据类型")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="视频采样帧率")
    parser.add_argument("--max-pixels", type=int, default=360*420,
                        help="视频最大像素数")

    # PSA 参数
    parser.add_argument("--use-psa", action="store_true",
                        help="使用 PSA 稀疏注意力")
    parser.add_argument("--query-block", type=int, default=128,
                        help="PSA query block size")
    parser.add_argument("--xattn-stride", type=int, default=16,
                        help="PSA xattn stride")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="模型缓存目录")

    args = parser.parse_args()

    # PSA 配置
    psa_config = None
    if args.use_psa:
        psa_config = AttentionConfig(
            query_block=args.query_block,
            xattn_stride=args.xattn_stride,
            mask_mode="energybound",
            mask_ratios={
                1: (0.0, 0.6),
                2: (0.6, 0.8),
                4: (0.8, 0.9),
                8: (0.9, 0.9),
                0: (0.9, 1.0),
            },
        )

    # 初始化模型
    model = QwenVLInference(
        model_name=args.model,
        dtype=args.dtype,
        use_psa=args.use_psa,
        psa_config=psa_config,
        cache_dir=args.cache_dir,
    )

    # 推理
    print(f"\n[Input] {args.input}")
    print(f"[Prompt] {args.prompt}\n")

    if args.type == "image":
        result = model.inference_image(args.input, args.prompt, args.max_tokens)
    else:
        result = model.inference_video(
            args.input, args.prompt, args.max_tokens,
            fps=args.fps, max_pixels=args.max_pixels
        )

    print("=" * 60)
    print("Output:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    model.print_stats()


if __name__ == "__main__":
    main()
