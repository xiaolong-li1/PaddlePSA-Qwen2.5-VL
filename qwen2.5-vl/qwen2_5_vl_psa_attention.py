"""
Qwen2.5-VL PSA Attention Implementation for PaddlePaddle
将Qwen2.5-VL的注意力机制替换为Pyramid Sparse Attention (PSA)
"""

import math
from typing import Optional, Tuple

import os
import sys

# 相对路径导入
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_current_dir, 'PaddleMIX'))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers.linear_utils import Linear

# 导入PSA模块
from psa_paddle import AttentionConfig, PyramidAdaptiveBlockSparseAttnTrain

# 导入PaddleMIX的原始函数
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


class Qwen2_5_VLPSAAttention(nn.Layer):
    """
    Qwen2.5-VL Attention with Pyramid Sparse Attention (PSA) for PaddlePaddle.
    在prefill阶段使用PSA稀疏注意力，在decode阶段使用标准注意力。
    """

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        psa_config: Optional[AttentionConfig] = None,
        log_dir: str = os.path.join(_project_root, "PSA_Log"),
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.output_token_idx = 0

        # 模型维度
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim ** -0.5

        # 处理tensor parallel
        if hasattr(config, 'tensor_parallel_degree') and config.tensor_parallel_degree > 1:
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

        # 投影层
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=True)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

        # PSA配置
        self.sparse_config = psa_config or AttentionConfig(
            text_length=512,
            query_block=128,
            warmup_steps=0,
            mask_mode="energybound",
            mask_ratios={
                1: (0.0, 0.6),
                2: (0.6, 0.8),
                4: (0.8, 0.9),
                8: (0.9, 0.9),
                0: (0.9, 1.0),
            },
            importance_method="xattn",
            xattn_stride=16,
            xattn_chunk_size=8192,
            causal_main=True,
            sim_2x_threshold=0.7,
            sim_4x_threshold=0.7,
            sim_8x_threshold=0.6,
        )

        # PSA模块
        self.sparse_fn = PyramidAdaptiveBlockSparseAttnTrain(
            config=self.sparse_config,
            layer_idx=self.layer_idx,
            log_dir=log_dir if log_dir else None,
        )

        # 控制是否在prefill阶段使用稀疏注意力
        self.prefill_sparsity = True
        self.log_dir = log_dir

    def sparse_forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """使用PSA进行稀疏注意力计算"""
        # 如果有GQA，需要扩展K和V
        if self.num_key_value_groups > 1:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # 调用PSA
        out = self.sparse_fn(q, k, v, layer_idx=self.layer_idx)

        # 转换输出形状: (batch, heads, seq, dim) -> (batch, seq, hidden)
        out = out.transpose([0, 2, 1, 3]).reshape([q.shape[0], q.shape[2], -1])
        return out, None

    def eager_forward(
        self,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """标准eager attention用于decode阶段"""
        bsz, num_heads, q_len, head_dim = query_states.shape

        # 扩展K和V用于GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力分数
        query_states = query_states.astype("float32")
        key_states = key_states.astype("float32")

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = paddle.matmul(attn_weights.astype(value_states.dtype), value_states)

        # 转换输出形状
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, -1])

        return attn_output

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        # 计算Q, K, V投影
        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.config.dtype)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 重塑为多头格式
        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])

        # 转置为 (batch, heads, seq, dim)
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        # 应用RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
            )

        # 处理KV cache
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if cache_position is not None:
                kv_seq_len += cache_position[0] + 1
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 选择注意力计算方式
        # prefill阶段(q_len > 1)且启用稀疏注意力时使用PSA
        if q_len > 1 and self.prefill_sparsity:
            attn_output, attn_weights = self.sparse_forward(
                query_states, key_states, value_states
            )
        else:
            # decode阶段使用标准注意力
            attn_output = self.eager_forward(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None

        # 输出投影
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def replace_attention_with_psa(model, psa_config: Optional[AttentionConfig] = None):
    """
    将模型中的Qwen2_5_VLAttention替换为Qwen2_5_VLPSAAttention

    Args:
        model: Qwen2.5-VL模型
        psa_config: PSA配置，如果为None则使用默认配置

    Returns:
        替换后的模型
    """
    from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLAttention,
        Qwen2_5_VLFlashAttention2,
        Qwen2_5_VLSdpaAttention,
    )

    # 打印配置
    cfg = psa_config or AttentionConfig()
    print(f"[PSA Config] mode={cfg.mask_mode}, block={cfg.query_block}, stride={cfg.xattn_stride}")
    print(f"[PSA Config] ratios={cfg.mask_ratios}")
    print(f"[PSA Config] sim_thresholds: 2x={cfg.sim_2x_threshold}, 4x={cfg.sim_4x_threshold}, 8x={cfg.sim_8x_threshold}")

    attention_classes = (Qwen2_5_VLAttention, Qwen2_5_VLFlashAttention2, Qwen2_5_VLSdpaAttention)

    replaced_count = 0

    def replace_attention_in_module(module, parent_name=""):
        nonlocal replaced_count
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, attention_classes):
                # 创建PSA attention替换
                psa_attention = Qwen2_5_VLPSAAttention(
                    config=child.config,
                    layer_idx=child.layer_idx,
                    psa_config=psa_config,
                )

                # 复制权重
                psa_attention.q_proj.weight.set_value(child.q_proj.weight)
                psa_attention.k_proj.weight.set_value(child.k_proj.weight)
                psa_attention.v_proj.weight.set_value(child.v_proj.weight)
                psa_attention.o_proj.weight.set_value(child.o_proj.weight)

                if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                    psa_attention.q_proj.bias.set_value(child.q_proj.bias)
                if hasattr(child.k_proj, 'bias') and child.k_proj.bias is not None:
                    psa_attention.k_proj.bias.set_value(child.k_proj.bias)
                if hasattr(child.v_proj, 'bias') and child.v_proj.bias is not None:
                    psa_attention.v_proj.bias.set_value(child.v_proj.bias)

                # 替换模块
                setattr(module, name, psa_attention)
                replaced_count += 1
            else:
                replace_attention_in_module(child, full_name)

    replace_attention_in_module(model)
    print(f"[PSA] Replaced {replaced_count} attention layers")

    return model


def patch_qwen2_5_vl_for_psa():
    """
    Monkey-patch PaddleMIX中的Qwen2.5-VL以使用PSA注意力。
    在导入模型之前调用此函数。
    """
    import paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling

    # 保存原始类
    modeling._OriginalQwen2_5_VLAttention = modeling.Qwen2_5_VLAttention

    # 替换为PSA版本
    modeling.Qwen2_5_VLAttention = Qwen2_5_VLPSAAttention

    print("Patched Qwen2_5_VLAttention with PSA version")
