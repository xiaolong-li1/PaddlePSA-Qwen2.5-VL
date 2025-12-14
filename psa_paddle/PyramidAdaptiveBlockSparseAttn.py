"""
Pyramid Adaptive Block Sparse Attention for PaddlePaddle
Based on the PyTorch implementation, adapted for PaddlePaddle.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .kernels.psa_kernel_causal import sparse_attention_factory, calc_k_similarity_triton
from .utils.block_importance import estimate_block_importance
from .utils.psa_logger import PSALogger


@dataclass
class AttentionConfig:
    """Runtime configuration for pyramid adaptive sparse attention."""

    text_length: int = 512
    query_block: int = 128
    warmup_steps: int = 0

    mask_ratios: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: {
            1: (0.0, 0.05),
            2: (0.05, 0.15),
            4: (0.15, 0.25),
            8: (0.25, 0.5),
            0: (0.5, 1.0),
        }
    )
    mask_mode: str = "topk"
    importance_method: str = "xattn"  # xattn only
    xattn_stride: int = 16
    xattn_chunk_size: int = 4096
    causal_main: bool = True

    # K similarity thresholds for pooling level selection
    sim_2x_threshold: float = 0.75
    sim_4x_threshold: float = 0.7
    sim_8x_threshold: float = 0.7

    def __post_init__(self) -> None:
        self.importance_method = self.importance_method.lower()
        if self.importance_method not in {"xattn"}:
            raise ValueError("importance_method must be 'xattn'")

        self.mask_mode = self.mask_mode.lower()
        if self.mask_mode not in {"topk", "energybound"}:
            raise ValueError("mask_mode must be 'topk' or 'energybound'")


def x_block_imp_estimate(
    query,
    key,
    *,
    block_size: int,
    stride: int,
    chunk_size: int,
    causal: bool,
):
    return estimate_block_importance(
        query,
        key,
        block_size=block_size,
        stride=stride,
        norm=1.0,
        softmax=True,
        chunk_size=chunk_size,
        select_mode="inverse",
        use_triton=True,
        causal=causal,
        kdb=1,
    )


def transfer_attn_to_mask(
    attn,
    mask_ratios: Dict[int, Tuple[float, float]],
    text_length: int,
    mode: str,
    *,
    block_size: int,
    causal: bool,
):
    """Convert block-level attention scores to multi-scale pooling masks."""

    if not mask_ratios:
        raise ValueError("mask_ratios must not be empty")

    batch, heads, seq, _ = attn.shape
    mask = paddle.zeros_like(attn, dtype='int32')

    if mode not in {"topk", "energybound"}:
        raise ValueError("mode must be 'topk' or 'energybound'")

    # PaddlePaddle: paddle.sort returns sorted tensor, paddle.argsort returns indices
    sorted_weights = paddle.sort(attn, axis=-1, descending=True)
    indices = paddle.argsort(attn, axis=-1, descending=True)

    row_indices = paddle.arange(seq)
    if causal:
        valid_lengths = row_indices + 1
    else:
        valid_lengths = paddle.full([seq], seq, dtype='int64')

    if mode == "topk":
        position_range = paddle.arange(seq)
        for value, (start_ratio, end_ratio) in mask_ratios.items():
            start_idx = (valid_lengths.astype('float32') * start_ratio).astype('int64').clip(max=seq)
            end_idx = (valid_lengths.astype('float32') * end_ratio).astype('int64').clip(max=seq)
            range_mask = (position_range.unsqueeze(0) >= start_idx.unsqueeze(1)) & (
                position_range.unsqueeze(0) < end_idx.unsqueeze(1)
            )
            gathered_mask = paddle.take_along_axis(mask, indices, axis=-1)
            update = paddle.full_like(gathered_mask, value)
            new_values = paddle.where(
                range_mask.unsqueeze(0).unsqueeze(0).astype('bool'),
                update,
                gathered_mask,
            )
            inv_indices = paddle.argsort(indices, axis=-1)
            mask = paddle.take_along_axis(new_values, inv_indices, axis=-1)
    else:
        row_sum = attn.sum(axis=-1, keepdim=True).clip(min=1e-10)
        energy_ratio = paddle.cumsum(sorted_weights, axis=-1) / row_sum
        prev_upper = 0.0
        for value, (start_ratio, end_ratio) in mask_ratios.items():
            lower = max(start_ratio, prev_upper)
            prev_upper = end_ratio
            range_mask = (energy_ratio > lower) & (energy_ratio <= end_ratio)
            gathered_mask = paddle.take_along_axis(mask, indices, axis=-1)
            update = paddle.full_like(gathered_mask, value)
            new_values = paddle.where(
                range_mask,
                update,
                gathered_mask,
            )
            inv_indices = paddle.argsort(indices, axis=-1)
            mask = paddle.take_along_axis(new_values, inv_indices, axis=-1)

    num_special_blocks = min((
        math.ceil(text_length / block_size) if text_length > 0 else 0
    ), mask.shape[-1])
    if num_special_blocks:
        mask[:, :, :, -num_special_blocks:] = 1
        mask[:, :, -num_special_blocks:, :] = 1

    if causal:
        upper = paddle.triu(
            paddle.ones([seq, seq], dtype='bool'), diagonal=1
        )
        mask = mask * (1 - upper.unsqueeze(0).unsqueeze(0).astype('int32'))

    diag_mask = paddle.eye(seq, dtype='int32')
    mask = paddle.maximum(mask, diag_mask.unsqueeze(0).unsqueeze(0))
    mask[:, :, :, 0] = 1
    return mask


def calc_density(mask, causal: bool) -> Tuple[float, list]:
    """Calculate density from mask."""
    density = paddle.zeros_like(mask, dtype='float32')
    non_zero = mask > 0
    density = paddle.where(non_zero, 1.0 / mask.astype('float32'), density)

    seq = mask.shape[-1]
    if causal:
        valid = paddle.tril(
            paddle.ones([seq, seq], dtype='float32'), diagonal=0
        )
    else:
        valid = paddle.ones([seq, seq], dtype='float32')

    valid = valid.reshape([1, 1, seq, seq]).expand([mask.shape[0], mask.shape[1], -1, -1])
    density_valid = density * valid

    valid_counts = valid.sum().clip(min=1.0)
    avg_density = density_valid.sum() / valid_counts

    per_head_counts = valid.sum(axis=[0, 2, 3]).clip(min=1.0)
    per_head_density = (
        density_valid.sum(axis=[0, 2, 3]) / per_head_counts
    ).tolist()

    return avg_density.item(), per_head_density


def adaptive_block_sparse_attn(
    q,
    k,
    v,
    config: AttentionConfig,
    sparse_attention_fn,
) -> Tuple[paddle.Tensor, float, list]:
    """Perform adaptive block sparse attention."""
    block_size = config.query_block

    with paddle.no_grad():
        if config.importance_method == "xattn":
            attn_est = x_block_imp_estimate(
                q,
                k,
                block_size=block_size,
                stride=config.xattn_stride,
                chunk_size=config.xattn_chunk_size,
                causal=config.causal_main,
            )
        else:
            raise ValueError(
                f"Unknown importance_method: {config.importance_method}"
            )

        mask = transfer_attn_to_mask(
            attn_est,
            config.mask_ratios,
            config.text_length,
            config.mask_mode,
            block_size=block_size,
            causal=config.causal_main,
        )

        sim_mask = calc_k_similarity_triton(
            k,
            block_size,
            config,
        ).unsqueeze(-2).tile([1, 1, mask.shape[-2], 1])
        mask = paddle.minimum(sim_mask.astype(mask.dtype), mask)

    out = sparse_attention_fn(
        q,
        k,
        v,
        mask,
        None,
    )

    avg_density, per_head_density = calc_density(mask, config.causal_main)
    return out, 1 - avg_density, per_head_density


class PyramidAdaptiveBlockSparseAttnTrain(nn.Layer):
    """Adaptive sparse attention with optional logging."""

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        *,
        layer_idx: int = -1,
        log_dir: Optional[str] = None,
        log_prefix: str = "psa_sparsity",
        session_name: Optional[str] = None,
        **extra: object,
    ) -> None:
        super().__init__()
        self.config = config or AttentionConfig()
        self.layer_idx = layer_idx

        self.sparse_attention_fn = sparse_attention_factory(
            causal=self.config.causal_main,
        )

        self.sparsity_acc = 0.0
        self.sparsity_counter = 0
        self.last_seq_len = 0

        if log_dir is None and "save_dir" in extra:
            log_dir = extra["save_dir"]

        # Initialize PSA logger
        self.logger: Optional[PSALogger] = None
        if log_dir is not None:
            self.logger = PSALogger(
                log_dir=log_dir,
                config=self.config,
                layer_idx=layer_idx,
                session_name=session_name or log_prefix,
            )

    def __del__(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def forward(
        self,
        q,
        k,
        v,
        layer_idx: Optional[int] = None,
    ):
        if layer_idx is not None:
            self.layer_idx = layer_idx
        self.sparsity_counter += 1
        self.last_seq_len = q.shape[2]  # (batch, heads, seq, dim)

        in_warmup = self.sparsity_counter <= self.config.warmup_steps
        if in_warmup:
            return F.scaled_dot_product_attention(q, k, v)

        out, sparsity, per_head_density = adaptive_block_sparse_attn(
            q,
            k,
            v,
            self.config,
            self.sparse_attention_fn,
        )

        self.sparsity_acc += sparsity

        # Log using PSA logger
        if self.logger is not None:
            qkv_shape = {
                "Q": tuple(q.shape),
                "K": tuple(k.shape),
                "V": tuple(v.shape),
            }
            self.logger.log_sparsity(
                layer_idx=self.layer_idx,
                sparsity=sparsity,
                per_head_density=per_head_density,
                sequence_length=q.shape[2],
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                qkv_shape=qkv_shape,
            )
            # Print progress at intervals
            self.logger.print_progress(self.layer_idx, interval=100)

        return out
