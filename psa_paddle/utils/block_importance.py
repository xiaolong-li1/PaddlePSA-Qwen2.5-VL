"""
Block Importance Estimation for PaddlePaddle
Based on the PyTorch implementation, adapted for PaddlePaddle.
"""

import math
import paddle
import paddle.nn.functional as F

from ..kernels.block_importance_kernels import (
    flat_group_gemm_fuse_reshape,
    softmax_fuse_block_sum,
)


def pad_to_multiple(x, multiple):
    """
    Pad sequence dimension (dim=2) to make it a multiple of `multiple`.
    x: [B, H, L, D]
    """
    L = x.shape[2]
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        x = F.pad(x, [0, 0, 0, pad_len], mode='replicate')
    return x


def calc_k_similarity(k, block_size):
    """
    Calculate key similarity for adaptive sparse attention.
    k: [B, H, L, D]
    """
    SIM_2_THRESHOLD = 0.75
    SIM_4_THRESHOLD = 0.7
    SIM_8_THRESHOLD = 0.7
    
    k = pad_to_multiple(k, block_size)
    k_chunked_num = k.shape[-2] // block_size
    
    # Reshape to [B, H, k_chunked_num, block_size, D]
    k_chunked = k[:, :, :k_chunked_num*block_size, :].reshape(
        [k.shape[0], k.shape[1], k_chunked_num, block_size, k.shape[-1]]
    )
    
    # Split even and odd positions
    k_chunked_1 = k_chunked[:, :, :, ::2, :]   # [B, H, k_chunked_num, block_size//2, D]
    k_chunked_2 = k_chunked[:, :, :, 1::2, :]  # [B, H, k_chunked_num, block_size//2, D]
    
    # Compute cosine similarity
    similarity_2 = F.cosine_similarity(k_chunked_1, k_chunked_2, axis=-1).mean(axis=-1)
    similarity_4 = F.cosine_similarity(
        k_chunked[:, :, :, 0::4, :],
        k_chunked[:, :, :, 3::4, :],
        axis=-1
    ).mean(axis=-1)
    similarity_8 = F.cosine_similarity(
        k_chunked[:, :, :, 0::8, :],
        k_chunked[:, :, :, 7::8, :],
        axis=-1
    ).mean(axis=-1)
    
    sim_2_mask = 2 * (similarity_2 > SIM_2_THRESHOLD).astype('int32')
    sim_4_mask = 4 * (similarity_4 > SIM_4_THRESHOLD).astype('int32')
    sim_8_mask = 8 * (similarity_8 > SIM_8_THRESHOLD).astype('int32')
    
    one_tensor = paddle.ones_like(sim_2_mask)
    sim_mask = paddle.maximum(one_tensor, paddle.maximum(sim_2_mask, paddle.maximum(sim_4_mask, sim_8_mask)))
    
    return sim_mask


def estimate_block_importance(
    query_states,
    key_states,
    block_size: int,
    stride: int,
    *,
    norm: float = 1.0,
    softmax: bool = True,
    chunk_size: int = 16384,
    select_mode: str = "inverse",
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
):
    """Estimate block-wise attention importance via antidiagonal sampling."""

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    _, num_q_head, q_len, _ = query_states.shape
    assert num_q_head == num_kv_head

    target_place = key_states.place

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, [0, 0, 0, k_num_to_pad], value=0)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, [0, 0, 0, q_num_to_pad], value=0)
    else:
        pad_query_states = query_states

    attn_sum_list = []

    # Check if we can use triton
    if use_triton:
        if not paddle.device.is_compiled_with_cuda():
            use_triton = False
        else:
            # Check device compatibility
            device_name = paddle.device.cuda.get_device_name()
            if "100" not in device_name and "200" not in device_name and "6000" not in device_name:
                use_triton = False
                print("Setting use_triton to false. Triton kernel not supported on this device")

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    if not use_triton:
        if select_mode == "random":
            perm_idx = paddle.randperm(stride)
            reshaped_key = paddle.concat(
                [pad_key_states[:, :, k::stride, :] for k in range(stride)], axis=-1
            )
            reshaped_query = paddle.concat(
                [pad_query_states[:, :, perm_idx[i].item()::stride, :] for i in range(stride)],
                axis=-1,
            )
        elif select_mode in {"inverse", ""}:
            reshaped_key = paddle.concat(
                [pad_key_states[:, :, k::stride, :] for k in range(stride)], axis=-1
            )
            reshaped_query = paddle.concat(
                [pad_query_states[:, :, (stride - 1 - q)::(stride * kdb), :] for q in range(stride)],
                axis=-1,
            )
        elif select_mode == "slash":
            reshaped_key = paddle.concat(
                [pad_key_states[:, :, k::stride, :] for k in range(stride)], axis=-1
            )
            reshaped_query = paddle.concat(
                [pad_query_states[:, :, q::stride, :] for q in range(stride)], axis=-1
            )
        else:
            raise ValueError(f"Unsupported select_mode: {select_mode}")
        
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size) * stride:(chunk_idx * reshaped_chunk_size + reshaped_chunk_size) * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) // kdb:(chunk_idx * reshaped_chunk_size + reshaped_chunk_size) // kdb,
                :,
            ]
            attn_weights_slice = paddle.matmul(
                chunked_query,
                reshaped_key.transpose([0, 1, 3, 2]),
            )

            attn_weights_slice = attn_weights_slice / math.sqrt(head_dim) / stride / norm

            if causal:
                causal_mask = paddle.zeros(
                    shape=[batch_size, num_q_head, reshaped_chunk_size, reshaped_chunk_size * k_chunk_num],
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                
                triu_mask = paddle.triu(
                    paddle.ones(shape=[1, num_q_head, reshaped_chunk_size, reshaped_chunk_size]) * float("-inf"),
                    diagonal=1,
                )
                causal_mask[:, :, :, chunk_start:chunk_end] = triu_mask

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)):, :] = float("-inf")

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1::kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice.astype('float32'), axis=-1
                ).astype(pad_query_states.dtype)
            else:
                attn_weights_slice = paddle.exp(attn_weights_slice).astype(pad_query_states.dtype)
            
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)):, :] = 0

            attn_sum = (
                attn_weights_slice.reshape([
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                ])
                .sum(axis=-1)
                .sum(axis=-2)
            )

        attn_sum_list.append(attn_sum)

        del attn_weights_slice
        if not use_triton:
            del chunked_query

    if not use_triton:
        del reshaped_query, reshaped_key

    attn_sums = paddle.concat(attn_sum_list, axis=-2)

    q_blocks_valid = math.ceil(q_len / block_size)
    k_blocks_valid = math.ceil(k_len / block_size)

    attn_sums = attn_sums[:, :, :q_blocks_valid, :k_blocks_valid]

    return attn_sums
