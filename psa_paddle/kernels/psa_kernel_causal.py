"""
PSA Triton Kernel with Optional Causal Mask Support for PaddlePaddle

Based on the PyTorch implementation from qwenvl2.5-clean, adapted for PaddlePaddle.

This kernel supports:
- Multi-level pooled KV representations (1x, 2x, 4x, 8x pooling)
- Optional causal masking with proper handling of pooled tokens (causal=True/False)
- Old mask format where mask values indicate pooling level (0, 1, 2, 4, 8)
"""

import math
import paddle
import paddle.nn.functional as F

import triton
import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# ============ calc_k_similarity Triton Kernel ============

@triton.jit
def _calc_k_similarity_split_kernel(
    K_ptr,           # [B, H, N, D]
    Out_ptr,         # [B, H, N_blocks]
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_ob, stride_oh, stride_on,
    t2, t4, t8,
    N,               # total sequence length
    USER_BLOCK_SIZE: tl.constexpr,   # e.g. 32
    HEAD_DIM: tl.constexpr,          # e.g. 128
    TRITON_BLOCK: tl.constexpr,      # e.g. 128 (number of K vectors per triton program)
):
    """
    Fused kernel using split optimization pattern.

    Grid: (num_chunks, B * H)
    - Each program processes TRITON_BLOCK K vectors (e.g. 128)
    - This corresponds to NUM_SUB = TRITON_BLOCK // USER_BLOCK_SIZE user blocks (e.g. 4)

    Algorithm:
    1. Load even (0,2,4,...) and odd (1,3,5,...) K vectors - only 2 loads
    2. Precompute norms for even and odd
    3. Compute 2x similarity directly
    4. Use split to get 4x positions, compute 4x similarity
    5. Split again for 8x positions, compute 8x similarity
    6. Apply thresholds and store results
    """
    pid_chunk = tl.program_id(0)  # which chunk of TRITON_BLOCK vectors
    pid_bh = tl.program_id(1)     # which batch*head

    # Number of user blocks per triton block
    NUM_SUB: tl.constexpr = TRITON_BLOCK // USER_BLOCK_SIZE  # e.g. 4
    HALF_BS: tl.constexpr = USER_BLOCK_SIZE // 2             # e.g. 16

    # Base K pointer for this batch*head
    k_base = K_ptr + pid_bh * stride_kh

    # ========== 1. Compute load offsets ==========
    # offs_base: [0, USER_BLOCK_SIZE, 2*USER_BLOCK_SIZE, ...]
    offs_base = (pid_chunk * TRITON_BLOCK) + tl.arange(0, NUM_SUB) * USER_BLOCK_SIZE
    offs_base = offs_base[:, None]  # [NUM_SUB, 1]

    # r_half: [0, 1, 2, ..., HALF_BS-1]
    r_half = tl.arange(0, HALF_BS)[None, :]  # [1, HALF_BS]

    # even: 0, 2, 4, ...  odd: 1, 3, 5, ...
    offs_even = offs_base + r_half * 2       # [NUM_SUB, HALF_BS]
    offs_odd = offs_base + r_half * 2 + 1    # [NUM_SUB, HALF_BS]

    offs_d = tl.arange(0, HEAD_DIM)[None, None, :]  # [1, 1, HEAD_DIM]

    # ========== 2. Load (only two loads!) ==========
    # Clip offsets to valid range [0, N-1] to avoid mask load
    offs_even_clipped = tl.minimum(offs_even, N - 1)
    offs_odd_clipped = tl.minimum(offs_odd, N - 1)

    # ptr shape: [NUM_SUB, HALF_BS, HEAD_DIM]
    ptr_even = k_base + offs_even_clipped[:, :, None] * stride_kn + offs_d * stride_kd
    ptr_odd = k_base + offs_odd_clipped[:, :, None] * stride_kn + offs_d * stride_kd

    val_even = tl.load(ptr_even)  # [NUM_SUB, HALF_BS, HEAD_DIM]
    val_odd = tl.load(ptr_odd)    # [NUM_SUB, HALF_BS, HEAD_DIM]

    # ========== 3. Precompute norms ==========
    # [NUM_SUB, HALF_BS]
    norm_even = tl.sqrt(tl.sum(val_even * val_even, axis=2) + 1e-12)
    norm_odd = tl.sqrt(tl.sum(val_odd * val_odd, axis=2) + 1e-12)

    # ========== 4. Sim 2x: directly compute ==========
    dot_2 = tl.sum(val_even * val_odd, axis=2)  # [NUM_SUB, HALF_BS]
    cos_2 = dot_2 / (norm_even * norm_odd + 1e-6)
    sim_2 = tl.sum(cos_2, axis=1) / HALF_BS  # [NUM_SUB]

    # ========== 5. Sim 4x: use split ==========
    QUARTER_BS: tl.constexpr = USER_BLOCK_SIZE // 4  # e.g. 8

    # Reshape val_even: [NUM_SUB, HALF_BS, HEAD_DIM] -> [NUM_SUB, QUARTER_BS, 2, HEAD_DIM]
    val_even_re = tl.reshape(val_even, (NUM_SUB, QUARTER_BS, 2, HEAD_DIM))
    val_even_perm = tl.permute(val_even_re, (0, 1, 3, 2))
    k_0, k_2 = tl.split(val_even_perm)  # each: [NUM_SUB, QUARTER_BS, HEAD_DIM]

    val_odd_re = tl.reshape(val_odd, (NUM_SUB, QUARTER_BS, 2, HEAD_DIM))
    val_odd_perm = tl.permute(val_odd_re, (0, 1, 3, 2))
    k_1, k_3 = tl.split(val_odd_perm)  # each: [NUM_SUB, QUARTER_BS, HEAD_DIM]

    # Reshape norms: [NUM_SUB, HALF_BS] -> [NUM_SUB, QUARTER_BS, 2]
    norm_even_re = tl.reshape(norm_even, (NUM_SUB, QUARTER_BS, 2))
    n_0, n_2 = tl.split(norm_even_re)  # each: [NUM_SUB, QUARTER_BS]

    norm_odd_re = tl.reshape(norm_odd, (NUM_SUB, QUARTER_BS, 2))
    n_1, n_3 = tl.split(norm_odd_re)  # each: [NUM_SUB, QUARTER_BS]

    # Compute Sim 4x: k_0 (0, 4, ...) vs k_3 (3, 7, ...)
    dot_4 = tl.sum(k_0 * k_3, axis=2)  # [NUM_SUB, QUARTER_BS]
    cos_4 = dot_4 / (n_0 * n_3 + 1e-6)
    sim_4 = tl.sum(cos_4, axis=1) / QUARTER_BS  # [NUM_SUB]

    # ========== 6. Sim 8x: split again ==========
    EIGHTH_BS: tl.constexpr = USER_BLOCK_SIZE // 8  # e.g. 4

    k_0_re = tl.reshape(k_0, (NUM_SUB, EIGHTH_BS, 2, HEAD_DIM))
    k_0_perm = tl.permute(k_0_re, (0, 1, 3, 2))
    k_00, k_04 = tl.split(k_0_perm)  # each: [NUM_SUB, EIGHTH_BS, HEAD_DIM]

    k_3_re = tl.reshape(k_3, (NUM_SUB, EIGHTH_BS, 2, HEAD_DIM))
    k_3_perm = tl.permute(k_3_re, (0, 1, 3, 2))
    k_31, k_33 = tl.split(k_3_perm)  # each: [NUM_SUB, EIGHTH_BS, HEAD_DIM]

    # Norms
    n_0_re = tl.reshape(n_0, (NUM_SUB, EIGHTH_BS, 2))
    n_00, n_04 = tl.split(n_0_re)  # each: [NUM_SUB, EIGHTH_BS]

    n_3_re = tl.reshape(n_3, (NUM_SUB, EIGHTH_BS, 2))
    n_31, n_33 = tl.split(n_3_re)  # each: [NUM_SUB, EIGHTH_BS]

    # Compute Sim 8x: k_00 (0, 8, ...) vs k_33 (7, 15, ...)
    dot_8 = tl.sum(k_00 * k_33, axis=2)  # [NUM_SUB, EIGHTH_BS]
    cos_8 = dot_8 / (n_00 * n_33 + 1e-6)
    sim_8 = tl.sum(cos_8, axis=1) / EIGHTH_BS  # [NUM_SUB]

    # ========== 7. Apply thresholds and compute final score ==========
    final_score = tl.full((NUM_SUB,), 1.0, dtype=tl.float32)
    final_score = tl.where(sim_2 > t2, 2.0, final_score)
    final_score = tl.where(sim_4 > t4, 4.0, final_score)
    final_score = tl.where(sim_8 > t8, 8.0, final_score)

    # ========== 8. Store results ==========
    out_idx = pid_chunk * NUM_SUB + tl.arange(0, NUM_SUB)
    num_blocks = N // USER_BLOCK_SIZE
    out_mask = out_idx < num_blocks

    out_ptr = Out_ptr + pid_bh * stride_oh + out_idx * stride_on
    tl.store(out_ptr, final_score.to(tl.int32), mask=out_mask)


def calc_k_similarity_triton(k: paddle.Tensor, blocksize: int, config) -> paddle.Tensor:
    """
    Triton implementation of K similarity calculation using split optimization.

    This kernel computes cosine similarity between K vectors at different strides
    to determine the optimal pooling level for each block:
    - 2x: similarity between adjacent pairs (0,1), (2,3), ...
    - 4x: similarity between (0,3), (4,7), ...
    - 8x: similarity between (0,7), (8,15), ...

    Args:
        k: Key tensor of shape [B, H, L, D] - PaddlePaddle tensor
        blocksize: Block size (must be divisible by 8)
        config: Configuration object with sim_2x_threshold, sim_4x_threshold, sim_8x_threshold

    Returns:
        Similarity mask of shape [B, H, num_blocks] with values in {1, 2, 4, 8}
    """
    assert blocksize % 8 == 0, "blocksize must be divisible by 8"
    assert blocksize >= 8, "blocksize must be at least 8"

    B, H, L, D = k.shape
    num_blocks = (L + blocksize - 1) // blocksize

    # Ensure contiguous float32
    if not k.is_contiguous():
        k = paddle.to_tensor(k, place=k.place)
    if k.dtype != paddle.float32:
        k = k.astype('float32')

    # Output tensor
    out = paddle.empty(shape=[B, H, num_blocks], dtype='int32')

    # Thresholds
    t2 = float(getattr(config, 'sim_2x_threshold', 0.0))
    t4 = float(getattr(config, 'sim_4x_threshold', 0.0))
    t8 = float(getattr(config, 'sim_8x_threshold', -1.0))

    # Triton block size (how many K vectors per program)
    TRITON_BLOCK = max(128, blocksize)
    if TRITON_BLOCK % blocksize != 0:
        TRITON_BLOCK = ((TRITON_BLOCK + blocksize - 1) // blocksize) * blocksize

    # Number of chunks needed
    N_effective = num_blocks * blocksize
    num_chunks = (N_effective + TRITON_BLOCK - 1) // TRITON_BLOCK

    # Grid: (num_chunks, B * H)
    grid = (num_chunks, B * H)

    # Strides
    stride_kb, stride_kh, stride_kn, stride_kd = k.strides
    stride_ob, stride_oh, stride_on = out.strides

    _calc_k_similarity_split_kernel[grid](
        k, out,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_ob, stride_oh, stride_on,
        t2, t4, t8,
        N_effective,
        USER_BLOCK_SIZE=blocksize,
        HEAD_DIM=D,
        TRITON_BLOCK=TRITON_BLOCK,
    )

    return out


# ============ PSA Forward Kernels ============

@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    pooling_bias: tl.constexpr,  # log(pooling_level)
    POOLING_RATIO: tl.constexpr,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for inner_idx in range(POOLING_BLOCK_N // BLOCK_N):
        start_n = pooling_block_idx * POOLING_BLOCK_N + inner_idx * BLOCK_N

        valid_k = offs_n[None, :] + start_n < seqlen_k
        k = tl.load(
            k_ptrs + start_n * stride_kt,
            mask=valid_k,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk += pooling_bias

        if LAST_K_BLOCK:
            qk += tl.where(valid_k, 0.0, -float("inf"))
        if CAUSAL:
            key_upper = (start_n + offs_n + 1) * POOLING_RATIO - 1
            causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
            qk = tl.where(causal_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk * RCP_LN2)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k
        )
        p = p.to(v.type.element_ty)
        acc += tl.dot(p, v).to(tl.float32)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_1(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale

    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_positions = start_n + offs_n
        causal_mask = key_positions[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
    p = p.to(v.type.element_ty)
    acc += tl.dot(p, v).to(tl.float32)
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_2(
    acc,
    l_i,
    m_i,
    q,
    pooling_block_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = pooling_block_idx * BLOCK_N

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale
    qk += 0.6931471805599453094  # log(2)

    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 2 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
    p = p.to(v.type.element_ty)
    acc += tl.dot(p, v).to(tl.float32)
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_4(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale
    qk += 1.3862943611198906188  # log(4)

    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 4 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
    p = p.to(v.type.element_ty)
    acc += tl.dot(p, v).to(tl.float32)
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_inner_8(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    v_ptrs,
    offs_m_idx,
    offs_n,
    stride_kt,
    stride_vt,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_n = k_block_col_idx * BLOCK_N

    valid_k = offs_n[None, :] + start_n < seqlen_k
    k = tl.load(
        k_ptrs + start_n * stride_kt,
        mask=valid_k,
        other=0.0,
    )

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= sm_scale
    qk += 2.0794415416798359283  # log(8)

    if LAST_K_BLOCK:
        qk += tl.where(valid_k, 0.0, -float("inf"))
    if CAUSAL:
        key_upper = (start_n + offs_n + 1) * 8 - 1
        causal_mask = key_upper[None, :] <= offs_m_idx[:, None]
        qk = tl.where(causal_mask, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk * RCP_LN2)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
    p = p.to(v.type.element_ty)
    acc += tl.dot(p, v).to(tl.float32)
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q,
    K,
    K_2,
    K_4,
    K_8,
    K_16,
    V,
    V_2,
    V_4,
    V_8,
    V_16,
    sm_scale,
    block_mask_ptr,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kz_2,
    stride_kh_2,
    stride_kn_2,
    stride_kd_2,
    stride_kz_4,
    stride_kh_4,
    stride_kn_4,
    stride_kd_4,
    stride_kz_8,
    stride_kh_8,
    stride_kn_8,
    stride_kd_8,
    stride_kz_16,
    stride_kh_16,
    stride_kn_16,
    stride_kd_16,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_vz_2,
    stride_vh_2,
    stride_vn_2,
    stride_vd_2,
    stride_vz_4,
    stride_vh_4,
    stride_vn_4,
    stride_vd_4,
    stride_vz_8,
    stride_vh_8,
    stride_vn_8,
    stride_vd_8,
    stride_vz_16,
    stride_vh_16,
    stride_vn_16,
    stride_vd_16,
    stride_bmz,
    stride_bmh,
    stride_bmm,
    stride_bmn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    N_CTX_2,
    N_CTX_4,
    N_CTX_8,
    N_CTX_16,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N: tl.constexpr,
    POOLING_BLOCK_N_2: tl.constexpr,
    POOLING_BLOCK_N_4: tl.constexpr,
    POOLING_BLOCK_N_8: tl.constexpr,
    POOLING_BLOCK_N_16: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    LOG_1 = 0.0
    LOG_2 = 0.6931471805599453094  # log(2)
    LOG_4 = 1.3862943611198906188  # log(4)
    LOG_8 = 2.0794415416798359283  # log(8)

    Q_LEN = N_CTX
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    K_2 += off_z * stride_kz_2 + off_h * stride_kh_2
    K_4 += off_z * stride_kz_4 + off_h * stride_kh_4
    K_8 += off_z * stride_kz_8 + off_h * stride_kh_8
    K_16 += off_z * stride_kz_16 + off_h * stride_kh_16
    V += off_z * stride_vz + off_h * stride_vh
    V_2 += off_z * stride_vz_2 + off_h * stride_vh_2
    V_4 += off_z * stride_vz_4 + off_h * stride_vh_4
    V_8 += off_z * stride_vz_8 + off_h * stride_vh_8
    V_16 += off_z * stride_vz_16 + off_h * stride_vh_16
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n_1 = tl.arange(0, POOLING_BLOCK_N)
    offs_n_2 = tl.arange(0, POOLING_BLOCK_N_2)
    offs_n_4 = tl.arange(0, POOLING_BLOCK_N_4)
    offs_n_8 = tl.arange(0, POOLING_BLOCK_N_8)
    offs_n_16 = tl.arange(0, POOLING_BLOCK_N_16)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    if POOLING_BLOCK_N < BLOCK_N:
        off_k_1 = offs_n_1[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n_1[:, None] * stride_vn + offs_d[None, :] * stride_vd
    else:
        off_k_1 = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        off_v_1 = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    if POOLING_BLOCK_N_2 < BLOCK_N:
        off_k_2 = offs_n_2[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n_2[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    else:
        off_k_2 = offs_n[None, :] * stride_kn_2 + offs_d[:, None] * stride_kd_2
        off_v_2 = offs_n[:, None] * stride_vn_2 + offs_d[None, :] * stride_vd_2
    if POOLING_BLOCK_N_4 < BLOCK_N:
        off_k_4 = offs_n_4[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n_4[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    else:
        off_k_4 = offs_n[None, :] * stride_kn_4 + offs_d[:, None] * stride_kd_4
        off_v_4 = offs_n[:, None] * stride_vn_4 + offs_d[None, :] * stride_vd_4
    if POOLING_BLOCK_N_8 < BLOCK_N:
        off_k_8 = offs_n_8[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n_8[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8
    else:
        off_k_8 = offs_n[None, :] * stride_kn_8 + offs_d[:, None] * stride_kd_8
        off_v_8 = offs_n[:, None] * stride_vn_8 + offs_d[None, :] * stride_vd_8

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs_1 = K + off_k_1
    k_ptrs_2 = K_2 + off_k_2
    k_ptrs_4 = K_4 + off_k_4
    k_ptrs_8 = K_8 + off_k_8
    v_ptrs_1 = V + off_v_1
    v_ptrs_2 = V_2 + off_v_2
    v_ptrs_4 = V_4 + off_v_4
    v_ptrs_8 = V_8 + off_v_8
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    pooling_block_start = 0
    pooling_block_end = tl.cdiv(N_CTX, POOLING_BLOCK_N)

    # For causal attention, only process K blocks up to current Q position
    if CAUSAL:
        causal_block_end = tl.cdiv((start_m + 1) * BLOCK_M, POOLING_BLOCK_N)
        pooling_block_end = tl.minimum(pooling_block_end, causal_block_end)

    # loop over k, v and update accumulator
    for pb_idx in range(pooling_block_start, pooling_block_end):
        mask = tl.load(mask_ptrs + pb_idx * stride_bmn)
        is_last_block = pb_idx == pooling_block_end - 1

        if mask > 0:
            if mask < 4:
                if mask == 1:
                    if POOLING_BLOCK_N <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_1(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_1, v_ptrs_1, offs_m, offs_n_1,
                            stride_kn, stride_vn, sm_scale, N_CTX,
                            is_last_block, CAUSAL, BLOCK_M, POOLING_BLOCK_N,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_1, v_ptrs_1, offs_m, offs_n,
                            stride_kn, stride_vn, sm_scale, N_CTX,
                            LOG_1, 1, is_last_block, CAUSAL,
                            BLOCK_M, BLOCK_N, POOLING_BLOCK_N,
                        )
                if mask == 2:
                    if POOLING_BLOCK_N_2 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_2(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_2, v_ptrs_2, offs_m, offs_n_2,
                            stride_kn_2, stride_vn_2, sm_scale, N_CTX_2,
                            is_last_block, CAUSAL, BLOCK_M, POOLING_BLOCK_N_2,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_2, v_ptrs_2, offs_m, offs_n,
                            stride_kn_2, stride_vn_2, sm_scale, N_CTX_2,
                            LOG_2, 2, is_last_block, CAUSAL,
                            BLOCK_M, BLOCK_N, POOLING_BLOCK_N_2,
                        )
            else:
                if mask == 8:
                    if POOLING_BLOCK_N_8 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_8(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_8, v_ptrs_8, offs_m, offs_n_8,
                            stride_kn_8, stride_vn_8, sm_scale, N_CTX_8,
                            is_last_block, CAUSAL, BLOCK_M, POOLING_BLOCK_N_8,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_8, v_ptrs_8, offs_m, offs_n,
                            stride_kn_8, stride_vn_8, sm_scale, N_CTX_8,
                            LOG_8, 8, is_last_block, CAUSAL,
                            BLOCK_M, BLOCK_N, POOLING_BLOCK_N_8,
                        )
                if mask == 4:
                    if POOLING_BLOCK_N_4 <= BLOCK_N:
                        acc, l_i, m_i = _fwd_kernel_inner_4(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_4, v_ptrs_4, offs_m, offs_n_4,
                            stride_kn_4, stride_vn_4, sm_scale, N_CTX_4,
                            is_last_block, CAUSAL, BLOCK_M, POOLING_BLOCK_N_4,
                        )
                    else:
                        acc, l_i, m_i = _fwd_kernel_inner(
                            acc, l_i, m_i, q, pb_idx,
                            k_ptrs_4, v_ptrs_4, offs_m, offs_n,
                            stride_kn_4, stride_vn_4, sm_scale, N_CTX_4,
                            LOG_4, 4, is_last_block, CAUSAL,
                            BLOCK_M, BLOCK_N, POOLING_BLOCK_N_4,
                        )

    # Add epsilon for numerical stability
    EPS: tl.constexpr = 1e-8
    l_i_safe = tl.maximum(l_i, EPS)

    m_i += tl.math.log(l_i_safe)
    l_recip = 1 / l_i_safe[:, None]
    acc = acc * l_recip

    acc = acc.to(Out.dtype.element_ty)

    off_o = (
        off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q_LEN)


# ============ Helper Functions ============

def pad_to_multiple(x, multiple):
    """
    Pad sequence dimension (dim=2) to make it a multiple of `multiple`.
    x: [B, H, L, D] - PaddlePaddle tensor
    """
    L = x.shape[2]
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        x = F.pad(x, [0, 0, 0, pad_len], mode='replicate')
    return x


def pooling(x, zoom_ratio):
    """
    Pooling operation using average pooling.
    x: [B, H, L, D] - PaddlePaddle tensor
    zoom_ratio: pooling ratio
    """
    B, H, L, D = x.shape

    remainder = L % zoom_ratio
    if remainder != 0:
        pad_len = zoom_ratio - remainder
        x = F.pad(x, [0, 0, 0, pad_len], mode='replicate')
        L = x.shape[2]

    x = paddle.mean(x.reshape([B, H, -1, zoom_ratio, D]), axis=3)
    return x


def _forward(
    q,
    k,
    v,
    block_sparse_mask,
    sm_scale,
    BLOCK_M=64,
    BLOCK_N=64,
    POOLING_BLOCK_N=128,
    num_warps=None,
    num_stages=1,
    out=None,
    causal=False,
):
    """Forward pass for block sparse attention (PaddlePaddle version)"""

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]

    o = out if out is not None else paddle.empty_like(q)

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    H = q.shape[1]

    k_padding = pad_to_multiple(k, POOLING_BLOCK_N)
    v_padding = pad_to_multiple(v, POOLING_BLOCK_N)
    k_2 = pooling(k_padding, 2)
    v_2 = pooling(v_padding, 2)
    k_4 = pooling(k_2, 2)
    v_4 = pooling(v_2, 2)
    k_8 = pooling(k_4, 2)
    v_8 = pooling(v_4, 2)
    k_16 = pooling(k_8, 2)
    v_16 = pooling(v_8, 2)

    N_CTX = k.shape[2]
    N_CTX_2 = k_2.shape[2]
    N_CTX_4 = k_4.shape[2]
    N_CTX_8 = k_8.shape[2]
    N_CTX_16 = k_16.shape[2]

    _fwd_kernel[grid](
        q, k, k_2, k_4, k_8, k_16, v, v_2, v_4, v_8, v_16, sm_scale,
        block_sparse_mask,
        o,
        q.strides[0], q.strides[1], q.strides[2], q.strides[3],
        k.strides[0], k.strides[1], k.strides[2], k.strides[3],
        k_2.strides[0], k_2.strides[1], k_2.strides[2], k_2.strides[3],
        k_4.strides[0], k_4.strides[1], k_4.strides[2], k_4.strides[3],
        k_8.strides[0], k_8.strides[1], k_8.strides[2], k_8.strides[3],
        k_16.strides[0], k_16.strides[1], k_16.strides[2], k_16.strides[3],
        v.strides[0], v.strides[1], v.strides[2], v.strides[3],
        v_2.strides[0], v_2.strides[1], v_2.strides[2], v_2.strides[3],
        v_4.strides[0], v_4.strides[1], v_4.strides[2], v_4.strides[3],
        v_8.strides[0], v_8.strides[1], v_8.strides[2], v_8.strides[3],
        v_16.strides[0], v_16.strides[1], v_16.strides[2], v_16.strides[3],
        block_sparse_mask.strides[0], block_sparse_mask.strides[1],
        block_sparse_mask.strides[2], block_sparse_mask.strides[3],
        o.strides[0], o.strides[1], o.strides[2], o.strides[3],
        H, N_CTX, N_CTX_2, N_CTX_4, N_CTX_8, N_CTX_16,
        causal,
        BLOCK_M,
        BLOCK_N,
        POOLING_BLOCK_N,
        POOLING_BLOCK_N // 2,
        POOLING_BLOCK_N // 4,
        POOLING_BLOCK_N // 8,
        POOLING_BLOCK_N // 16,
        BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o


def sparse_attention_factory(BLOCK_M=128, BLOCK_N=32, POOLING_BLOCK_N=128, causal=False, **kwargs):
    """
    Factory function to create sparse attention function with specific configuration.
    Returns a callable that performs sparse attention.
    """
    def sparse_attention_fn(q, k, v, block_sparse_dense, sm_scale=None):
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        return _forward(
            q,
            k,
            v,
            block_sparse_dense,
            sm_scale,
            BLOCK_M,
            BLOCK_N,
            POOLING_BLOCK_N,
            causal=causal,
            **kwargs,
        )
    return sparse_attention_fn


# For direct use
block_sparse_triton_fn = sparse_attention_factory()
