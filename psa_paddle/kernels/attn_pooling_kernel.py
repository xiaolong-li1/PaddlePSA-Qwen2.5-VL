"""
Attention Pooling Kernel for PaddlePaddle
Based on the PyTorch Triton implementation, adapted for PaddlePaddle.
"""

import triton
import triton.language as tl
import paddle


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner_optimized(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    R_block_ptr,
    A_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * TILE_M
    elif STAGE == 2:
        lo, hi = start_m * TILE_M, (start_m + 1) * TILE_M
        lo = tl.multiple_of(lo, TILE_M)
    else:
        lo, hi = 0, N_CTX
        
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    
    for start_n in range(lo, hi, TILE_N):
        start_n = tl.multiple_of(start_n, TILE_N)
        is_last_block = start_n + TILE_N >= hi
        remaining = hi - start_n
        mask = tl.arange(0, TILE_N) < remaining
        k = tl.load(K_block_ptr)

        qk = tl.dot(q, k)
        if is_last_block:
            qk = tl.where(mask, qk, -float("inf"))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)
            
        # Compute block row max
        blocked_qk = tl.reshape(qk, (TILE_M, TILE_N // BLOCK_N, BLOCK_N))
        block_row_max = (
            tl.max(blocked_qk, axis=2) * qk_scale
        )
        max_val = tl.max(block_row_max, axis=1)
        m_ij = tl.maximum(m_i, max_val)
        qk = qk * qk_scale - m_ij[:, None]
        tl.store(
            tl.advance(R_block_ptr, (0, start_n // BLOCK_N)), block_row_max.to(q.dtype)
        )
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, TILE_N))

    # Update pooling output
    if STAGE == 2:
        for start_n in range(0, (start_m + 1) * BLOCK_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi) / l_i[:, None]
            blocked_row_max = tl.reshape(
                row_max, (TILE_M // BLOCK_M, BLOCK_M, TILE_N // BLOCK_N)
            )
            col_max = tl.max(blocked_row_max, axis=1)
            col_max = col_max.to(q.dtype)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, TILE_N // BLOCK_N))
            R_block_ptr = tl.advance(R_block_ptr, (0, TILE_N // BLOCK_N))

    elif STAGE == 3:
        for start_n in range(lo, hi, TILE_N):
            start_n = tl.multiple_of(start_n, TILE_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi) / l_i[:, None]
            blocked_row_max = tl.reshape(
                row_max, (TILE_M // BLOCK_M, BLOCK_M, TILE_N // BLOCK_N)
            )
            col_max = tl.max(blocked_row_max, axis=1)
            col_max = col_max.to(q.dtype)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, TILE_N // BLOCK_N))
            R_block_ptr = tl.advance(R_block_ptr, (0, TILE_N // BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_optimized(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    R,
    Po,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,
    stride_poz,
    stride_poh,
    stride_pom,
    stride_pon,
    Z,
    H,
    N_CTX,
    n_rep,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    N_DOWNSAMPLE: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kvh = off_h // n_rep
    
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kvh.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_kvh.to(tl.int64) * stride_vh
    r_offset = off_z.to(tl.int64) * stride_rz + off_h.to(tl.int64) * stride_rh
    po_offset = off_z.to(tl.int64) * stride_poz + off_h.to(tl.int64) * stride_poh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * TILE_M, 0),
        block_shape=(TILE_M, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_N),
        order=(0, 1),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * TILE_M, 0),
        block_shape=(TILE_M, HEAD_DIM),
        order=(1, 0),
    )

    R_block_ptr = tl.make_block_ptr(
        base=R + r_offset,
        shape=(N_CTX, N_DOWNSAMPLE),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * TILE_M, 0),
        block_shape=(TILE_M, TILE_N // BLOCK_N),
        order=(0, 1),
    )
    
    A_block_ptr = tl.make_block_ptr(
        base=Po + po_offset,
        shape=(N_DOWNSAMPLE, N_DOWNSAMPLE),
        strides=(stride_pom, stride_pon),
        offsets=(start_m * (TILE_M // BLOCK_M), 0),
        block_shape=(TILE_M // BLOCK_M, TILE_N // BLOCK_N),
        order=(0, 1),
    )
    
    # initialize offsets
    offs_m = start_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, TILE_N)
    
    # initialize pointer to m and l
    m_i = tl.zeros([TILE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([TILE_M, HEAD_DIM], dtype=tl.float32)
    
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    
    # load q
    q = tl.load(Q_block_ptr)
    
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_optimized(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            None,
            R_block_ptr,
            A_block_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            TILE_M,
            TILE_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
        
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_optimized(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            None,
            R_block_ptr,
            A_block_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            TILE_M,
            TILE_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
        
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def attn_with_pooling(q, k, v, causal, sm_scale, block_size, tile_m=64, tile_n=64):
    """
    Attention with pooling for PaddlePaddle tensors.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]  
        v: Value tensor [B, H, L, D]
        causal: Whether to use causal attention
        sm_scale: Scale factor for attention
        block_size: Block size for computation
        tile_m: Tile size for M dimension (default: 64)
        tile_n: Tile size for N dimension (default: 64)
        
    Returns:
        Tuple of (output, pooling_map)
    """
    assert block_size in {16, 32, 64, 128}
    assert tile_n % block_size == 0 and tile_m % block_size == 0
    assert tile_m > 0 and tile_n > 0
    assert tile_m <= 256 and tile_n <= 256
    
    orig_dtype = q.dtype
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    
    NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
    assert NUM_HEADS_K == NUM_HEADS_V
    n_rep = NUM_HEADS_Q // NUM_HEADS_K
    
    o = paddle.empty_like(q)
    BLOCK_N = block_size
    n_d = triton.cdiv(q.shape[2], BLOCK_N)
    
    R = paddle.full(
        shape=[q.shape[0], q.shape[1], q.shape[2], n_d],
        fill_value=-65504.0,
        dtype=q.dtype,
    )
    Po = paddle.zeros(
        shape=[q.shape[0], q.shape[1], n_d, n_d], 
        dtype=q.dtype
    )
    
    stage = 3 if causal else 1
    extra_kern_args = {}
    
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    grid = lambda args: (
        triton.cdiv(q.shape[2], tile_m),
        q.shape[0] * q.shape[1],
        1,
    )
    
    M = paddle.empty(
        shape=[q.shape[0], q.shape[1], q.shape[2]], 
        dtype='float32'
    )
    
    _attn_fwd_optimized[grid](
        q,
        k,
        v,
        sm_scale,
        M,
        o,
        R,
        Po,
        q.strides[0], q.strides[1], q.strides[2], q.strides[3],
        k.strides[0], k.strides[1], k.strides[2], k.strides[3],
        v.strides[0], v.strides[1], v.strides[2], v.strides[3],
        o.strides[0], o.strides[1], o.strides[2], o.strides[3],
        R.strides[0], R.strides[1], R.strides[2], R.strides[3],
        Po.strides[0], Po.strides[1], Po.strides[2], Po.strides[3],
        q.shape[0],
        q.shape[1],
        N_CTX=q.shape[2],
        n_rep=n_rep,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        TILE_M=tile_m,
        TILE_N=tile_n,
        N_DOWNSAMPLE=n_d,
        num_stages=3,
        num_warps=4,
        **extra_kern_args
    )
    
    Sum = paddle.sum(Po, axis=-1, keepdim=True)
    Po = Po / Sum
    o = o.astype(orig_dtype)
    
    return o, Po
