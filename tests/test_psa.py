"""
Tests for PSA Paddle Implementation
Tests the block sparse attention kernel and the full PSA module.
"""

import triton
import paddle
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psa_paddle import AttentionConfig, PyramidAdaptiveBlockSparseAttnTrain
from psa_paddle.kernels.psa_kernel_causal import sparse_attention_factory


def test_sparse_attention_kernel():
    """Test the block sparse attention kernel with simple mask."""
    print("=" * 60)
    print("Testing Block Sparse Attention Kernel")
    print("=" * 60)

    # Test parameters
    BATCH_SIZE = 1
    NUM_HEADS = 4
    SEQ_LEN = 512
    HEAD_DIM = 64
    BLOCK_SIZE = 128

    # Create random input tensors
    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    # Create a simple mask - all ones means full attention
    num_blocks = SEQ_LEN // BLOCK_SIZE
    mask = paddle.ones([BATCH_SIZE, NUM_HEADS, num_blocks, num_blocks], dtype='int32')

    # Create sparse attention function
    sparse_attn_fn = sparse_attention_factory(
        BLOCK_M=128,
        BLOCK_N=64,
        POOLING_BLOCK_N=128,
        causal=False
    )

    # Run sparse attention
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Mask shape: {mask.shape}")

    try:
        out = sparse_attn_fn(q, k, v, mask)
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")
        print(f"Output has NaN: {paddle.isnan(out).any().item()}")
        print(f"Output has Inf: {paddle.isinf(out).any().item()}")
        print(f"Output mean: {out.mean().item():.6f}")
        print(f"Output std: {out.std().item():.6f}")
        print("Sparse attention kernel test PASSED")
        return True
    except Exception as e:
        print(f"Sparse attention kernel test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparse_attention_causal():
    """Test the block sparse attention kernel with causal mask."""
    print("\n" + "=" * 60)
    print("Testing Block Sparse Attention Kernel (Causal)")
    print("=" * 60)

    BATCH_SIZE = 1
    NUM_HEADS = 4
    SEQ_LEN = 512
    HEAD_DIM = 64
    BLOCK_SIZE = 128

    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    num_blocks = SEQ_LEN // BLOCK_SIZE
    # Create causal mask - lower triangular
    mask = paddle.tril(paddle.ones([num_blocks, num_blocks], dtype='int32'))
    mask = mask.unsqueeze(0).unsqueeze(0).expand([BATCH_SIZE, NUM_HEADS, num_blocks, num_blocks])

    sparse_attn_fn = sparse_attention_factory(
        BLOCK_M=128,
        BLOCK_N=32,
        POOLING_BLOCK_N=128,
        causal=True
    )

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Mask shape: {mask.shape}")

    try:
        out = sparse_attn_fn(q, k, v, mask)
        print(f"Output shape: {out.shape}")
        print(f"Output has NaN: {paddle.isnan(out).any().item()}")
        print(f"Output has Inf: {paddle.isinf(out).any().item()}")
        print(f"Output mean: {out.mean().item():.6f}")
        print("Causal sparse attention test PASSED")
        return True
    except Exception as e:
        print(f"Causal sparse attention test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_scale_pooling():
    """Test sparse attention with multi-scale pooling masks (1, 2, 4, 8)."""
    print("\n" + "=" * 60)
    print("Testing Multi-Scale Pooling Masks")
    print("=" * 60)

    BATCH_SIZE = 1
    NUM_HEADS = 4
    SEQ_LEN = 1024
    HEAD_DIM = 64
    BLOCK_SIZE = 128

    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    num_blocks = SEQ_LEN // BLOCK_SIZE

    # Create mask with different pooling levels
    mask = paddle.zeros([BATCH_SIZE, NUM_HEADS, num_blocks, num_blocks], dtype='int32')

    # First block row: full attention (1)
    mask[:, :, 0, :] = 1
    # Second block row: 2x pooling
    mask[:, :, 1, :] = 2
    # Third block row: 4x pooling
    mask[:, :, 2, :] = 4
    # Rest: 8x pooling
    mask[:, :, 3:, :] = 8

    sparse_attn_fn = sparse_attention_factory(
        BLOCK_M=128,
        BLOCK_N=32,
        POOLING_BLOCK_N=128,
        causal=False
    )

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Mask values: 1={int((mask==1).sum())}, 2={int((mask==2).sum())}, 4={int((mask==4).sum())}, 8={int((mask==8).sum())}")

    try:
        out = sparse_attn_fn(q, k, v, mask)
        print(f"Output shape: {out.shape}")
        print(f"Output has NaN: {paddle.isnan(out).any().item()}")
        print(f"Output has Inf: {paddle.isinf(out).any().item()}")
        print(f"Output mean: {out.mean().item():.6f}")
        print("Multi-scale pooling test PASSED")
        return True
    except Exception as e:
        print(f"Multi-scale pooling test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_psa_module():
    """Test the full PyramidAdaptiveBlockSparseAttnTrain module."""
    print("\n" + "=" * 60)
    print("Testing Full PSA Module")
    print("=" * 60)

    BATCH_SIZE = 1
    NUM_HEADS = 4
    SEQ_LEN = 1024
    HEAD_DIM = 64

    config = AttentionConfig(
        query_block=128,
        mask_mode="topk",
        mask_ratios={
            1: (0.0, 0.15),
            2: (0.15, 0.30),
            4: (0.30, 0.50),
            8: (0.50, 1.0),
        },
        importance_method="xattn",
        xattn_stride=16,
        causal_main=True,
        warmup_steps=0,
    )

    print(f"Config: mask_mode={config.mask_mode}, importance_method={config.importance_method}")

    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    try:
        psa_module = PyramidAdaptiveBlockSparseAttnTrain(config=config, layer_idx=0)
        out = psa_module(q, k, v)
        print(f"Output shape: {out.shape}")
        print(f"Output has NaN: {paddle.isnan(out).any().item()}")
        print(f"Output has Inf: {paddle.isinf(out).any().item()}")
        print(f"Output mean: {out.mean().item():.6f}")
        print(f"Sparsity accumulated: {psa_module.sparsity_acc:.4f}")
        print("PSA module test PASSED")
        return True
    except Exception as e:
        print(f"PSA module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Benchmark the sparse attention kernel."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    BATCH_SIZE = 1
    NUM_HEADS = 8
    SEQ_LEN = 4096
    HEAD_DIM = 128
    BLOCK_SIZE = 128

    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    num_blocks = SEQ_LEN // BLOCK_SIZE
    mask = paddle.ones([BATCH_SIZE, NUM_HEADS, num_blocks, num_blocks], dtype='int32')

    sparse_attn_fn = sparse_attention_factory(
        BLOCK_M=128,
        BLOCK_N=32,
        POOLING_BLOCK_N=128,
        causal=True
    )

    print(f"Benchmark config: B={BATCH_SIZE}, H={NUM_HEADS}, L={SEQ_LEN}, D={HEAD_DIM}")

    # Warmup
    for _ in range(3):
        out = sparse_attn_fn(q, k, v, mask)
    paddle.device.cuda.synchronize()

    # Benchmark
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        out = sparse_attn_fn(q, k, v, mask)
    paddle.device.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average time per forward: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} forward/s")
    print("Performance benchmark completed")
    return True


def compare_with_reference():
    """Compare sparse attention output with reference scaled dot product attention."""
    print("\n" + "=" * 60)
    print("Comparing with Reference Attention")
    print("=" * 60)

    BATCH_SIZE = 1
    NUM_HEADS = 4
    SEQ_LEN = 256
    HEAD_DIM = 64
    BLOCK_SIZE = 128

    paddle.seed(42)
    q = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    k = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')
    v = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]).astype('float16')

    # Full attention mask
    num_blocks = SEQ_LEN // BLOCK_SIZE
    mask = paddle.ones([BATCH_SIZE, NUM_HEADS, num_blocks, num_blocks], dtype='int32')

    sparse_attn_fn = sparse_attention_factory(
        BLOCK_M=128,
        BLOCK_N=32,
        POOLING_BLOCK_N=128,
        causal=False
    )

    # Sparse attention output
    sparse_out = sparse_attn_fn(q, k, v, mask)

    # Reference attention (standard scaled dot-product)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    attn_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
    attn_weights = paddle.nn.functional.softmax(attn_weights.astype('float32'), axis=-1).astype('float16')
    ref_out = paddle.matmul(attn_weights, v)

    # Compare
    diff = paddle.abs(sparse_out - ref_out)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Sparse output shape: {sparse_out.shape}")
    print(f"Reference output shape: {ref_out.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Check if difference is acceptable (considering fp16 precision)
    if max_diff < 0.1 and mean_diff < 0.01:
        print("Reference comparison PASSED (within tolerance)")
        return True
    else:
        print("Reference comparison FAILED (difference too large)")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PSA Paddle Implementation Tests")
    print("=" * 60)

    # Check CUDA availability
    if not paddle.device.is_compiled_with_cuda():
        print("ERROR: CUDA is not available. Tests require GPU.")
        return 1

    print(f"Using device: {paddle.device.cuda.get_device_name()}")
    print(f"PaddlePaddle version: {paddle.__version__}")

    results = []

    # Run tests
    results.append(("Sparse Attention Kernel", test_sparse_attention_kernel()))
    results.append(("Causal Sparse Attention", test_sparse_attention_causal()))
    results.append(("Multi-Scale Pooling", test_multi_scale_pooling()))
    results.append(("PSA Module", test_psa_module()))
    results.append(("Reference Comparison", compare_with_reference()))
    results.append(("Performance Benchmark", test_performance()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
