---
title: Triton 入门-Softmax-与 Torch 对比 Benchmark
tags: 学习笔记与作业
---

学习一下 triton。本文实现一个最简单的 softmax，并与 Torch 的性能进行比较。

## 实验环境

- NVIDIA-A100-PCIE-40GB
- Debian 11
- spack@0.23.0
- cuda@12.6.2
- py-triton@2.1.0
- py-torch@2.4.1+cuda
- py-matplotlib@3.7.5
- py-pandas@1.5.3

## 源代码 `softmax.py`

指定了 `num_warps=32`，实测不加这个的时候在 `n_col` 特别大的时候（`ncol=65536`）性能会大幅下降（`0.63ms->2.41ms`）。

```python
# spack load py-triton@2.1.0 py-torch@2.4.1+cuda py-matplotlib@3.7.5 py-pandas@1.5.3
# PATH=/usr/sbin:$PATH python3 relu.py
import triton
import triton.language as tl
import torch


@triton.jit
def kernel_softmax(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
    x = tl.exp(x - tl.max(x))
    x /= tl.sum(x)
    tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim0(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax[(n_rows,)](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=32,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_col"],
        x_vals=[2**i for i in range(8, 18)],  # triton maximum tensor numel (131072)
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        plot_name="softmax-time",
        args={},
    )
)
def benchmark(n_col, provider):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, n_col], device=DEVICE)
    if provider == "torch":
        method = lambda: torch.softmax(x, dim=1)
    if provider == "triton":
        method = lambda: triton_softmax_dim0(x)
    return triton.testing.do_bench(method)  # ms


if __name__ == "__main__":
    torch.manual_seed(3407)
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2, 4], device=DEVICE)
    y_torch = torch.softmax(x, dim=1)
    y_triton = triton_softmax_dim0(x)
    print(y_torch)
    print(y_triton)
    print(f"Maxdiff is " f"{torch.max(torch.abs(y_torch - y_triton))}")
    benchmark.run(print_data=True, show_plots=False, save_path=".")
```

## 程序输出

如图，可发现当 `n_col < 131072` 时 triton 的算子融合方式性能较好，否则性能差。推测过多的寄存器/SMEM占用影响了 occupancy。`n_col > 131072` 时直接无法启动。

```plain_text
tensor([[0.3550, 0.2351, 0.1792, 0.2307],
        [0.2826, 0.2976, 0.1968, 0.2230]], device='cuda:0')
tensor([[0.3550, 0.2351, 0.1792, 0.2307],
        [0.2826, 0.2976, 0.1968, 0.2230]], device='cuda:0')
Maxdiff is 1.4901161193847656e-08
softmax-time:
      n_col    Triton     Torch
0     256.0  0.016137  0.008521
1     512.0  0.016823  0.011331
2    1024.0  0.017489  0.015334
3    2048.0  0.021186  0.024648
4    4096.0  0.031589  0.034773
5    8192.0  0.056796  0.062525
6   16384.0  0.106099  0.120396
7   32768.0  0.208282  0.363455
8   65536.0  0.637752  0.794378
9  131072.0  3.938859  1.567888
```

![softmax-time](https://Mizuno-Ai.wu-kan.cn/assets/image/2024/11/21/softmax-time.png)
