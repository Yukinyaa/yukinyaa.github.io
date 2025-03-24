---
title: Triton 入门-比 Torch 快的 OnlineSoftmax
tags: 学习笔记与作业
---

本文基于 Triton 逐步实现 online-softmax 算子，并与 Torch 的性能进行比较。实验结果显示，我的版本相较于 Torch 稳定快 24%。

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

分别实现了三个版本的 softmax：

1. softmax_fuse：每个 block 一次 load 一整列元素，直接算。这样做在 n 较小（`n_col < 65536`） 时 triton 的算子融合方式性能较好，否则性能差。推测过多的寄存器/SMEM占用影响了 occupancy。`n_col > 131072` 时直接无法启动。平均每个元素 1 次 load 1 次 store。
2. softmax_tile：按照列分tile处理寄存器/SMEM不够处理整列的问题。使用三次循环：第一次算整列的 max，第二次算整列的 sum of exp，第三次逐元素算 softmax。平均每个元素 3 次 load 次store。
3. softmax_online：在 softmax_tile 基础上将前两次 online 算 max 和 sum，这样可以减少 1 次 load 操作。由于 softmax 是访存密集算子，多几次 `exp` 交换也是可以接受的。

顺带一提，我认为区间规约操作 `tl.max` `tl.sum` 开销很大，因此把他们都写在循环外面。

尤其值得注意的是 114 行处处理的 `nan` 问题，真的 debug 了很久。

对 `num_warps=32`、`BLOCK_SIZE` 进行了一些手动调优。

```python
# spack load py-triton@2.1.0 py-torch@2.4.1+cuda py-matplotlib@3.7.5 py-pandas@1.5.3
# PATH=/usr/sbin:$PATH python3 softmax.py
import triton
import triton.language as tl
import torch


@triton.jit
def kernel_softmax_fuse(
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
    eps = float(1e-9)
    x /= tl.maximum(tl.sum(x), eps)
    tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_fuse(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_fuse[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_tile(
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

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm = tl.maximum(mm, x)
    mm = tl.max(mm)

    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm)
        ss += x
    ss = tl.sum(ss)
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_tile(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_tile[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**14,
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_online(
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

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm_new = tl.maximum(mm, x)
        if i:  # 第 1 轮不需要，且容易整出 nan
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new

    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new

    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_online(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        num_warps=32,
    )
    return y


def test():
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, 2**15], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online": lambda: triton_softmax_dim1_online(x),
    }
    y_torch = mp["torch"]()
    for k, v in mp.items():
        y_triton = v()
        print("{}: Maxdiff is {}".format(k, torch.max(torch.abs(y_torch - y_triton))))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_col"],
        x_vals=[2**i for i in range(8, 18)],  # triton maximum tensor numel (131072)
        line_arg="provider",
        line_vals=["torch", "triton_fuse", "triton_tile", "triton_online"],
        line_names=["Torch", "Triton_fuse", "Triton_tile", "Triton_online"],
        plot_name="softmax-time",
        args={},
    )
)
def benchmark(n_col, provider):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, n_col], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online": lambda: triton_softmax_dim1_online(x),
    }
    return triton.testing.do_bench(mp[provider])  # ms


if __name__ == "__main__":
    torch.manual_seed(3407)
    test()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
```

## 程序输出

```plain_text
torch: Maxdiff is 0.0
triton_fuse: Maxdiff is 1.0913936421275139e-11
triton_tile: Maxdiff is 1.4551915228366852e-11
triton_online: Maxdiff is 1.4551915228366852e-11
softmax-time:
      n_col     Torch  Triton_fuse  Triton_tile  Triton_online
0     256.0  0.009117     0.016346     0.063920       0.026654
1     512.0  0.010990     0.017012     0.063512       0.027439
2    1024.0  0.015108     0.017947     0.063908       0.028337
3    2048.0  0.024704     0.020698     0.065377       0.029616
4    4096.0  0.035482     0.032225     0.071429       0.035339
5    8192.0  0.061941     0.056879     0.079440       0.062371
6   16384.0  0.120909     0.106085     0.122564       0.122939
7   32768.0  0.363950     0.209142     0.231075       0.297316
8   65536.0  0.796443     0.640258     0.597622       0.606097
9  131072.0  1.567942     3.947307     1.390112       1.199244
```

![softmax-time](https://Mizuno-Ai.wu-kan.cn/assets/image/2024/11/21/softmax-time.png)
