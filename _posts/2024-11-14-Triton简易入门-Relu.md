---
title: Triton简易入门-Relu
tags: 学习笔记与作业
---

学习一下 triton。配置环境的时候遇到一个问题，运行时需要用的 `ldconfig` 在 `/usr/sbin/` 目录下，很奇怪为啥要这么搞。我的解决方式是直接 `cp` 到当前工作目录下，并在运行时加入 `PATH`。

```shell
cp /usr/sbin/ldconfig .
PATH=$HOME/triton-train:$PATH python3 relu.py
```

以下是首次上手编写的 `relu` 函数，作为一个最小化的上手例子。

```python3
# spack load py-triton@2.1.0 py-torch@2.4.1+cuda py-matplotlib@3.7.5 py-pandas@1.5.3
# cp /usr/sbin/ldconfig .
# PATH=$HOME/triton-train:$PATH python3 relu.py
import triton
import triton.language as tl
import torch


@triton.jit
def relu_kernel(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    # tl.arange 返回一个数组 [0, 1 , ..., BLOCK_SIZE - 1]
    # tl.program_id(0) * BLOCK_SIZE 是一个数，加在一起 idx 是一个数组
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask 和 idx 一样是数组
    mask = idx < size
    # x_ptr + idx 是数组，所以可以一次 load 一整个数组
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(y_ptr + idx, y, mask=mask)


def triton_relu(x: torch.Tensor):
    assert x.is_contiguous()  # 确保输入是1D张量
    size = x.numel()
    y = torch.empty_like(x)
    gridDim = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    relu_kernel[gridDim](x, y, size, BLOCK_SIZE=256)
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        plot_name="relu-performance",
        args={},
    )
)
def benchmark(size, provider):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    x -= 0.5
    if provider == "torch":
        method = lambda: torch.relu(x)
    if provider == "triton":
        method = lambda: triton_relu(x)
    return triton.testing.do_bench(method)  # ms


def test(size):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand(size, device=DEVICE)
    x -= 0.5
    y_torch = torch.relu(x)
    y_triton = triton_relu(x)
    print(y_torch)
    print(y_triton)
    print(f"Maxdiff is " f"{torch.max(torch.abs(y_torch - y_triton))}")


if __name__ == "__main__":
    torch.manual_seed(3407)
    test(2**20)
    benchmark.run(print_data=True, show_plots=False, save_path=".")
```

程序输出如下：

```shell
tensor([0.3659, 0.0000, 0.0000,  ..., 0.0000, 0.3017, 0.0000], device='cuda:0')
tensor([0.3659, 0.0000, 0.0000,  ..., 0.0000, 0.3017, 0.0000], device='cuda:0')
Maxdiff is 0.0
relu-performance:
           size    Triton     Torch
0        4096.0  0.029745  0.006186
1        8192.0  0.006276  0.005931
2       16384.0  0.006349  0.006032
3       32768.0  0.006436  0.006093
4       65536.0  0.006222  0.006691
5      131072.0  0.006567  0.007026
6      262144.0  0.007626  0.008013
7      524288.0  0.009485  0.009593
8     1048576.0  0.012198  0.011711
9     2097152.0  0.018028  0.017868
10    4194304.0  0.030379  0.030201
11    8388608.0  0.056574  0.056206
12   16777216.0  0.104953  0.104733
13   33554432.0  0.201216  0.201807
14   67108864.0  0.394154  0.394326
15  134217728.0  0.777879  0.781234
```

性能如图，貌似 size 较小时 triton 不如 torch，可考虑调小 `BLOCK_SIZE`。

![relu-performance](https://Mizuno-Ai.wu-kan.cn/assets/image/2024/11/14/relu-performance.png)
