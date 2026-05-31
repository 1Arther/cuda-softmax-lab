# CUDA Softmax Kernel Lab

本项目是一个用于学习和验证 **CUDA Row-wise Softmax 算子优化** 的小型实验项目。

项目实现了多个 Softmax kernel 版本，并在不同矩阵形状下进行 benchmark，对比各版本的正确性、执行时间、有效带宽和加速比。

测试环境：

```text
GPU: NVIDIA A40
SM count: 84
```

---

## 1. 项目目标

本项目主要关注二维矩阵的行级 Softmax：

```text
input:  [M, N]
output: [M, N]
```

对每一行独立计算：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

其中减去 `max(x)` 是为了数值稳定，避免 `exp` 溢出。

---

## 2. 已实现版本

目前实现了以下几个 CUDA kernel：

| 版本 | Kernel                        | 核心思想                                                     |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| v1   | `softmax_naive_kernel`        | 一个 block 处理一行，shared memory 做 block reduce           |
| v2   | `softmax_warp_kernel`         | 一个 block 处理一行，warp-level reduce 减少 shared memory 访问 |
| v3   | `softmax_warp_per_row_kernel` | 一个 warp 处理一行，适合短 row                               |
| v4   | `softmax_vec4_kernel`         | `float4` 向量化读写，提升 global memory 访问效率             |
| v5   | `softmax_vec4_fast_kernel`    | `float4 + __expf`，测试 fast math 对性能的影响               |
| v6   | `launch_dispatch`             | 根据 N 的大小自动选择 kernel                                 |

---

## 3. 编译方式

A40 对应 `sm_86`：

```bash
nvcc -O3 -arch=sm_86 main.cu softmax_row.cu -o softmax_bench
```

运行：

```bash
./softmax_bench
```

如果是 Ada 架构 GPU，例如 RTX 4090 / L40，可以改成：

```bash
nvcc -O3 -arch=sm_89 main.cu softmax_row.cu -o softmax_bench
```

---

## 4. Benchmark 方法

每个 kernel 都使用相同的 benchmark 流程：

1. 生成随机输入。
2. CPU 端计算 reference softmax。
3. CUDA kernel 计算输出。
4. 对比 CPU reference，计算最大绝对误差。
5. 检查每一行 softmax 输出和是否接近 1。
6. 进行 warmup。
7. 使用 `cudaEvent` 计时，多次 repeat 后取平均时间。
8. 估算有效带宽。

有效带宽近似按下面的访存量估计：

```text
traffic ≈ 5 * M * N * sizeof(float)
```

对应当前实现中的主要访存：

```text
1. 读 input 求 row max
2. 读 input 计算 exp
3. 写 output 保存 exp 中间结果
4. 读 output 做 normalize
5. 写 output 保存最终 softmax
```

这个带宽只是一个近似指标，主要用于横向比较不同 kernel。

---

## 5. Kernel 设计说明

### 5.1 naive shared-memory 版本

`softmax_naive_kernel` 使用一个 block 处理一行。

流程：

```text
1. 每个线程 stride 遍历当前行的一部分元素，得到 local max
2. shared memory 做 block reduce，得到 row max
3. 每个线程计算 exp(x - max)，并累加 local sum
4. shared memory 做 block reduce，得到 row sum
5. 每个线程将 output 除以 row sum
```

特点：

- 实现简单。
- 使用 shared memory 和 `__syncthreads()` 完成 block 内归约。
- 同步和 shared memory 访问开销相对较大。

---

### 5.2 warp-level block reduce 版本

`softmax_warp_kernel` 仍然是一个 block 处理一行，但是先在 warp 内用 `__shfl_down_sync` 做规约。

归约结构：

```text
thread local result
→ warp reduce
→ 每个 warp 的 lane 0 写 shared memory
→ 第一个 warp 归约所有 warp 的结果
→ 写回 smem[0]
→ 全 block 读取最终结果
```

相比 naive 版本，它减少了 shared memory reduce 的开销。

---

### 5.3 one-warp-per-row 版本

`softmax_warp_per_row_kernel` 使用一个 warp 处理一行。

映射关系：

```text
一个 block 有 blockDim.x / 32 个 warp
一个 warp 处理一行
一个 block 处理多行
```

优点：

- 不需要 shared memory。
- 不需要 block-level `__syncthreads()`。
- 短 row 场景下开销更小。

缺点：

- 一行只有 32 个线程处理。
- 当 N 较大时，单行并行度不足，性能明显下降。

实验结果显示，该版本主要适合 `N <= 128` 的短行场景。

---

### 5.4 float4 向量化版本

`softmax_vec4_kernel` 使用 `float4` 进行向量化读写。

原本每次处理一个 float：

```cpp
float x = row_input[i];
```

向量化后每次处理四个 float：

```cpp
float4 v = row_input4[i];
```

这样可以减少 global memory load/store 指令数量，提高访存效率。

该版本要求：

```text
N % 4 == 0
```

原因是：

```text
cudaMalloc 返回的起始地址通常是高对齐的；
但每一行起始地址 input + row * N 是否 16-byte 对齐，取决于 N 是否是 4 的倍数。
```

因此 dispatch 中只有当 `N % 4 == 0` 时才使用 vec4 kernel，否则 fallback 到普通 warp 版本。

---

### 5.5 vec4 + __expf 版本

`softmax_vec4_fast_kernel` 将：

```cpp
expf(x)
```

替换为：

```cpp
__expf(x)
```

`__expf` 是 CUDA 提供的快速近似指数函数，理论上可以降低指数计算开销，但精度略低。

本项目中额外 benchmark 了该版本，用来观察 fast exponential 对 Softmax 的影响。

当前实验结果显示：

- `__expf` 版本正确性正常。
- 输出误差仍在 `1e-8` 量级。
- 但性能相比普通 `vec4` 版本提升很小，很多场景基本持平。

因此默认 dispatch 暂时仍使用普通 `vec4` 版本，而不是 `vec4_fast`。

---

## 6. Dispatch 策略

当前自动选择策略：

```cpp
if (N <= 128) {
    launch_warp_per_row(...);
} else if (N % 4 == 0) {
    launch_vec4(...);
} else {
    launch_warp(...);
}
```

设计原因：

- `N <= 128` 时，one-warp-per-row 可以减少 shared memory 和同步开销。
- 中等和较大 N 时，一个 block 处理一行能提供更高单行并行度。
- 当 `N % 4 == 0` 时，优先使用 `float4` 向量化版本。
- `vec4_fast` 虽然被 benchmark，但默认不用于 dispatch，因为收益不稳定。

---

## 7. A40 Benchmark 结果

代表性结果如下：

|    M |     N | naive ms | warp ms | warp_row ms | vec4 ms | vec4_fast ms | auto ms | 最优加速比 |
| ---: | ----: | -------: | ------: | ----------: | ------: | -----------: | ------: | ---------: |
|  128 |    64 |   0.0029 |  0.0030 |      0.0028 |  0.0031 |       0.0031 |  0.0028 |     1.047x |
|  128 |   128 |   0.0029 |  0.0030 |      0.0028 |  0.0031 |       0.0031 |  0.0028 |     1.040x |
|  128 |   512 |   0.0038 |  0.0034 |      0.0048 |  0.0033 |       0.0033 |  0.0033 |     1.160x |
|  128 |  1024 |   0.0045 |  0.0041 |      0.0073 |  0.0036 |       0.0036 |  0.0036 |     1.255x |
|  512 |  1024 |   0.0084 |  0.0071 |      0.0075 |  0.0058 |       0.0058 |  0.0058 |     1.455x |
| 1024 |  1024 |   0.0185 |  0.0181 |      0.0234 |  0.0139 |       0.0139 |  0.0139 |     1.332x |
| 1024 |  4096 |   0.1323 |  0.1326 |      0.1861 |  0.1254 |       0.1252 |  0.1252 |     1.057x |
|  256 | 50000 |   0.5548 |  0.5552 |      1.8059 |  0.4812 |       0.4811 |  0.4809 |     1.154x |

正确性：

```text
max absolute error: 约 1e-8
row sum error:      约 1e-6 到 1e-5
```

---

## 8. 实验结论

### 8.1 one-warp-per-row 只适合短 row

当 `N = 64 / 128` 时，该版本略快，因为它避免了 shared memory 和 block 级同步。

但当 N 变大后，它明显变慢。原因是每一行只有 32 个线程处理，单行并行度不足。

---

### 8.2 warp-level block reduction 收益有限

相比 naive shared-memory reduce，warp-level block reduce 减少了一部分同步和 shared memory 访问。

但 Softmax 的主要开销还包括：

```text
global memory 读写
expf 指数计算
多轮 row 遍历
```

所以 warp reduce 的收益是有限且 shape-dependent 的。

---

### 8.3 float4 是当前最有效的优化

`float4` 向量化版本在中等 row length 上提升明显。

例如：

```text
M = 512, N = 1024
naive:    0.0084 ms
vec4fast: 0.0058 ms
speedup:  1.455x
```

说明向量化访存可以有效减少 global memory load/store 指令开销。

---

### 8.4 __expf 收益不明显

`vec4_fast` 使用 `__expf` 替代 `expf`。

实验显示：

- 正确性正常。
- 性能和普通 `vec4` 版本基本接近。
- 说明当前 shape 下瓶颈不完全是指数函数本身，还受到访存、调度和整体 kernel 结构影响。

---

## 9. 文件结构

```text
.
├── main.cu
└── softmax_row.cu
```

其中：

```text
main.cu        benchmark、CPU reference、正确性验证、dispatch 测试
softmax_row.cu CUDA Softmax kernel 实现
```

---

## 10. 面试表达总结

这个项目可以这样介绍：

> 我实现了一个 row-wise Softmax CUDA benchmark，包括 shared-memory block reduce、warp-level reduce、one-warp-per-row、float4 向量化、fast exponential 和 shape-based dispatch 多个版本。
>
> 实验发现 one-warp-per-row 只适合短 row；中等和大 row 更适合 block-per-row；float4 向量化在部分 shape 下可以获得 1.1x 到 1.45x 左右加速；而 __expf 在当前测试中收益不明显。
>
> 这个项目主要训练了我对 CUDA row-wise 算子、warp reduce、all-reduce、vectorized load/store、数值稳定性和 benchmark 方法的理解。