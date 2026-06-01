# CUDA Softmax Kernel Lab

本项目从零实现并测试了多种 **CUDA Row-wise Softmax** 及 Attention 中常见的 Softmax 变体，重点覆盖：

- 普通 Row-wise Softmax
- Warp-level reduction 优化
- One-warp-per-row 短序列优化
- Float4 向量化访存
- `__expf` 快速指数函数
- Scaled Masked Softmax
- Scaled Causal Softmax

这些算子是 Transformer Attention 中的核心组成部分，可作为继续实现 **Unfused Attention / FlashAttention 简化版** 的基础。

---

## 1. 项目文件

```text
.
├── main.cu          # CPU reference、correctness check、benchmark
├── softmax_row.cu   # CUDA softmax kernels
└── README.md
```

---

## 2. Row-wise Softmax

输入输出形状：

```text
input:  [M, N]
output: [M, N]
```

每一行独立计算 softmax：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

其中减去 `max(x)` 是为了数值稳定，避免 `exp` 溢出。

---

## 3. 已实现 Kernel

| 版本 | Kernel                              | 核心思想                                                     |
| ---- | ----------------------------------- | ------------------------------------------------------------ |
| v1   | `softmax_naive_kernel`              | 一个 block 处理一行，shared memory 做 block-level reduction  |
| v2   | `softmax_warp_kernel`               | 一个 block 处理一行，warp shuffle reduce + shared memory 汇总 |
| v3   | `softmax_warp_per_row_kernel`       | 一个 warp 处理一行，适合短 row                               |
| v4   | `softmax_vec4_kernel`               | 使用 `float4` 向量化读写                                     |
| v5   | `softmax_vec4_fast_kernel`          | `float4 + __expf`，测试快速指数函数                          |
| v6   | `launch_dispatch`                   | 根据 `N` 自动选择 kernel                                     |
| v7   | `scaled_masked_softmax_warp_kernel` | 支持 `scale` 和显式 mask                                     |
| v8   | `scaled_causal_softmax_warp_kernel` | 支持 decoder self-attention 中的 causal mask                 |

---

## 4. 普通 Softmax 优化路线

### 4.1 Naive Softmax

`softmax_naive_kernel` 使用一个 CUDA block 处理一行。

流程：

```text
1. 每个线程 stride 遍历当前行，得到 local max
2. shared memory reduction 得到 row max
3. 每个线程计算 exp(x - max)，并累加 local sum
4. shared memory reduction 得到 row sum
5. 每个线程将 output 除以 row sum
```

优点是结构清楚，适合作为 baseline。缺点是 shared memory 访问和 `__syncthreads()` 较多。

---

### 4.2 Warp-level Softmax

`softmax_warp_kernel` 仍然是一个 block 处理一行，但使用 `__shfl_down_sync` 做 warp 内 reduction。

归约流程：

```text
thread local result
-> warp reduce
-> 每个 warp 的 lane 0 写入 shared memory
-> 第一个 warp reduce 所有 warp partial result
-> 写回 smem[0]
-> 全 block 读取最终结果
```

相比 naive 版本，它减少了 shared memory 访问和 block-level synchronization。

---

### 4.3 One-warp-per-row Softmax

`softmax_warp_per_row_kernel` 使用一个 warp 处理一行。

映射关系：

```text
一个 block 有 blockDim.x / 32 个 warp
一个 warp 处理一行
一个 block 处理多行
```

优点：

```text
1. 不需要 shared memory
2. 不需要 block-level __syncthreads()
3. 短 row 场景下开销低
```

缺点：

```text
1. 一行只有 32 个线程处理
2. 当 N 较大时，单行并行度不足
```

因此它更适合：

```text
N <= 128
```

---

### 4.4 Float4 Vectorized Softmax

`softmax_vec4_kernel` 使用 `float4` 进行向量化读写。

普通标量读写：

```cpp
float x = row_input[i];
```

向量化后：

```cpp
float4 v = row_input4[i];
```

每个线程一次处理 4 个连续 float，可以减少访存指令数量，提高 global memory load/store 效率。

该版本要求：

```text
N % 4 == 0
```

因为需要保证每一行起始地址能够按 `float4` 对齐访问。

---

### 4.5 Float4 + `__expf`

`softmax_vec4_fast_kernel` 将：

```cpp
expf(x)
```

替换为：

```cpp
__expf(x)
```

`__expf` 是 CUDA 的快速近似指数函数，可能更快，但精度略低。当前实验中，`__expf` 的正确性正常，但速度相比普通 `expf` 基本持平。

---

## 5. Dispatch 策略

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

- `N <= 128` 时，one-warp-per-row 可以减少同步和 shared memory 开销。
- 中等和较大 `N` 时，一个 block 处理一行能提供更高单行并行度。
- 当 `N % 4 == 0` 时，优先使用 `float4` 向量化版本。

---

## 6. Scaled Masked Softmax

Attention 中常用的形式不是普通 softmax，而是：

```text
softmax(score * scale + mask)
```

其中：

```text
scale = 1 / sqrt(head_dim)
```

显式 mask 的语义：

```text
mask[i] != 0: 当前位置有效
mask[i] == 0: 当前位置被屏蔽
```

实现时不需要真的给 masked 位置加 `-inf`，而是在 max 和 sum 两个 reduction 阶段直接跳过 masked 位置。

语义：

```text
if valid:
    output[i] = exp(input[i] * scale - max_valid) / sum_valid
else:
    output[i] = 0
```

核心判断：

```cpp
bool valid = (row_mask == nullptr) || (row_mask[i] != 0);
```

其中 `mask == nullptr` 表示没有 mask，所有位置都有效。

---

## 7. Scaled Causal Softmax

Causal Softmax 是 decoder self-attention 中使用的 mask。它保证当前位置不能看到未来 token：

```text
key index <= query index
```

实现中不需要显式构造 causal mask 矩阵，只需要根据列号判断：

```cpp
bool valid = i <= q;
```

其中：

```cpp
int q = row % query_len;
```

原因是 attention scores 通常是：

```text
scores: [B, H, Q, K]
```

为了做 row-wise softmax，会展平成：

```text
scores: [B * H * Q, K]
```

因此每一行对应一个 query。对于展平后的 row：

```text
q = row % query_len
```

可以恢复当前 query 在序列中的位置。

例如：

```text
B = 2, H = 1, query_len = 4, key_len = 4
M = B * H * query_len = 8
N = key_len = 4
```

|  row | q = row % query_len | 可见 key   |
| ---: | ------------------: | ---------- |
|    0 |                   0 | 0          |
|    1 |                   1 | 0, 1       |
|    2 |                   2 | 0, 1, 2    |
|    3 |                   3 | 0, 1, 2, 3 |
|    4 |                   0 | 0          |
|    5 |                   1 | 0, 1       |
|    6 |                   2 | 0, 1, 2    |
|    7 |                   3 | 0, 1, 2, 3 |

Causal mask 矩阵形状：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

---

## 8. 编译与运行

测试 GPU 为 NVIDIA A40，编译命令如下：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 main.cu softmax_row.cu -o softmax_bench
./softmax_bench
```

如果是 Ada 架构 GPU，例如 RTX 4090 / L40，可以使用：

```bash
nvcc -O3 -std=c++17 -arch=sm_89 main.cu softmax_row.cu -o softmax_bench
./softmax_bench
```

---

## 9. Benchmark 方法

每个 kernel 使用相同流程：

```text
1. 生成随机输入
2. CPU 端计算 reference softmax
3. CUDA kernel 计算输出
4. 对比 CPU reference，计算最大绝对误差
5. 检查每一行 softmax 输出和是否接近 1
6. warmup
7. 使用 cudaEvent 计时，多次 repeat 后取平均时间
8. 估算有效带宽
```

普通 softmax 的理论访存量近似为：

```text
traffic ≈ 5 * M * N * sizeof(float)
```

对应当前实现：

```text
1. 读 input 求 row max
2. 读 input 计算 exp
3. 写 output 保存 exp 中间结果
4. 读 output 做 normalize
5. 写 output 保存最终 softmax
```

Masked softmax 额外读取 mask：

```text
masked_traffic ≈ 5 * M * N * sizeof(float) + 2 * M * N * sizeof(uint8_t)
```

注意：README 中的 GB/s 是根据理论访存量估算的 **effective bandwidth**，不是 Nsight Compute 直接测得的 DRAM throughput。

---

## 10. 实验环境

```text
GPU: NVIDIA A40
SM count: 84
```

---

## 11. Row-wise Softmax Benchmark

代表性结果如下：

|    M |     N | naive ms | warp ms | warp_row ms | vec4 ms | vec4_fast ms | auto ms | best speedup |
| ---: | ----: | -------: | ------: | ----------: | ------: | -----------: | ------: | -----------: |
|  128 |    64 |   0.0029 |  0.0030 |      0.0029 |  0.0031 |       0.0031 |  0.0029 |       1.000x |
|  128 |   128 |   0.0029 |  0.0030 |      0.0029 |  0.0031 |       0.0031 |  0.0029 |       1.014x |
|  128 |   512 |   0.0038 |  0.0034 |      0.0048 |  0.0033 |       0.0033 |  0.0033 |       1.163x |
|  128 |  1024 |   0.0045 |  0.0041 |      0.0073 |  0.0036 |       0.0036 |  0.0036 |       1.256x |
|  512 |  1024 |   0.0084 |  0.0070 |      0.0075 |  0.0058 |       0.0058 |  0.0058 |       1.460x |
| 1024 |  1024 |   0.0184 |  0.0181 |      0.0233 |  0.0138 |       0.0139 |  0.0138 |       1.329x |
| 1024 |  4096 |   0.1322 |  0.1325 |      0.1861 |  0.1253 |       0.1250 |  0.1252 |       1.058x |
|  256 | 50000 |   0.5549 |  0.5552 |      1.8050 |  0.4810 |       0.4811 |  0.4807 |       1.154x |

正确性：

```text
max absolute error: around 1e-8
row sum error:      around 1e-7 to 1e-5
```

---

## 12. Scaled Masked / Causal Softmax Benchmark

|    M |    N | masked ms | causal ms | masked GB/s | causal GB/s | masked err | causal err | masked zero err | causal zero err |
| ---: | ---: | --------: | --------: | ----------: | ----------: | ---------: | ---------: | --------------: | --------------: |
|  128 |   64 |    0.0033 |    0.0031 |       54.83 |       53.33 |   1.49e-08 |   2.98e-08 |        0.00e+00 |        0.00e+00 |
|  128 |  128 |    0.0033 |    0.0032 |      108.31 |      103.56 |   7.45e-09 |   2.98e-08 |        0.00e+00 |        0.00e+00 |
|  256 |  256 |    0.0041 |    0.0036 |      352.00 |      366.76 |   5.59e-09 |   2.98e-08 |        0.00e+00 |        0.00e+00 |
|  512 |  512 |    0.0065 |    0.0056 |      881.38 |      932.60 |   4.66e-09 |   2.98e-08 |        0.00e+00 |        0.00e+00 |
| 1024 | 1024 |    0.0230 |    0.0134 |     1002.58 |     1568.15 |   2.56e-09 |   2.98e-08 |        0.00e+00 |        0.00e+00 |
| 2048 |  512 |    0.0217 |    0.0147 |     1061.14 |     1430.17 |   4.19e-09 |   4.47e-08 |        0.00e+00 |        0.00e+00 |
| 4096 | 1024 |    0.0726 |    0.0473 |     1270.97 |     1772.01 |   3.49e-09 |   5.96e-08 |        0.00e+00 |        0.00e+00 |

说明：

- `masked_zeroerr = 0` 表示 masked 位置输出严格为 0。
- `causal_zeroerr = 0` 表示未来 token 位置输出严格为 0。
- `causal_ms` 通常小于 `masked_ms`，因为 causal mask 不需要读取显式 mask tensor，只需要根据列号判断 `i <= q`。

---

## 13. 实验结论

### 13.1 One-warp-per-row 适合短 row

当 `N = 64 / 128` 时，one-warp-per-row 版本表现较好，因为它避免了 shared memory 和 block-level synchronization。

但当 `N` 变大后，它明显变慢。例如 `N = 50000` 时，one-warp-per-row 只有一个 warp 处理一行，单行并行度严重不足。

---

### 13.2 Warp-level block reduction 收益有限

相比 naive shared-memory reduction，warp-level block reduction 减少了一部分同步和 shared memory 访问。

但 Softmax 仍然有以下开销：

```text
1. 多次 global memory 访问
2. expf 指数计算
3. 两次 row-level reduction
4. 多轮 row 遍历
```

所以 warp reduce 的收益是 shape-dependent 的。

---

### 13.3 Float4 是当前最有效的普通 Softmax 优化

`float4` 向量化版本在中等 row length 上提升明显。

例如：

```text
M = 512, N = 1024
naive:     0.0084 ms
vec4_fast: 0.0058 ms
speedup:   1.460x
```

说明向量化访存可以有效减少 global memory load/store 指令开销。

---

### 13.4 `__expf` 收益不明显

`__expf` 理论上可以加速指数计算，但当前实验中相比普通 `expf` 提升很小，很多场景基本持平。

可能原因是当前 kernel 仍然受到多轮访存、同步和 reduction 开销影响，单独替换指数函数并不能显著改变整体性能。

---

### 13.5 Masked / Causal Softmax 更贴近 Attention

普通 Softmax 是基础版本，但 Attention 中更常见的是：

```text
scaled masked softmax
scaled causal softmax
```

这两个版本的关键不是最后把无效位置置 0，而是要在 **max 和 sum reduction 阶段就排除无效位置**。否则被 mask 的大值仍然会影响 softmax 的数值结果。

---

## 14. 面试讲法

可以这样介绍这个项目：

```text
我实现了 row-wise softmax 的多个 CUDA 版本，包括 naive shared-memory reduction、
warp-level reduction、one-warp-per-row、float4 vectorized、float4 + __expf，以及 dispatch 策略。

普通 softmax 中，每一行先 reduce max，再计算 exp 和 reduce sum，最后 normalize。
为了数值稳定，计算 exp 前会减去 row max。

对于 attention softmax，我进一步实现了 scaled masked softmax 和 scaled causal softmax。
scale 来自 1 / sqrt(head_dim)，用于控制 QK 点积的数值范围。

masked softmax 不直接对无效位置加 -inf，而是在 max 和 sum reduction 阶段跳过无效位置，
最后 masked 位置输出 0。

causal softmax 用于 decoder self-attention，核心判断是 key index <= query index。
在 scores 展平成 [B * H * query_len, key_len] 后，可以通过 row % query_len 恢复当前 query index。
```

---

## 15. 后续方向

下一步可以继续实现：

```text
1. Unfused Attention Forward: QK^T + scaled causal softmax + P @ V
2. Tiled QK^T
3. Online Softmax
4. Simplified FlashAttention
```

FlashAttention 的核心思想是：

```text
不显式落地完整 S x S attention matrix，
而是在 tile 内完成 QK、online softmax 和 PV 累加。
```