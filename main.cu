// main.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// kernel 声明，对应 softmax_row.cu 里的实现
__global__ void softmax_naive_kernel(const float* input, float* output, int M, int N);
__global__ void softmax_warp_kernel(const float* input, float* output, int M, int N);
__global__ void softmax_warp_per_row_kernel(const float* input, float* output, int M, int N);
__global__ void softmax_vec4_kernel(const float* input, float* output, int M, int N);
__global__ void softmax_vec4_fast_kernel(const float* input, float* output, int M, int N);

struct BenchConfig {
    int M;
    int N;
    int block_size;
    int warmup;
    int repeat;
};

using Launcher = void (*)(
    const float*,
    float*,
    int,
    int,
    int
);

void softmax_cpu_reference(
    const std::vector<float>& input,
    std::vector<float>& output,
    int M,
    int N
) {
    for (int r = 0; r < M; ++r) {
        const float* row_input = input.data() + r * N;
        float* row_output = output.data() + r * N;

        float max_val = -FLT_MAX;
        for (int c = 0; c < N; ++c) {
            max_val = std::max(max_val, row_input[c]);
        }

        float sum_val = 0.0f;
        for (int c = 0; c < N; ++c) {
            float e = std::exp(row_input[c] - max_val);
            row_output[c] = e;
            sum_val += e;
        }

        for (int c = 0; c < N; ++c) {
            row_output[c] /= sum_val;
        }
    }
}

float max_abs_error(
    const std::vector<float>& ref,
    const std::vector<float>& out
) {
    float err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        err = std::max(err, std::fabs(ref[i] - out[i]));
    }
    return err;
}

float check_row_sum_error(
    const std::vector<float>& out,
    int M,
    int N
) {
    float max_err = 0.0f;

    for (int r = 0; r < M; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            sum += out[r * N + c];
        }
        max_err = std::max(max_err, std::fabs(sum - 1.0f));
    }

    return max_err;
}

void launch_naive(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    dim3 grid(M);
    size_t smem = block_size * sizeof(float);
    softmax_naive_kernel<<<grid, block_size, smem>>>(d_input, d_output, M, N);
}

void launch_warp(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    dim3 grid(M);
    int num_warps = (block_size + 31) / 32;
    size_t smem = num_warps * sizeof(float);
    softmax_warp_kernel<<<grid, block_size, smem>>>(d_input, d_output, M, N);
}

void launch_warp_per_row(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    int warps_per_block = block_size / 32;
    dim3 grid((M + warps_per_block - 1) / warps_per_block);

    softmax_warp_per_row_kernel<<<grid, block_size>>>(d_input, d_output, M, N);
}

void launch_vec4(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    // vec4 kernel 只处理 N % 4 == 0 的情况
    // 其他情况 fallback 到 warp block reduce 版本
    if (N % 4 != 0) {
        launch_warp(d_input, d_output, M, N, block_size);
        return;
    }

    dim3 grid(M);
    int num_warps = (block_size + 31) / 32;
    size_t smem = num_warps * sizeof(float);

    softmax_vec4_kernel<<<grid, block_size, smem>>>(d_input, d_output, M, N);
}

void launch_vec4_fast(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    if (N % 4 != 0) {
        launch_warp(d_input, d_output, M, N, block_size);
        return;
    }

    dim3 grid(M);
    int num_warps = (block_size + 31) / 32;
    size_t smem = num_warps * sizeof(float);

    softmax_vec4_fast_kernel<<<grid, block_size, smem>>>(d_input, d_output, M, N);
}

void launch_dispatch(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    if (N <= 128) {
        launch_warp_per_row(d_input, d_output, M, N, block_size);
    } else if (N % 4 == 0) {
        launch_vec4(d_input, d_output, M, N, block_size);
    } else {
        launch_warp(d_input, d_output, M, N, block_size);
    }
}

float benchmark_kernel(
    Launcher launcher,
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launcher(d_input, d_output, M, N, block_size);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launcher(d_input, d_output, M, N, block_size);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return total_ms / repeat;
}

float check_correctness(
    Launcher launcher,
    const std::vector<float>& h_ref,
    std::vector<float>& h_out,
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    size_t bytes = static_cast<size_t>(M) * N * sizeof(float);

    launcher(d_input, d_output, M, N, block_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    return max_abs_error(h_ref, h_out);
}

float check_sum_error(
    Launcher launcher,
    std::vector<float>& h_out,
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int block_size
) {
    size_t bytes = static_cast<size_t>(M) * N * sizeof(float);

    launcher(d_input, d_output, M, N, block_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    return check_row_sum_error(h_out, M, N);
}

void run_one_config(const BenchConfig& cfg) {
    int M = cfg.M;
    int N = cfg.N;
    int block_size = cfg.block_size;

    size_t elems = static_cast<size_t>(M) * N;
    size_t bytes = elems * sizeof(float);

    std::vector<float> h_input(elems);
    std::vector<float> h_ref(elems);
    std::vector<float> h_out(elems);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (float& x : h_input) {
        x = dist(gen);
    }

    softmax_cpu_reference(h_input, h_ref, M, N);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    float naive_err = check_correctness(
        launch_naive, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float warp_err = check_correctness(
        launch_warp, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float row_err = check_correctness(
        launch_warp_per_row, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float vec4_err = check_correctness(
        launch_vec4, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float vec4_fast_err = check_correctness(
        launch_vec4_fast, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float auto_err = check_correctness(
        launch_dispatch, h_ref, h_out, d_input, d_output, M, N, block_size
    );

    float naive_sumerr = check_sum_error(
        launch_naive, h_out, d_input, d_output, M, N, block_size
    );

    float warp_sumerr = check_sum_error(
        launch_warp, h_out, d_input, d_output, M, N, block_size
    );

    float row_sumerr = check_sum_error(
        launch_warp_per_row, h_out, d_input, d_output, M, N, block_size
    );

    float vec4_sumerr = check_sum_error(
        launch_vec4, h_out, d_input, d_output, M, N, block_size
    );

    float vec4_fast_sumerr = check_sum_error(
        launch_vec4_fast, h_out, d_input, d_output, M, N, block_size
    );

    float auto_sumerr = check_sum_error(
        launch_dispatch, h_out, d_input, d_output, M, N, block_size
    );

    float naive_ms = benchmark_kernel(
        launch_naive,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    float warp_ms = benchmark_kernel(
        launch_warp,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    float row_ms = benchmark_kernel(
        launch_warp_per_row,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    float vec4_ms = benchmark_kernel(
        launch_vec4,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    float vec4_fast_ms = benchmark_kernel(
        launch_vec4_fast,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    float auto_ms = benchmark_kernel(
        launch_dispatch,
        d_input,
        d_output,
        M,
        N,
        block_size,
        cfg.warmup,
        cfg.repeat
    );

    // 当前实现大致访存：
    // 1. 读 input 求 max
    // 2. 读 input 写 output 保存 exp
    // 3. 读 output 写 output 做 normalize
    // 约等于 5 * M * N * sizeof(float)
    double traffic_bytes = 5.0 * static_cast<double>(M) * N * sizeof(float);

    double naive_gbs = traffic_bytes / (naive_ms / 1000.0) / 1e9;
    double warp_gbs = traffic_bytes / (warp_ms / 1000.0) / 1e9;
    double row_gbs = traffic_bytes / (row_ms / 1000.0) / 1e9;
    double vec4_gbs = traffic_bytes / (vec4_ms / 1000.0) / 1e9;
    double vec4_fast_gbs = traffic_bytes / (vec4_fast_ms / 1000.0) / 1e9;
    double auto_gbs = traffic_bytes / (auto_ms / 1000.0) / 1e9;

    std::cout << std::left
              << std::setw(8) << M
              << std::setw(8) << N
              << std::setw(8) << block_size

              << std::setw(12) << std::fixed << std::setprecision(4) << naive_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << warp_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << row_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << vec4_ms
              << std::setw(16) << std::fixed << std::setprecision(4) << vec4_fast_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << auto_ms

              << std::setw(10) << std::fixed << std::setprecision(3) << naive_ms / warp_ms
              << std::setw(10) << std::fixed << std::setprecision(3) << naive_ms / row_ms
              << std::setw(10) << std::fixed << std::setprecision(3) << naive_ms / vec4_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << naive_ms / vec4_fast_ms
              << std::setw(10) << std::fixed << std::setprecision(3) << naive_ms / auto_ms

              << std::setw(12) << std::fixed << std::setprecision(2) << naive_gbs
              << std::setw(12) << std::fixed << std::setprecision(2) << warp_gbs
              << std::setw(12) << std::fixed << std::setprecision(2) << row_gbs
              << std::setw(12) << std::fixed << std::setprecision(2) << vec4_gbs
              << std::setw(16) << std::fixed << std::setprecision(2) << vec4_fast_gbs
              << std::setw(12) << std::fixed << std::setprecision(2) << auto_gbs

              << std::setw(14) << std::scientific << std::setprecision(2) << naive_err
              << std::setw(14) << std::scientific << std::setprecision(2) << warp_err
              << std::setw(14) << std::scientific << std::setprecision(2) << row_err
              << std::setw(14) << std::scientific << std::setprecision(2) << vec4_err
              << std::setw(16) << std::scientific << std::setprecision(2) << vec4_fast_err
              << std::setw(14) << std::scientific << std::setprecision(2) << auto_err

              << std::setw(14) << std::scientific << std::setprecision(2) << naive_sumerr
              << std::setw(14) << std::scientific << std::setprecision(2) << warp_sumerr
              << std::setw(14) << std::scientific << std::setprecision(2) << row_sumerr
              << std::setw(14) << std::scientific << std::setprecision(2) << vec4_sumerr
              << std::setw(16) << std::scientific << std::setprecision(2) << vec4_fast_sumerr
              << std::setw(14) << std::scientific << std::setprecision(2) << auto_sumerr
              << "\n";

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n\n";

    std::cout << std::left
              << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "block"

              << std::setw(12) << "naive_ms"
              << std::setw(12) << "warp_ms"
              << std::setw(14) << "warp_row_ms"
              << std::setw(12) << "vec4_ms"
              << std::setw(16) << "vec4_fast_ms"
              << std::setw(12) << "auto_ms"

              << std::setw(10) << "warp_spd"
              << std::setw(10) << "row_spd"
              << std::setw(10) << "vec4_spd"
              << std::setw(14) << "v4fast_spd"
              << std::setw(10) << "auto_spd"

              << std::setw(12) << "naive_GB/s"
              << std::setw(12) << "warp_GB/s"
              << std::setw(12) << "row_GB/s"
              << std::setw(12) << "vec4_GB/s"
              << std::setw(16) << "v4fast_GB/s"
              << std::setw(12) << "auto_GB/s"

              << std::setw(14) << "naive_err"
              << std::setw(14) << "warp_err"
              << std::setw(14) << "row_err"
              << std::setw(14) << "vec4_err"
              << std::setw(16) << "v4fast_err"
              << std::setw(14) << "auto_err"

              << std::setw(14) << "naive_sumerr"
              << std::setw(14) << "warp_sumerr"
              << std::setw(14) << "row_sumerr"
              << std::setw(14) << "vec4_sumerr"
              << std::setw(16) << "v4fast_sumerr"
              << std::setw(14) << "auto_sumerr"
              << "\n";

    std::vector<BenchConfig> configs = {
        {128,   64,    128, 20, 100},
        {128,   128,   128, 20, 100},
        {128,   256,   128, 20, 100},
        {128,   512,   256, 20, 100},
        {128,   1024,  256, 20, 100},
        {512,   1024,  256, 20, 100},
        {1024,  1024,  256, 20, 100},
        {1024,  2048,  256, 20, 100},
        {1024,  4096,  256, 20, 100},
        {256,   32000, 256, 10, 50},
        {256,   50000, 256, 10, 50}
    };

    for (const auto& cfg : configs) {
        run_one_config(cfg);
    }

    return 0;
}