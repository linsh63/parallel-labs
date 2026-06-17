#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifndef USE_CUDNN
#define USE_CUDNN 0
#endif

#if USE_CUDNN
#include <cudnn.h>
#endif

namespace {

constexpr int kChannels = 3;
constexpr int kDefaultN = 256;
constexpr int kDefaultKernel = 3;
constexpr int kDefaultStride = 1;
constexpr int kDefaultBlock = 16;
constexpr int kDefaultRepeat = 5;
constexpr int kPreview = 6;

struct Times {
    float direct = -1.0f;
    float im2col = -1.0f;
    float cudnn = -1.0f;
    bool im2col_ok = false;
    bool cudnn_ok = false;
};

// 计算向上取整的整数除法。
int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

// 计算卷积输出边长。
int output_size(int n, int kernel, int pad, int stride)
{
    return (n + 2 * pad - kernel) / stride + 1;
}

// 生成固定随机种子的单精度数组。
std::vector<float> make_data(int count, unsigned seed)
{
    std::vector<float> data(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &x : data) {
        x = dist(rng);
    }
    return data;
}

// 读取带 padding 的输入元素，越界位置返回 0。
__device__ float input_at(const float *input, int n, int c, int row, int col)
{
    if (row < 0 || row >= n || col < 0 || col >= n) {
        return 0.0f;
    }
    return input[(c * n + row) * n + col];
}

// 二维线程块映射：每个线程计算一个输出像素。
__global__ void direct_2d_kernel(const float *input, const float *filter,
                                 float *output, int n, int kernel, int pad,
                                 int stride, int out_n)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (oh >= out_n || ow >= out_n) {
        return;
    }

    float sum = 0.0f;
    for (int c = 0; c < kChannels; ++c) {
        for (int r = 0; r < kernel; ++r) {
            for (int s = 0; s < kernel; ++s) {
                int ih = oh * stride + r - pad;
                int iw = ow * stride + s - pad;
                float x = input_at(input, n, c, ih, iw);
                float w = filter[(c * kernel + r) * kernel + s];
                sum += x * w;
            }
        }
    }
    output[oh * out_n + ow] = sum;
}

// 一维线性映射：每个线程按线性编号计算一个输出像素。
__global__ void direct_linear_kernel(const float *input, const float *filter,
                                     float *output, int n, int kernel, int pad,
                                     int stride, int out_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_n * out_n;
    if (idx >= total) {
        return;
    }

    int oh = idx / out_n;
    int ow = idx - oh * out_n;
    float sum = 0.0f;
    for (int c = 0; c < kChannels; ++c) {
        for (int r = 0; r < kernel; ++r) {
            for (int s = 0; s < kernel; ++s) {
                int ih = oh * stride + r - pad;
                int iw = ow * stride + s - pad;
                float x = input_at(input, n, c, ih, iw);
                float w = filter[(c * kernel + r) * kernel + s];
                sum += x * w;
            }
        }
    }
    output[idx] = sum;
}

// 将每个卷积窗口展开为 im2col 矩阵的一行。
__global__ void im2col_kernel(const float *input, float *col, int n, int kernel,
                              int pad, int stride, int out_n)
{
    int col_width = kChannels * kernel * kernel;
    int total = out_n * out_n * col_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int kidx = idx % col_width;
    int pos = idx / col_width;
    int ow = pos % out_n;
    int oh = pos / out_n;
    int s = kidx % kernel;
    int r = (kidx / kernel) % kernel;
    int c = kidx / (kernel * kernel);
    int ih = oh * stride + r - pad;
    int iw = ow * stride + s - pad;
    col[idx] = input_at(input, n, c, ih, iw);
}

// 通用矩阵乘法 C = A x B。
__global__ void gemm_kernel(const float *a, const float *b, float *c, int m,
                            int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

// 启动二维映射直接卷积。
void launch_direct_2d(const float *input, const float *filter, float *output,
                      int n, int kernel, int pad, int stride, int out_n,
                      int block_size)
{
    dim3 block(block_size, block_size);
    dim3 grid(ceil_div(out_n, block.x), ceil_div(out_n, block.y));
    direct_2d_kernel<<<grid, block>>>(input, filter, output, n, kernel, pad,
                                      stride, out_n);
}

// 启动一维映射直接卷积。
void launch_direct_linear(const float *input, const float *filter,
                          float *output, int n, int kernel, int pad,
                          int stride, int out_n, int block_size)
{
    int threads = block_size * block_size;
    int total = out_n * out_n;
    direct_linear_kernel<<<ceil_div(total, threads), threads>>>(
        input, filter, output, n, kernel, pad, stride, out_n);
}

// 启动 im2col 展开和矩阵乘法。
void launch_im2col(const float *input, const float *filter, float *col,
                   float *output, int n, int kernel, int pad, int stride,
                   int out_n)
{
    int rows = out_n * out_n;
    int depth = kChannels * kernel * kernel;
    int total = rows * depth;
    im2col_kernel<<<ceil_div(total, 256), 256>>>(input, col, n, kernel, pad,
                                                 stride, out_n);
    dim3 block(16, 16);
    dim3 grid(1, ceil_div(rows, block.y));
    gemm_kernel<<<grid, block>>>(col, filter, output, rows, 1, depth);
}

// 使用 CUDA event 统计一个执行函数的平均耗时。
template <typename Fn>
float time_cuda(Fn fn, int repeat)
{
    fn();
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        fn();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeat;
}

#if USE_CUDNN
// 使用 cuDNN implicit GEMM 算法执行同参数卷积。
float time_cudnn(const float *input, const float *filter, float *output, int n,
                 int kernel, int pad, int stride, int out_n, int repeat)
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreate(&handle);
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                               kChannels, n, n);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                               1, out_n, out_n);
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1,
                               kChannels, kernel, kernel);
    cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc,
                                            y_desc, algo, &workspace_bytes);
    void *workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float ms = time_cuda(
        [&] {
            cudnnConvolutionForward(handle, &alpha, x_desc, input, w_desc,
                                    filter, conv_desc, algo, workspace,
                                    workspace_bytes, &beta, y_desc, output);
        },
        repeat);

    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(handle);
    return ms;
}
#endif

// 将设备数组复制回主机。
std::vector<float> copy_device(const float *device, int count)
{
    std::vector<float> host(count);
    cudaMemcpy(host.data(), device, static_cast<size_t>(count) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return host;
}

// 判断两个输出矩阵是否近似相等。
bool close_to(const std::vector<float> &a, const std::vector<float> &b)
{
    for (size_t i = 0; i < a.size(); ++i) {
        float eps = 1e-2f * std::max(1.0f, std::fabs(a[i]));
        if (std::fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

// 运行一组卷积参数并返回三种方法的时间。
Times run_case(int n, int kernel, int stride, int block_size,
               const std::string &mapping, int repeat, bool run_im2col,
               bool run_cudnn)
{
    int pad = kernel / 2;
    int out_n = output_size(n, kernel, pad, stride);
    int input_count = kChannels * n * n;
    int filter_count = kChannels * kernel * kernel;
    int output_count = out_n * out_n;
    int col_count = output_count * filter_count;

    std::vector<float> input = make_data(input_count, 20260617 + n + stride);
    std::vector<float> filter = make_data(filter_count, 20260618 + kernel);

    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_direct = nullptr;
    float *d_im2col = nullptr;
    float *d_cudnn = nullptr;
    float *d_col = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_input), input_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_filter),
               filter_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_direct),
               output_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_im2col),
               output_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_cudnn),
               output_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_col), col_count * sizeof(float));
    cudaMemcpy(d_input, input.data(), input_count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter.data(), filter_count * sizeof(float),
               cudaMemcpyHostToDevice);

    Times times;
    if (mapping == "linear") {
        times.direct = time_cuda(
            [&] {
                launch_direct_linear(d_input, d_filter, d_direct, n, kernel,
                                     pad, stride, out_n, block_size);
            },
            repeat);
    } else {
        times.direct = time_cuda(
            [&] {
                launch_direct_2d(d_input, d_filter, d_direct, n, kernel, pad,
                                 stride, out_n, block_size);
            },
            repeat);
    }
    std::vector<float> direct = copy_device(d_direct, output_count);

    if (run_im2col) {
        times.im2col = time_cuda(
            [&] {
                launch_im2col(d_input, d_filter, d_col, d_im2col, n, kernel,
                              pad, stride, out_n);
            },
            repeat);
        times.im2col_ok = close_to(direct, copy_device(d_im2col, output_count));
    }

#if USE_CUDNN
    if (run_cudnn) {
        times.cudnn = time_cudnn(d_input, d_filter, d_cudnn, n, kernel, pad,
                                 stride, out_n, repeat);
        times.cudnn_ok = close_to(direct, copy_device(d_cudnn, output_count));
    }
#else
    (void)run_cudnn;
#endif

    cudaFree(d_col);
    cudaFree(d_cudnn);
    cudaFree(d_im2col);
    cudaFree(d_direct);
    cudaFree(d_filter);
    cudaFree(d_input);
    return times;
}

// 打印左上角输出预览。
void print_preview(const std::vector<float> &data, int n, const char *name)
{
    int limit = std::min(n, kPreview);
    std::cout << name << " " << limit << "x" << limit << " preview:\n"
              << std::fixed << std::setprecision(3);
    for (int r = 0; r < limit; ++r) {
        for (int c = 0; c < limit; ++c) {
            std::cout << std::setw(9) << data[r * n + c];
        }
        std::cout << '\n';
    }
}

// 单组运行时打印输出预览和计时。
void run_single(int n, int kernel, int stride, int block_size, int repeat)
{
    int pad = kernel / 2;
    int out_n = output_size(n, kernel, pad, stride);
    int input_count = kChannels * n * n;
    int filter_count = kChannels * kernel * kernel;
    int output_count = out_n * out_n;

    std::vector<float> input = make_data(input_count, 20260617 + n + stride);
    std::vector<float> filter = make_data(filter_count, 20260618 + kernel);
    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_input), input_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_filter),
               filter_count * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_output),
               output_count * sizeof(float));
    cudaMemcpy(d_input, input.data(), input_count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter.data(), filter_count * sizeof(float),
               cudaMemcpyHostToDevice);

    float ms = time_cuda(
        [&] {
            launch_direct_2d(d_input, d_filter, d_output, n, kernel, pad,
                             stride, out_n, block_size);
        },
        repeat);
    print_preview(copy_device(d_output, output_count), out_n, "Output");
    std::cout << "\nn,kernel,stride,pad,block,out_n,direct_ms\n";
    std::cout << n << ',' << kernel << ',' << stride << ',' << pad << ','
              << block_size << ',' << out_n << ',' << std::fixed
              << std::setprecision(4) << ms << '\n';

    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_input);
}

// 打印 benchmark 的 CSV 表头。
void print_header()
{
    std::cout << "mode,n,kernel,stride,pad,block,mapping,direct_ms,im2col_ms,"
                 "cudnn_ms,valid_im2col,valid_cudnn,out_n\n";
}

// 打印一行 benchmark 结果。
void print_row(const std::string &mode, int n, int kernel, int stride,
               int block, const std::string &mapping, const Times &times)
{
    int pad = kernel / 2;
    int out_n = output_size(n, kernel, pad, stride);
    std::cout << mode << ',' << n << ',' << kernel << ',' << stride << ','
              << pad << ',' << block << ',' << mapping << ',' << std::fixed
              << std::setprecision(4) << times.direct << ',' << times.im2col
              << ',' << times.cudnn << ',' << times.im2col_ok << ','
              << times.cudnn_ok << ',' << out_n << '\n';
}

// 运行报告需要的 benchmark 组合。
void run_benchmark(int repeat)
{
    int kernel = 3;
    print_header();

    for (int block : {8, 16, 32}) {
        Times t = run_case(256, kernel, 1, block, "2d", repeat, false, false);
        print_row("block", 256, kernel, 1, block, "2d", t);
    }

    for (const std::string &mapping : {"2d", "linear"}) {
        Times t = run_case(256, kernel, 1, 16, mapping, repeat, false, false);
        print_row("mapping", 256, kernel, 1, 16, mapping, t);
    }

    for (int n : {32, 64, 128, 256, 512}) {
        for (int stride : {1, 2, 3}) {
            Times t = run_case(n, kernel, stride, 16, "2d", repeat, true,
                               true);
            print_row("compare", n, kernel, stride, 16, "2d", t);
        }
    }
}

// 打印程序用法。
void print_usage(const char *program)
{
    std::cerr << "Usage:\n"
              << "  " << program
              << " <N> [kernel=3] [stride=1] [block=16] [repeat=5]\n"
              << "  " << program << " --benchmark [repeat]\n";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        int repeat = argc >= 3 ? std::atoi(argv[2]) : kDefaultRepeat;
        run_benchmark(std::max(1, repeat));
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    int n = argc >= 2 ? std::atoi(argv[1]) : kDefaultN;
    int kernel = argc >= 3 ? std::atoi(argv[2]) : kDefaultKernel;
    int stride = argc >= 4 ? std::atoi(argv[3]) : kDefaultStride;
    int block = argc >= 5 ? std::atoi(argv[4]) : kDefaultBlock;
    int repeat = argc >= 6 ? std::atoi(argv[5]) : kDefaultRepeat;
    if (n < 32 || kernel < 1 || kernel % 2 == 0 || stride < 1 || block < 1) {
        print_usage(argv[0]);
        return 1;
    }
    run_single(n, kernel, stride, block, std::max(1, repeat));
    return 0;
}
