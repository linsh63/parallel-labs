#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kMinDim = 128;
constexpr int kMaxDim = 2048;
constexpr int kDefaultM = 512;
constexpr int kDefaultN = 512;
constexpr int kDefaultK = 512;
constexpr int kDefaultBlockSize = 16;
constexpr int kDefaultRepeat = 5;
constexpr int kBenchmarkRepeat = 1;
constexpr int kPreviewLimit = 6;

enum PartitionKind {
    kRowPartition = 0,
    kColumnPartition = 1,
    kTilePartition = 2,
};

enum KernelKind {
    kNaiveKernel = 0,
    kSharedKernel = 1,
    kRegisterKernel = 2,
};

struct Result {
    int partition = kTilePartition;
    float naive_ms = 0.0f;
    float shared_ms = 0.0f;
    float register_ms = 0.0f;
    bool naive_ok = false;
    bool shared_ok = false;
    bool register_ok = false;
    std::vector<float> c;
};

// 计算向上取整的整数除法结果。
int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

// 根据任务划分方式计算当前线程块对应的输出块起点。
__device__ void output_block_start(int partition, int output_tile, int k,
                                   int *row0, int *col0)
{
    if (partition == kRowPartition) {
        *row0 = blockIdx.x * output_tile;
        *col0 = blockIdx.y * output_tile;
    } else if (partition == kColumnPartition) {
        *row0 = blockIdx.y * output_tile;
        *col0 = blockIdx.x * output_tile;
    } else {
        int tile_cols = (k + output_tile - 1) / output_tile;
        int tile_row = blockIdx.x / tile_cols;
        int tile_col = blockIdx.x - tile_row * tile_cols;
        *row0 = tile_row * output_tile;
        *col0 = tile_col * output_tile;
    }
}

// 每个线程直接在全局内存中计算 C 的一个元素。
__global__ void matmul_naive(const float *a, const float *b, float *c, int m,
                             int n, int k, int partition)
{
    int row0 = 0;
    int col0 = 0;
    int tile = blockDim.x;
    output_block_start(partition, tile, k, &row0, &col0);

    int row = row0 + threadIdx.y;
    int col = col0 + threadIdx.x;
    if (row >= m || col >= k) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
}

// 每个线程块用共享内存缓存 A 和 B 的一个分块后计算。
__global__ void matmul_shared(const float *a, const float *b, float *c, int m,
                              int n, int k, int partition)
{
    extern __shared__ float shared[];
    int tile = blockDim.x;
    float *tile_a = shared;
    float *tile_b = tile_a + tile * tile;

    int row0 = 0;
    int col0 = 0;
    output_block_start(partition, tile, k, &row0, &col0);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = row0 + ty;
    int col = col0 + tx;
    float sum = 0.0f;

    for (int base = 0; base < n; base += tile) {
        int a_col = base + tx;
        int b_row = base + ty;
        tile_a[ty * tile + tx] =
            (row < m && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[ty * tile + tx] =
            (b_row < n && col < k) ? b[b_row * k + col] : 0.0f;

        __syncthreads();
        for (int i = 0; i < tile; ++i) {
            sum += tile_a[ty * tile + i] * tile_b[i * tile + tx];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = sum;
    }
}

// 每个线程用寄存器累加一个 2x2 输出小块。
__global__ void matmul_register_2x2(const float *a, const float *b, float *c,
                                    int m, int n, int k, int partition)
{
    extern __shared__ float shared[];
    int tile = blockDim.x;
    float *tile_a = shared;
    float *tile_b = tile_a + 2 * tile * tile;

    int row_base = 0;
    int col_base = 0;
    output_block_start(partition, 2 * tile, k, &row_base, &col_base);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = row_base + ty;
    int row1 = row0 + tile;
    int col0 = col_base + tx;
    int col1 = col0 + tile;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    for (int base = 0; base < n; base += tile) {
        int a_col = base + tx;
        int b_row = base + ty;
        tile_a[ty * tile + tx] =
            (row0 < m && a_col < n) ? a[row0 * n + a_col] : 0.0f;
        tile_a[(ty + tile) * tile + tx] =
            (row1 < m && a_col < n) ? a[row1 * n + a_col] : 0.0f;
        tile_b[ty * (2 * tile) + tx] =
            (b_row < n && col0 < k) ? b[b_row * k + col0] : 0.0f;
        tile_b[ty * (2 * tile) + tx + tile] =
            (b_row < n && col1 < k) ? b[b_row * k + col1] : 0.0f;

        __syncthreads();
        for (int i = 0; i < tile; ++i) {
            float a0 = tile_a[ty * tile + i];
            float a1 = tile_a[(ty + tile) * tile + i];
            float b0 = tile_b[i * (2 * tile) + tx];
            float b1 = tile_b[i * (2 * tile) + tx + tile];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < m && col0 < k) {
        c[row0 * k + col0] = c00;
    }
    if (row0 < m && col1 < k) {
        c[row0 * k + col1] = c01;
    }
    if (row1 < m && col0 < k) {
        c[row1 * k + col0] = c10;
    }
    if (row1 < m && col1 < k) {
        c[row1 * k + col1] = c11;
    }
}

// 返回任务划分方式的输出名称。
const char *partition_name(int partition)
{
    if (partition == kRowPartition) {
        return "row";
    }
    if (partition == kColumnPartition) {
        return "column";
    }
    return "tile";
}

// 判断矩阵维度是否在实验要求范围内。
bool valid_dim(int value)
{
    return value >= kMinDim && value <= kMaxDim;
}

// 判断线程块大小是否为实验对比使用的配置。
bool valid_block_size(int block_size)
{
    return block_size == 8 || block_size == 16 || block_size == 32;
}

// 生成固定随机种子的矩阵，保证多次运行可复现。
std::vector<float> make_matrix(int rows, int cols, unsigned seed)
{
    std::vector<float> matrix(static_cast<size_t>(rows) * cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &value : matrix) {
        value = dist(rng);
    }
    return matrix;
}

// 在主机端计算 C 的单个元素用于校验。
float cpu_value(const std::vector<float> &a, const std::vector<float> &b,
                int n, int k, int row, int col)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[static_cast<size_t>(row) * n + i] *
               b[static_cast<size_t>(i) * k + col];
    }
    return sum;
}

// 抽样检查设备端输出是否与主机端计算结果一致。
bool validate(const std::vector<float> &a, const std::vector<float> &b,
              const std::vector<float> &c, int m, int n, int k)
{
    int rows[] = {0, m - 1, m / 2, m / 3, (2 * m) / 3};
    int cols[] = {0, k - 1, k / 2, k / 4, (3 * k) / 4};
    for (int i = 0; i < 5; ++i) {
        int row = rows[i];
        int col = cols[i];
        float expected = cpu_value(a, b, n, k, row, col);
        float actual = c[static_cast<size_t>(row) * k + col];
        float eps = 1e-2f * std::max(1.0f, std::fabs(expected));
        if (std::fabs(expected - actual) > eps) {
            return false;
        }
    }
    return true;
}

// 根据划分方式和输出块大小设置 CUDA 网格。
dim3 grid_dim(int partition, int output_tile, int m, int k)
{
    int row_tiles = ceil_div(m, output_tile);
    int col_tiles = ceil_div(k, output_tile);
    if (partition == kRowPartition) {
        return dim3(row_tiles, col_tiles);
    }
    if (partition == kColumnPartition) {
        return dim3(col_tiles, row_tiles);
    }
    return dim3(row_tiles * col_tiles);
}

// 启动指定版本的矩阵乘法核函数。
void launch_kernel(int partition, int kernel, const float *a, const float *b,
                   float *c, int m, int n, int k, int block_size)
{
    dim3 block(block_size, block_size);
    if (kernel == kNaiveKernel) {
        dim3 grid = grid_dim(partition, block_size, m, k);
        matmul_naive<<<grid, block>>>(a, b, c, m, n, k, partition);
    } else if (kernel == kSharedKernel) {
        dim3 grid = grid_dim(partition, block_size, m, k);
        size_t bytes = static_cast<size_t>(2 * block_size * block_size) *
                       sizeof(float);
        matmul_shared<<<grid, block, bytes>>>(a, b, c, m, n, k, partition);
    } else {
        dim3 grid = grid_dim(partition, 2 * block_size, m, k);
        size_t bytes = static_cast<size_t>(4 * block_size * block_size) *
                       sizeof(float);
        matmul_register_2x2<<<grid, block, bytes>>>(a, b, c, m, n, k,
                                                    partition);
    }
}

// 使用 CUDA 事件统计核函数平均运行时间。
float time_kernel(int partition, int kernel, const float *a, const float *b,
                  float *c, int m, int n, int k, int block_size, int repeat)
{
    launch_kernel(partition, kernel, a, b, c, m, n, k, block_size);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        launch_kernel(partition, kernel, a, b, c, m, n, k, block_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeat;
}

// 将设备端的 C 矩阵复制回主机。
std::vector<float> copy_c(const float *device_c, int rows, int cols)
{
    std::vector<float> host_c(static_cast<size_t>(rows) * cols);
    cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    return host_c;
}

// 运行一种任务划分下的三种矩阵乘法实现。
Result run_partition(int partition, const std::vector<float> &a,
                     const std::vector<float> &b, int m, int n, int k,
                     int block_size, int repeat)
{
    size_t a_bytes = a.size() * sizeof(float);
    size_t b_bytes = b.size() * sizeof(float);
    size_t c_bytes = static_cast<size_t>(m) * k * sizeof(float);

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&device_a), a_bytes);
    cudaMalloc(reinterpret_cast<void **>(&device_b), b_bytes);
    cudaMalloc(reinterpret_cast<void **>(&device_c), c_bytes);
    cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice);

    Result result;
    result.partition = partition;

    cudaMemset(device_c, 0, c_bytes);
    result.naive_ms = time_kernel(partition, kNaiveKernel, device_a, device_b,
                                  device_c, m, n, k, block_size, repeat);
    result.c = copy_c(device_c, m, k);
    result.naive_ok = validate(a, b, result.c, m, n, k);

    cudaMemset(device_c, 0, c_bytes);
    result.shared_ms = time_kernel(partition, kSharedKernel, device_a, device_b,
                                   device_c, m, n, k, block_size, repeat);
    result.c = copy_c(device_c, m, k);
    result.shared_ok = validate(a, b, result.c, m, n, k);

    cudaMemset(device_c, 0, c_bytes);
    result.register_ms = time_kernel(partition, kRegisterKernel, device_a,
                                     device_b, device_c, m, n, k, block_size,
                                     repeat);
    result.c = copy_c(device_c, m, k);
    result.register_ok = validate(a, b, result.c, m, n, k);

    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    return result;
}

// 依次运行按行、按列和按数据块三种任务划分。
std::array<Result, 3> run_all_partitions(const std::vector<float> &a,
                                         const std::vector<float> &b, int m,
                                         int n, int k, int block_size,
                                         int repeat)
{
    return {run_partition(kRowPartition, a, b, m, n, k, block_size, repeat),
            run_partition(kColumnPartition, a, b, m, n, k, block_size, repeat),
            run_partition(kTilePartition, a, b, m, n, k, block_size, repeat)};
}

// 打印矩阵左上角预览或完整矩阵。
void print_matrix(const std::vector<float> &matrix, int rows, int cols,
                  const std::string &name, bool full)
{
    int row_limit = full ? rows : std::min(rows, kPreviewLimit);
    int col_limit = full ? cols : std::min(cols, kPreviewLimit);
    std::cout << name << " " << row_limit << "x" << col_limit;
    if (!full) {
        std::cout << " preview";
    }
    std::cout << ":\n" << std::fixed << std::setprecision(3);

    for (int row = 0; row < row_limit; ++row) {
        for (int col = 0; col < col_limit; ++col) {
            std::cout << std::setw(9)
                      << matrix[static_cast<size_t>(row) * cols + col];
        }
        std::cout << '\n';
    }
}

// 打印逗号分隔结果表头。
void print_header()
{
    std::cout << "m,n,k,partition,block_size,repeat,naive_ms,shared_tiled_ms,"
                 "register_2x2_ms,valid_naive,valid_shared_tiled,"
                 "valid_register_2x2\n";
}

// 打印单行逗号分隔实验结果。
void print_result(int m, int n, int k, int block_size, int repeat,
                  const Result &result)
{
    std::cout << m << ',' << n << ',' << k << ','
              << partition_name(result.partition) << ',' << block_size << ','
              << repeat << ',' << std::fixed << std::setprecision(4)
              << result.naive_ms << ',' << result.shared_ms << ','
              << result.register_ms << ',' << result.naive_ok << ','
              << result.shared_ok << ',' << result.register_ok << '\n';
}

// 运行一组参数并输出矩阵预览与计时结果。
void run_single(int m, int n, int k, int block_size, int repeat, bool full)
{
    std::vector<float> a = make_matrix(m, n, 20250616);
    std::vector<float> b = make_matrix(n, k, 20250617);
    std::array<Result, 3> results =
        run_all_partitions(a, b, m, n, k, block_size, repeat);

    print_matrix(a, m, n, "A", full);
    std::cout << '\n';
    print_matrix(b, n, k, "B", full);
    std::cout << '\n';
    print_matrix(results[2].c, m, k, "C", full);

    std::cout << "\nTiming average over " << repeat << " run(s):\n";
    print_header();
    for (const Result &result : results) {
        print_result(m, n, k, block_size, repeat, result);
    }
}

// 运行报告表格需要的矩阵规模与线程块大小组合。
void run_benchmark(int repeat)
{
    int sizes[] = {512, 1024, 2048};
    int block_sizes[] = {8, 16, 32};

    print_header();
    for (int size : sizes) {
        std::vector<float> a = make_matrix(size, size, 20250616 + size);
        std::vector<float> b = make_matrix(size, size, 20250617 + size);
        for (int block_size : block_sizes) {
            std::array<Result, 3> results =
                run_all_partitions(a, b, size, size, size, block_size, repeat);
            for (const Result &result : results) {
                print_result(size, size, size, block_size, repeat, result);
            }
        }
    }
}

// 输出程序用法。
void print_usage(const char *program)
{
    std::cerr << "Usage:\n"
              << "  " << program
              << " <M> <N> <K> [block_size] [repeat] [--print-full]\n"
              << "  " << program << " --benchmark [repeat]\n";
}

} // 匿名命名空间

// 解析命令行参数并选择单组运行或完整 benchmark。
int main(int argc, char **argv)
{
    if (argc >= 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        int repeat = argc >= 3 ? std::atoi(argv[2]) : kBenchmarkRepeat;
        run_benchmark(std::max(1, repeat));
        return 0;
    }

    bool full = false;
    std::vector<int> nums;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--print-full") {
            full = true;
        } else {
            nums.push_back(std::atoi(argv[i]));
        }
    }

    int m = nums.size() > 0 ? nums[0] : kDefaultM;
    int n = nums.size() > 1 ? nums[1] : kDefaultN;
    int k = nums.size() > 2 ? nums[2] : kDefaultK;
    int block_size = nums.size() > 3 ? nums[3] : kDefaultBlockSize;
    int repeat = nums.size() > 4 ? nums[4] : kDefaultRepeat;

    if (!valid_dim(m) || !valid_dim(n) || !valid_dim(k) ||
        !valid_block_size(block_size) || repeat < 1) {
        print_usage(argv[0]);
        return 1;
    }

    run_single(m, n, k, block_size, repeat, full);
    return 0;
}
