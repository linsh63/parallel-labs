#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kMinN = 512;
constexpr int kMaxN = 2048;
constexpr int kMaxBlockSize = 32;
constexpr int kDefaultRepeat = 20;
constexpr int kPreviewLimit = 6;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                      << ": " << cudaGetErrorString(err__) << '\n';           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

enum class KernelKind {
    Global,
    Shared,
    SharedPadded,
};

struct ExperimentResult {
    float global_ms = 0.0f;
    float shared_ms = 0.0f;
    float padded_ms = 0.0f;
    bool global_ok = false;
    bool shared_ok = false;
    bool padded_ok = false;
    std::vector<float> last_output;
};

__global__ void transpose_global_kernel(const float *input, float *output, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void transpose_shared_kernel(const float *input, float *output, int n)
{
    __shared__ float tile[kMaxBlockSize][kMaxBlockSize];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    __syncthreads();

    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;

    if (row < n && col < n) {
        output[row * n + col] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_shared_padded_kernel(const float *input, float *output,
                                               int n)
{
    __shared__ float tile[kMaxBlockSize][kMaxBlockSize + 1];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    __syncthreads();

    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;

    if (row < n && col < n) {
        output[row * n + col] = tile[threadIdx.x][threadIdx.y];
    }
}

bool parse_int(const char *text, int *value)
{
    errno = 0;
    char *end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT_MIN ||
        parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

void print_usage(const char *program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program << " <N> <block_size> [repeat]\n"
        << "  " << program << " --benchmark [repeat]\n"
        << "Constraints:\n"
        << "  N in [512, 2048], block_size in {8, 16, 32}, repeat >= 1\n";
}

bool is_valid_problem_size(int n)
{
    return n >= kMinN && n <= kMaxN;
}

bool is_valid_block_size(int block_size)
{
    return block_size == 8 || block_size == 16 || block_size == 32;
}

std::vector<float> make_matrix(int n)
{
    std::vector<float> matrix(static_cast<size_t>(n) * n);
    std::mt19937 rng(20250611);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (float &value : matrix) {
        value = dist(rng);
    }
    return matrix;
}

bool validate_transpose(const std::vector<float> &input,
                        const std::vector<float> &output, int n)
{
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float expected = input[static_cast<size_t>(row) * n + col];
            float actual = output[static_cast<size_t>(col) * n + row];
            if (std::fabs(expected - actual) > 1e-6f) {
                std::cerr << "Mismatch at input(" << row << ", " << col
                          << ") -> output(" << col << ", " << row
                          << "): expected " << expected << ", got " << actual
                          << '\n';
                return false;
            }
        }
    }
    return true;
}

void launch_kernel(KernelKind kind, const float *d_input, float *d_output, int n,
                   dim3 grid_dim, dim3 block_dim)
{
    switch (kind) {
    case KernelKind::Global:
        transpose_global_kernel<<<grid_dim, block_dim>>>(d_input, d_output, n);
        break;
    case KernelKind::Shared:
        transpose_shared_kernel<<<grid_dim, block_dim>>>(d_input, d_output, n);
        break;
    case KernelKind::SharedPadded:
        transpose_shared_padded_kernel<<<grid_dim, block_dim>>>(d_input,
                                                                d_output, n);
        break;
    }
}

float time_kernel(KernelKind kind, const float *d_input, float *d_output, int n,
                  int block_size, int repeat)
{
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((n + block_size - 1) / block_size,
                  (n + block_size - 1) / block_size);

    launch_kernel(kind, d_input, d_output, n, grid_dim, block_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        launch_kernel(kind, d_input, d_output, n, grid_dim, block_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(repeat);
}

std::vector<float> copy_device_output(const float *d_output, int n)
{
    std::vector<float> output(static_cast<size_t>(n) * n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                          output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return output;
}

ExperimentResult run_experiment(const std::vector<float> &input, int n,
                                int block_size, int repeat)
{
    const size_t bytes = input.size() * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    ExperimentResult result;

    result.global_ms =
        time_kernel(KernelKind::Global, d_input, d_output, n, block_size, repeat);
    result.last_output = copy_device_output(d_output, n);
    result.global_ok = validate_transpose(input, result.last_output, n);

    result.shared_ms =
        time_kernel(KernelKind::Shared, d_input, d_output, n, block_size, repeat);
    result.last_output = copy_device_output(d_output, n);
    result.shared_ok = validate_transpose(input, result.last_output, n);

    result.padded_ms = time_kernel(KernelKind::SharedPadded, d_input, d_output,
                                   n, block_size, repeat);
    result.last_output = copy_device_output(d_output, n);
    result.padded_ok = validate_transpose(input, result.last_output, n);

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    return result;
}

void print_matrix_preview(const std::vector<float> &matrix, int n,
                          const std::string &name)
{
    int limit = std::min(n, kPreviewLimit);
    std::cout << name << " top-left " << limit << "x" << limit << ":\n";
    std::cout << std::fixed << std::setprecision(2);
    for (int row = 0; row < limit; ++row) {
        for (int col = 0; col < limit; ++col) {
            std::cout << std::setw(9)
                      << matrix[static_cast<size_t>(row) * n + col];
        }
        std::cout << '\n';
    }
}

void print_single_result(int n, int block_size, int repeat,
                         const ExperimentResult &result)
{
    std::cout << "\nTiming average over " << repeat << " run(s):\n";
    std::cout << "global_memory_ms,shared_memory_ms,shared_padded_ms\n";
    std::cout << std::fixed << std::setprecision(4) << result.global_ms << ','
              << result.shared_ms << ',' << result.padded_ms << '\n';
    std::cout << "valid_global,valid_shared,valid_shared_padded\n";
    std::cout << result.global_ok << ',' << result.shared_ok << ','
              << result.padded_ok << "\n";
    std::cout << "N=" << n << ", block_size=" << block_size << '\n';
}

void run_single(int n, int block_size, int repeat)
{
    std::vector<float> input = make_matrix(n);
    ExperimentResult result = run_experiment(input, n, block_size, repeat);

    print_matrix_preview(input, n, "A");
    std::cout << '\n';
    print_matrix_preview(result.last_output, n, "A^T");
    print_single_result(n, block_size, repeat, result);
}

void run_benchmark(int repeat)
{
    const int sizes[] = {512, 1024, 2048};
    const int block_sizes[] = {8, 16, 32};

    std::cout << "n,block_size,repeat,global_memory_ms,shared_memory_ms,"
                 "shared_padded_ms,valid_global,valid_shared,valid_shared_padded\n";

    for (int n : sizes) {
        std::vector<float> input = make_matrix(n);
        for (int block_size : block_sizes) {
            ExperimentResult result = run_experiment(input, n, block_size, repeat);
            std::cout << n << ',' << block_size << ',' << repeat << ','
                      << std::fixed << std::setprecision(4) << result.global_ms
                      << ',' << result.shared_ms << ',' << result.padded_ms
                      << ',' << result.global_ok << ',' << result.shared_ok
                      << ',' << result.padded_ok << '\n';
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        int repeat = kDefaultRepeat;
        if (argc >= 3 && !parse_int(argv[2], &repeat)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (argc > 3 || repeat < 1) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        run_benchmark(repeat);
        return EXIT_SUCCESS;
    }

    int n = 1024;
    int block_size = 16;
    int repeat = kDefaultRepeat;

    if (argc >= 2 && !parse_int(argv[1], &n)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 3 && !parse_int(argv[2], &block_size)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 4 && !parse_int(argv[3], &repeat)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 4 || !is_valid_problem_size(n) ||
        !is_valid_block_size(block_size) || repeat < 1) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    run_single(n, block_size, repeat);
    return EXIT_SUCCESS;
}
