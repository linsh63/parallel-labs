#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));              \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

__global__ void hello_kernel()
{
    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdx.y,
           threadIdx.x, blockIdx.x);
}

static bool parse_int(const char *text, int *value)
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

static bool in_range(int value)
{
    return value >= 1 && value <= 32;
}

static void print_usage(const char *program)
{
    std::fprintf(stderr,
                 "Usage: %s <num_blocks> <block_rows> <block_cols>\n"
                 "All inputs must be integers in [1, 32].\n",
                 program);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int num_blocks = 0;
    int block_rows = 0;
    int block_cols = 0;
    if (!parse_int(argv[1], &num_blocks) || !parse_int(argv[2], &block_rows) ||
        !parse_int(argv[3], &block_cols) || !in_range(num_blocks) ||
        !in_range(block_rows) || !in_range(block_cols)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::printf("Hello World from the host!\n");
    std::fflush(stdout);

    dim3 grid_dim(num_blocks);
    dim3 block_dim(block_cols, block_rows);
    hello_kernel<<<grid_dim, block_dim>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return EXIT_SUCCESS;
}
