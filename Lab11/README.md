# Lab11 - CUDA 卷积计算

本实验实现三种卷积方式：

- 直接卷积：二维线程块映射和一维线性映射。
- im2col + GEMM：将卷积窗口展开为矩阵，再做矩阵乘法。
- cuDNN 卷积：服务器安装 cuDNN 时可启用。

## 构建

默认不链接 cuDNN：

```bash
make clean all
```

如果服务器提供 cuDNN：

```bash
make clean all CUDNN=1
```

## 运行

单组运行：

```bash
./build/convolution <N> [kernel=3] [stride=1] [block=16] [repeat=5]
```

完整 benchmark：

```bash
./build/convolution --benchmark 5 > results/convolution_benchmark.csv
```

使用 cuDNN 时：

```bash
make clean all CUDNN=1
./build/convolution --benchmark 5 > results/convolution_benchmark.csv
```

输出 CSV 包含：

- `block`: 线程块大小对直接卷积的影响。
- `mapping`: 直接卷积中不同任务映射方式的影响。
- `compare`: 直接卷积、im2col 和 cuDNN 的性能对比。

本地若没有 CUDA 环境，只需把本目录同步到服务器后运行上述命令。
