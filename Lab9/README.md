# Lab9 - CUDA 矩阵转置

本目录按实验要求初始化，原始实验说明和报告模板保留在 `doc/`。

## 目录

- `src/hello_world.cu`: CUDA Hello World，输入线程块数和二维线程块维度。
- `src/matrix_transpose.cu`: 三种矩阵转置实现：全局内存、共享内存、带 padding 的共享内存。
- `scripts/run_on_server.sh`: 在有 CUDA 的服务器上一键编译并运行实验。
- `scripts/package_code.sh`: 打包源码和脚本，生成 `Lab9_code.zip`。
- `report/main.tex`: 报告入口，来自实验 9 模板。
- `results/`: 服务器运行结果输出目录，本地只保留 `.gitkeep`。

## 服务器运行

```bash
cd Lab9
make clean all
make run-hello
make run-transpose
make benchmark
```

或者直接运行：

```bash
bash scripts/run_on_server.sh
```

脚本会生成：

- `results/hello_world.txt`
- `results/transpose_single_512_16.txt`
- `results/transpose_benchmark.csv`

## 单独运行

CUDA Hello World：

```bash
./build/hello_world <num_blocks> <block_rows> <block_cols>
```

矩阵转置：

```bash
./build/matrix_transpose <N> <block_size> <repeat>
./build/matrix_transpose --benchmark <repeat>
```

实验要求的输入范围：

- `num_blocks`、`block_rows`、`block_cols`: `1..32`
- `N`: `512..2048`
- `block_size`: 建议使用 `8`、`16`、`32`

## 报告

```bash
cd Lab9
make report
```

最终按课程要求将报告 PDF 命名为 `并行程序设计_学号_姓名.pdf`，代码可用 `make package` 生成压缩包。
