# Lab10 - CUDA 并行矩阵乘法

本目录按实验 10 要求初始化，原始实验说明、CUBLAS 参考资料和报告模板保留在 `doc/`。

## 目录

- `src/matrix_multiply.cu`: CUDA 通用矩阵乘法起始实现，输入矩阵维度 `M N K`。
- `scripts/run_on_server.sh`: 在有 CUDA 的服务器上一键编译并运行单组实验和 benchmark。
- `scripts/package_code.sh`: 打包源码和脚本，生成 `Lab10_code.zip`。
- `report/main.tex`: 报告入口，来自实验 10 模板。
- `results/`: 服务器运行结果输出目录，本地只保留 `.gitkeep`。

## 服务器运行

```bash
cd Lab10
make clean all
make run
make benchmark
```

或者直接运行：

```bash
bash scripts/run_on_server.sh
```

脚本会生成：

- `results/matmul_single_512_16.txt`
- `results/matmul_benchmark.csv`

## 单独运行

```bash
./build/matrix_multiply <M> <N> <K> [block_size] [repeat] [--print-full]
./build/matrix_multiply --benchmark [repeat]
```

实验要求的输入范围：

- `M`、`N`、`K`: `128..2048`
- `block_size`: 建议使用 `8`、`16`、`32`
- `repeat`: CUDA event 计时重复次数，默认 `5`

程序当前提供三种实现，便于后续扩展和写报告时对比：

- `naive`: 每个线程计算 `C` 的一个元素，直接访问全局内存。
- `shared_tiled`: 使用共享内存按 tile 分块计算。
- `register_2x2`: 每个线程用寄存器计算 `2x2` 输出块，并配合共享内存 tile。

每种实现都会在三种任务/数据划分方式下运行，输出字段 `partition` 对应报告里的三张表：

- `row`: 按行方向组织 grid，优先比较行方向任务划分。
- `column`: 按列方向组织 grid，优先比较列方向任务划分。
- `tile`: 将输出矩阵按数据块展开成 tile 列表，比较数据块划分。

默认单次运行只打印三个矩阵左上角 `6x6` 预览和计时结果；如需完整输出矩阵，可在命令末尾追加 `--print-full`。完整 benchmark 默认重复 1 次，避免一次服务器运行耗时过长；写报告前可手动提高重复次数，例如：

```bash
./build/matrix_multiply --benchmark 3
```

## 报告

```bash
cd Lab10
make report
```

最终按课程要求将报告 PDF 命名为 `并行程序设计_学号_姓名.pdf`，代码可用 `make package` 生成压缩包。
