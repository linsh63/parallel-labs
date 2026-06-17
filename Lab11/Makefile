NVCC ?= nvcc
BUILD_DIR ?= build
CUDNN ?= 0
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo

TARGET := $(BUILD_DIR)/convolution
CUDA_DEFS := -DUSE_CUDNN=$(CUDNN)
CUDA_LIBS :=

ifeq ($(CUDNN),1)
CUDA_LIBS += -lcudnn
endif

.PHONY: all clean run benchmark report

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): src/convolution.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_DEFS) $< -o $@ $(CUDA_LIBS)

run: $(TARGET)
	$(TARGET) 256 3 1 16 5

benchmark: $(TARGET)
	mkdir -p results
	$(TARGET) --benchmark 5 > results/convolution_benchmark.csv

report:
	latexmk -xelatex -cd report/main.tex

clean:
	rm -rf $(BUILD_DIR)
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C -cd report/main.tex; \
	fi
