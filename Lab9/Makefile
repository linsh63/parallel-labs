NVCC ?= nvcc
BUILD_DIR ?= build
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo

HELLO := $(BUILD_DIR)/hello_world
TRANSPOSE := $(BUILD_DIR)/matrix_transpose

.PHONY: all clean run-hello run-transpose benchmark report package

all: $(HELLO) $(TRANSPOSE)

$(BUILD_DIR):
	mkdir -p $@

$(HELLO): src/hello_world.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(TRANSPOSE): src/matrix_transpose.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-hello: $(HELLO)
	$(HELLO) 12 4 4

run-transpose: $(TRANSPOSE)
	$(TRANSPOSE) 1024 16 20

benchmark: $(TRANSPOSE)
	$(TRANSPOSE) --benchmark 20

report:
	latexmk -xelatex -cd report/main.tex

package:
	scripts/package_code.sh

clean:
	rm -rf $(BUILD_DIR)
	latexmk -C -cd report/main.tex
