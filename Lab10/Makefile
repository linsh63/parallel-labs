NVCC ?= nvcc
BUILD_DIR ?= build
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo

MATRIX_MUL := $(BUILD_DIR)/matrix_multiply

.PHONY: all clean run benchmark report package

all: $(MATRIX_MUL)

$(BUILD_DIR):
	mkdir -p $@

$(MATRIX_MUL): src/matrix_multiply.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(MATRIX_MUL)
	$(MATRIX_MUL) 512 512 512 16 5

benchmark: $(MATRIX_MUL)
	$(MATRIX_MUL) --benchmark 1

report:
	latexmk -xelatex -cd report/main.tex

package:
	scripts/package_code.sh

clean:
	rm -rf $(BUILD_DIR)
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C -cd report/main.tex; \
	fi
