JAXLIB_HOME ?= $(shell python3 -c "import jaxlib; import os; path = os.path.dirname(jaxlib.__file__); print(path)")

ifeq (, $(shell which nvcc 2> /dev/null))
CUDACXX ?= /usr/local/cuda/bin/nvcc
else
CUDACXX ?= nvcc
endif

CUDA_ARCH ?= 90

all: tab.so tab_gpu.so

tab.so: module.cpp
	c++ -shared module.cpp -o tab.so -I $(JAXLIB_HOME)/include -std=c++17  -march=native -O3 -fPIC

tab_gpu.so: module.cu
	$(CUDACXX) -shared -Xcompiler -fPIC -O2 -o tab_gpu.so module.cu -I $(JAXLIB_HOME)/include -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Xcompiler -Wno-attritutes -Xcompiler -Wno-return-type -Xcudafe "--diag_suppress=940 --diag_suppress=2473"
