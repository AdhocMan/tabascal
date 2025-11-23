JAXLIB_HOME ?= $(shell python3 -c "import jaxlib; import os; path = os.path.dirname(jaxlib.__file__); print(path)")

ifeq (, $(shell which nvcc 2> /dev/null))
CUDACXX ?= /usr/local/cuda/bin/nvcc
else
CUDACXX ?= nvcc
endif

CUDA_FLAGS ?= -O2 -DNDEBUG \
			  -gencode arch=compute_70,code=compute_70 \
			  -gencode arch=compute_80,code=compute_80 \
			  -gencode arch=compute_90,code=compute_90 \
			  -gencode arch=compute_90a,code=sm_90a

CXX_FLAGS ?= -O2 -DNDEBUG -march=native

all: cpu gpu

cpu: tabascal.so

gpu: tabascal_gpu.so

tabascal.so: module.cpp
	c++ -shared module.cpp -o tabascal.so -I $(JAXLIB_HOME)/include -std=c++17  -fPIC $(CXX_FLAGS)

tabascal_gpu.so: rfi_kernel.cu rfi_jvp_kernel.cu rfi_transpose_kernel.cu
	$(CUDACXX) -shared -Xcompiler -fPIC -o tabascal_gpu.so $^ -I $(JAXLIB_HOME)/include -std=c++17 $(CUDA_FLAGS) -Xcompiler -Wno-attritutes -Xcompiler -Wno-return-type -Xcudafe "--diag_suppress=940 --diag_suppress=2473"
