JAXLIB_HOME ?= $(shell python3 -c "import jaxlib; import os; path = os.path.dirname(jaxlib.__file__); print(path)")

all: tab.so

tab.so: module.cpp
	c++ -shared module.cpp -o tab.so -I $(JAXLIB_HOME)/include -std=c++17  -march=native -O3 -fPIC
