
CC = gcc
C_FLAGS = -g


ALL_INCLUDE_DIR = -isystem /home/as1669/.conda/envs/flash/include  -I/home/as1669/flash-attention/hopper -I/home/as1669/flash-attention/csrc/cutlass/include -I/home/as1669/.conda/envs/flash/lib/python3.12/site-packages/torch/include -I/home/as1669/.conda/envs/flash/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.8/include -I/home/as1669/.conda/envs/flash/include/python3.12

PYTHON_LIB_DIR = /home/as1669/.conda/envs/flash/lib
PYTHON_LIB = -lpython3

TORCH_LIB_DIR = /home/as1669/.conda/envs/flash/lib/python3.12/site-packages/torch/lib

TORCH_LIBS = -ltorch -ltorch_cpu -ltorch_python -ltorch_cuda -ltorch_global_deps -lc10 -lc10_cuda -lcaffe2_nvrtc

FLASH_LIB_DIR = /home/as1669/flash-attention/hopper/build/lib.linux-x86_64-cpython-312 

MY_FLASH_INCLUDE_DIR = /home/as1669/ml_dataflow/flash_lib
MY_FLASH_LIB_DIR = /home/as1669/ml_dataflow/flash_lib
MY_FLASH_LIB = -lflashattn



OTHER_LIBS = -lcudart -lcuda -ldl -lpython3

all: test_flash_lib

test_flash_lib: test_flash_lib.c
	${CC} ${C_FLAGS} -o $@ $^ ${ALL_INCLUDE_DIR} -I${MY_FLASH_INCLUDE_DIR} ${OTHER_LIBS} -L${PYTHON_LIB_DIR} ${PYTHON_LIB} -L${TORCH_LIB_DIR} ${TORCH_LIBS} -L${FLASH_LIB_DIR} -L${MY_FLASH_LIB_DIR} ${MY_FLASH_LIB}

