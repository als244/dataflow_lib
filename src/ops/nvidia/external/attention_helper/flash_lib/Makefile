
CXX = g++
CXX_FLAGS = -O3 -shared -fPIC

FLASH_INCLUDE_DIR = /home/as1669/flash-attention/hopper
FLASH_API_OBJ = /home/as1669/flash-attention/hopper/build/temp.linux-x86_64-cpython-312/flash_api.o
FLASH_FWD_COMBINE_OBJ = /home/as1669/flash-attention/hopper/build/temp.linux-x86_64-cpython-312/flash_fwd_combine.o

FLASH_KERNEL_OBJS_DIR = /home/as1669/flash-attention/hopper/build/temp.linux-x86_64-cpython-312/instantiations

OTHER_LIBS = -lm -ldl

all: libflashattn.so

libflashattn.so: flash_wrapper.cpp ${FLASH_API_OBJ} ${FLASH_FWD_COMBINE_OBJ} $(wildcard ${FLASH_KERNEL_OBJS_DIR}/*.o)
	${CXX} ${CXX_FLAGS} $^ -o $@ -I${FLASH_INCLUDE_DIR} ${OTHER_LIBS}

