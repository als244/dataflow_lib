## C++ Compiler
CXX = g++ -std=c++17
CXX_FLAGS = -fPIC -O3 -Wall

## NVCC Compilation
## SYSTEM-DEPENDENT
## (may be in weird location if python installation of CUDA
## within some environment...)
NVCC = nvcc
NVCC_FLAGS = -Xcompiler=-fPIC -O3 -std=c++17 --ftemplate-backtrace-limit=0 --use_fast_math --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG

## Flags for shared library
WRAPPER_LIB_DEPENDS = -pthread -lm -ldl

## SYSTEM-DEPENDENT PATHS
CUDA_INCLUDE_DIR = /usr/local/cuda/include
## required for cudaGetDevice/cudaSetAttribute/cudaGetError, etc...
## all embedded within various files

## using -rpath to hardcode library path so clients
## do not need to be aware of this linkage
CUDA_RUNTIME_LIB_PATH = /usr/local/cuda/lib64/libcudadrt.so




## Paths Internal to Source Code
FLASH3_INCLUDE_DIR = src
CUTLASS_CUTE_INCLUDE_DIR = ../include


ALL_INCLUDES = -I${FLASH3_INCLUDE_DIR} -I${CUTLASS_CUTE_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}

ALL_LIBS = ${WRAPPER_LIB_DEPENDS} -Wl,--enable-new-dtags -Wl,-rpath=${CUDA_RUNTIME_LIB_PATH}


### NOTE: FLASH3 IS ONLY FOR >= SM90!!!

## suffixes from cuda instantiation files
ARCH_VERSIONS = sm90

## for gencode arch/compute corresponding to arch versions from files
CUDA_ARCH_VERSIONS = sm_90a


## for each ARCH VERSION DO THIS
CU_KERNEL_SOURCES = ${wildcard src/instantiations/sm90/*.cu}

## create same base filename in different directory
CU_KERNEL_OBJS = ${patsubst src/instantiations/sm90/%.cu, build/instantiations/sm90/%.o, ${CU_KERNEL_SOURCES}}

CU_COMBINE_SRC = src/flash_fwd_combine.cu
CU_COMBINE_OBJ = build/flash_fwd_combine.o

FLASH_API_SRC = src/flash_api.cpp
FLASH_API_OBJ = build/flash_api.o

WRAPPER_SHARED_LIB = build/libflash3.so


all: ${WRAPPER_SHARED_LIB}

${WRAPPER_SHARED_LIB}: ${FLASH_API_OBJ} ${CU_COMBINE_OBJ} ${CU_KERNEL_OBJS}
	${CXX} ${CXX_FLAGS} -shared ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

${FLASH_API_OBJ}: ${FLASH_API_SRC}
	${CXX} ${CXX_FLAGS} ${ALL_INCLUDES} -c -o $@ $<

${CU_COMBINE_OBJ}: ${CU_COMBINE_SRC}
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_90a -code=sm_90a $< -c -o $@

build/instantiations/sm90/%.o: src/instantiations/sm90/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_90a -code=sm_90a $< -c -o $@ 


clean:
	rm -f ${WRAPPER_SHARED_LIB} ${FLASH_API_OBJ} ${CU_COMBINE_OBJ} ${CU_KERNEL_OBJS}