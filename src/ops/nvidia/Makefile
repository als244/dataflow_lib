NVCC = nvcc
DEV_NVCC_FLAGS = -g -G
BUILD_NVCC_FLAGS = -O4 --use_fast_math
PROF_NVCC_FLAGS = -O4 --use_fast_math --generate-line-info -Xptxas=-v
NVCC_FLAGS = ${PROF_NVCC_FLAGS}


CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB_DEPENDS = -lcuda

CUDA_ARCH_NUM = 86

PROJ_COMMON_HEADER = ../../../include
DATAFLOW_HANDLE_HEADER = ../../dataflow_handle/include
CUDA_DATAFLOW_HANDLE_HEADER = ../../dataflow_handle/nvidia/include
DATATYPE_HEADER = ../../datatype/include
TABLE_HEADER = ../../table/include
FINGERPRINT_HEADER = ../../fingerprint/include
OPS_HEADER = ../include
CUDA_OPS_HEADER = include

GEN_PTX = build/cuda_kernels.ptx
GEN_CUBIN = lib/cuda_kernels.cubin

CUDA_KERNELS_SRC = cuda_kernels.cu

NATIVE_FUNC_SRC_FILES = norm/rms_norm_config.c attention/attn_misc_config.c activations/swiglu_config.c loss/loss_misc_config.c
NATIVE_FUNC_LIB_OBJ_DEPENDS = ../../dataflow_handle/nvidia/objs/cuda_drv.o ../../datatype/objs/dataflow_datatype.o

NATIVE_FUNC_LIB = lib/cuda_kernels_config.so

all: ${NATIVE_FUNC_LIB} ${SAVE_FUNC_EXEC} cuda_kernels_ptx cuda_kernels_cubin

${NATIVE_FUNC_LIB}: ${NATIVE_FUNC_SRC_FILES} ${NATIVE_FUNC_LIB_OBJ_DEPENDS}
	gcc -g -fPIC -shared -I${CUDA_OPS_HEADER} -I${CUDA_DATAFLOW_HANDLE_HEADER} -I${OPS_HEADER} -I${DATAFLOW_HANDLE_HEADER} -I${TABLE_HEADER} -I${FINGERPRINT_HEADER} -I${DATATYPE_HEADER} -I${PROJ_COMMON_HEADER} -I${CUDA_INCLUDE_DIR} $^ -o $@ -L${CUDA_LIB_DIR} ${CUDA_LIB_DEPENDS}

cuda_kernels_ptx: ${CUDA_KERNELS_SRC} 
	nvcc ${NVCC_FLAGS} -I${CUDA_OPS_HEADER} -I${PROJ_COMMON_HEADER} -o ${GEN_PTX} -ptx $^ -arch=compute_${CUDA_ARCH_NUM} -code=sm_${CUDA_ARCH_NUM}

cuda_kernels_cubin: ${CUDA_KERNELS_SRC} 
	nvcc ${NVCC_FLAGS} -I${CUDA_OPS_HEADER} -I${PROJ_COMMON_HEADER} -o ${GEN_CUBIN} -cubin $^ -arch=compute_${CUDA_ARCH_NUM} -code=sm_${CUDA_ARCH_NUM}

clean:
	rm -f build/cuda_all_functions_meta.dat ${NATIVE_FUNC_LIB} ${SAVE_FUNC_EXEC} ${GEN_LIB_NAME}.ptx ${GEN_LIB_NAME}.cubin 

