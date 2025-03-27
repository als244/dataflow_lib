CC = gcc
CFLAGS = -g -fPIC -shared



PROJ_COMMON_HEADER = ../../../../../include
DATAFLOW_HANDLE_HEADER = ../../../../dataflow_handle/include
CUDA_DATAFLOW_HANDLE_HEADER = ../../../../dataflow_handle/nvidia/include
DATATYPE_HEADER = ../../../../datatype/include
TABLE_HEADER = ../../../../table/include
FINGERPRINT_HEADER = ../../../../fingerprint/include
OPS_HEADER = ../../../include
ATTENTION_HELPER_HEADER = include

ATTENTION_HELPER_LIB_DIR = ./lib
ATTENTION_HELPER_LIB = ${ATTENTION_HELPER_LIB_DIR}/libattentionwrapper.so

FLASH3_INCLUDE_DIR = flash3/include
FLASH3_LIB_DIR = flash3/lib
FLASH3_LIB = -lflash3

ATTENTION_HELPER_SRC = src/attention_helper.c

all: ${ATTENTION_HELPER_LIB}

${ATTENTION_HELPER_LIB}: ${ATTENTION_HELPER_SRC}
	${CC} ${CFLAGS} -I${ATTENTION_HELPER_HEADER} -I${FLASH3_INCLUDE_DIR} -I${CUDA_DATAFLOW_HANDLE_HEADER} -I${OPS_HEADER} -I${DATAFLOW_HANDLE_HEADER} -I${TABLE_HEADER} -I${FINGERPRINT_HEADER} -I${DATATYPE_HEADER} -I${PROJ_COMMON_HEADER} -I${CUDA_INCLUDE_DIR} $^ -o $@ -L${FLASH3_LIB_DIR} ${FLASH3_LIB} -Wl,-rpath,${FLASH3_LIB_DIR}

clean:
	rm -f ${ATTENTION_HELPER_LIB}