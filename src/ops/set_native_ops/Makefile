CC = gcc
CFLAGS = -g

PROJ_INCLUDE_DIR = ../../../../include
DATAFLOW_HANDLE_INCLUDE_DIR = ../../../dataflow_handle/include
CUDA_DATAFLOW_HANDLE_INCLUDE_DIR = ../../../dataflow_handle/nvidia/include
DATATYPE_INCLUDE_DIR = ../../../datatype/include
TABLE_INCLUDE_DIR = ../../../table/include
FINGERPRINT_INCLUDE_DIR = ../../../fingerprint/include
OPS_INCLUDE_DIR = ../../include
SET_NATIVE_OPS_INCLUDE_DIR = include

CUDA_INCLUDE_DIR = /usr/local/cuda/include

OBJ_DEPENDS = ../../../fingerprint/objs/fingerprint.o ../../../datatype/objs/dataflow_datatype.o ../../../datatype/objs/convert_datatype.o ../../../datatype/objs/convert_datatype_avx512.o
OBJS = objs/set_native_op_skeletons.o

all: ${OBJS}

objs/set_native_op_skeletons.o: src/set_native_op_skeletons.c ${OBJ_DEPENDS}
	${CC} ${CFLAGS}  -I${SET_NATIVE_OPS_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${DATAFLOW_HANDLE_INCLUDE_DIR} -I${CUDA_DATAFLOW_HANDLE_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -c -o $@ $<


clean:
	rm -f objs/set_native_op_skeletons.o
