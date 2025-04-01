CC = gcc
CFLAGS = -g -fPIC

PROJ_INCLUDE_DIR = ../../include
DATAFLOW_HANDLE_INCLUDE_DIR = ../dataflow_handle/include
DATATYPE_INCLUDE_DIR = ../datatype/include
TABLE_INCLUDE_DIR = ../table/include
FINGERPRINT_INCLUDE_DIR = ../fingerprint/include
OPS_INCLUDE_DIR = ../ops/include
SET_OPS_INCLUDE_DIR = include

CUDA_INCLUDE_DIR = /usr/local/cuda/include

OBJ_DEPENDS = ../fingerprint/objs/fingerprint.o ../datatype/objs/dataflow_datatype.o ../datatype/objs/solo_conversion.o ../datatype/objs/convert_datatype.o ../datatype/objs/convert_datatype_avx512.o
OBJS = objs/set_native_op_skeletons.o objs/set_external_op_skeletons.o

all: ${OBJS}

objs/set_native_op_skeletons.o: src/set_native_op_skeletons.c ${OBJ_DEPENDS}
	${CC} ${CFLAGS}  -I${SET_OPS_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${DATAFLOW_HANDLE_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -c -o $@ $<

objs/set_external_op_skeletons.o: src/set_external_op_skeletons.c ${OBJ_DEPENDS}
	${CC} ${CFLAGS}  -I${SET_OPS_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${DATAFLOW_HANDLE_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -c -o $@ $<

clean:
	rm -f ${OBJS}
