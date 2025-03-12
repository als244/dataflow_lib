CC = gcc
CFLAGS = -g

PROJ_INCLUDE_DIR = ../../include
DATATYPE_INCLUDE_DIR = ../datatype/include
FINGERPRINT_INCLUDE_DIR = ../fingerprint/include
TABLE_INCLUDE_DIR = ../table/include
DATAFLOW_HANDLE_INCLUDE_DIR = ../dataflow_handle/include
OPS_INCLUDE_DIR = ../ops/include
SET_NATIVE_OPS_INCLUDE_DIR = ../set_native_ops/include


SAVE_FUNC_EXEC = save_func_meta
SAVE_FUNC_OBJ_DEPENDS = ../fingerprint/objs/fingerprint.o ../datatype/objs/dataflow_datatype.o ../datatype/objs/convert_datatype.o ../datatype/objs/convert_datatype_avx512.o ../set_native_ops/objs/set_native_op_skeletons.o
SAVE_FUNC_LIB_DEPENDS = -lcrypto


all: ${SAVE_FUNC_EXEC}

${SAVE_FUNC_EXEC}: save_func_meta.c ${SAVE_FUNC_OBJ_DEPENDS}
	${CC} ${CFLAGS} -I${OPS_INCLUDE_DIR} -I${SET_NATIVE_OPS_INCLUDE_DIR} -I${DATAFLOW_HANDLE_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} $^ -o $@ ${SAVE_FUNC_LIB_DEPENDS}


clean:
	rm -f ${SAVE_FUNC_EXEC}


