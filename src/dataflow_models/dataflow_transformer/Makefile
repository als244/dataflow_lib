CC = gcc
CFLAGS = -g -fPIC

DATATYPE_PATH = ../../datatype
DATATYPE_INCLUDE_DIR = ${DATATYPE_PATH}/include
DATATYPE_OBJ_DEPENDS = ${DATATYPE_PATH}/objs/dataflow_datatype.o ${DATATYPE_PATH}/objs/solo_conversion.o ${DATATYPE_PATH}/objs/convert_datatype.o ${DATATYPE_PATH}/objs/convert_datatype_avx512.o

DATAFLOW_OP_INCLUDE_DIR = ${CURDIR}/../../dataflow_ops/include
DATAFLOW_OP_LIB_DIR = ${CURDIR}/../../dataflow_ops/lib
DATAFLOW_OP_LIB = -ldataflowops


DATAFLOW_HANDLE_INCLUDE_DIR = ../../dataflow_handle/include


DATAFLOW_MODELS_INCLUDE_DIR = ../include

DATAFLOW_TRANSFORMER_INCLUDE_DIR = include

DATAFLOW_INCLUDE_DIR = ../../../include

TABLE_INCLUDE_DIR = ../../table/include

OPS_INCLUDE_DIR = ../../ops/include

SET_OPS_INCLUDE_DIR = ../../set_ops/include

SET_OPS_OBJ_DEPENDS = ../../set_ops/objs/set_native_op_skeletons.o ../../set_ops/objs/set_external_op_skeletons.o

FINGERPRINT_INCLUDE_DIR = ../../fingerprint/include

FINGERPRINT_OBJ_DEPENDS = ../../fingerprint/objs/fingerprint.o

FINGERPRINT_LIB_DEPENDS = -lcrypto

ALL_INCLUDES = -I${TABLE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${DATAFLOW_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${DATAFLOW_HANDLE_INCLUDE_DIR} -I${DATAFLOW_OP_INCLUDE_DIR} -I${DATAFLOW_MODELS_INCLUDE_DIR} -I${DATAFLOW_TRANSFORMER_INCLUDE_DIR}

#ALL_LIB_DEPENDS = -L${DATAFLOW_OP_LIB_DIR} ${DATAFLOW_OP_LIB} -Wl,-rpath,${DATAFLOW_OP_LIB_DIR}
ALL_LIB_DEPENDS = -lm -ldl -pthread ${FINGERPRINT_LIB_DEPENDS} -L${DATAFLOW_OP_LIB_DIR} ${DATAFLOW_OP_LIB} -Wl,-rpath,${DATAFLOW_OP_LIB_DIR}

all: test/test_transformer

test/test_transformer: test/test_transformer.c objs/dataflow_transformer.o
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIB_DEPENDS}


objs/dataflow_transformer.o: src/dataflow_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

clean:
	rm -f test/test_transformer objs/dataflow_transformer.o
