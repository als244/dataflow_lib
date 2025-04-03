CC = gcc
CFLAGS = -g -fPIC


PROJ_INCLUDE_DIR = ../../include
DATATYPE_INCLUDE_DIR = ../datatype/include
OPS_INCLUDE_DIR = ../ops/include
HANDLE_INCLUDE_DIR = ../dataflow_handle/include
DATAFLOW_OP_INCLUDE_DIR = include
TABLE_INCLUDE_DIR = ../table/include
FINGERPRINT_INCLUDE_DIR = ../fingerprint/include
SET_OPS_INCLUDE_DIR = ../set_ops/include

DATAFLOW_OP_OBJS = objs/matmul_op.o objs/attention_op.o objs/preprocess_ops.o objs/norm_ops.o objs/attn_misc_ops.o objs/mlp_misc_ops.o objs/loss_misc_ops.o 

DATAFLOW_OP_LIB = lib/libdataflowops.so
DATAFLOW_OP_LIB_DEPENDS = -lcrypto -pthread -lm -ldl


DATATYPE_PATH = ../datatype
DATATYPE_OBJ_DEPENDS = ${DATATYPE_PATH}/objs/dataflow_datatype.o ${DATATYPE_PATH}/objs/solo_conversion.o ${DATATYPE_PATH}/objs/convert_datatype.o ${DATATYPE_PATH}/objs/convert_datatype_avx512.o
FINGERPRINT_OBJ_DEPENDS = ../fingerprint/objs/fingerprint.o
SET_OPS_OBJ_DEPENDS = ../set_ops/objs/set_native_op_skeletons.o ../set_ops/objs/set_external_op_skeletons.o

DATAFLOW_OP_LIB_OTHER_OBJ_DEPENDS = ${DATATYPE_OBJ_DEPENDS} ${FINGERPRINT_OBJ_DEPENDS} ${SET_OPS_OBJ_DEPENDS}


all: ${DATAFLOW_OP_LIB} ${DATAFLOW_OP_OBJS}

${DATAFLOW_OP_LIB}: ${DATAFLOW_OP_OBJS} ${DATAFLOW_OP_LIB_OTHER_OBJ_DEPENDS}
	${CC} ${CFLAGS} -shared -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} $^ -o $@ ${DATAFLOW_OP_LIB_DEPENDS} 

objs/matmul_op.o: src/matmul_op.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/attention_op.o: src/attention_op.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/preprocess_ops.o: src/preprocess_ops.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/norm_ops.o: src/norm_ops.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/attn_misc_ops.o: src/attn_misc_ops.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/mlp_misc_ops.o: src/mlp_misc_ops.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<

objs/loss_misc_ops.o: src/loss_misc_ops.c
	${CC} ${CFLAGS} -I${DATAFLOW_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${SET_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<


clean:
	rm -f objs/*.o ${DATAFLOW_OP_LIB}
