CC = gcc
CFLAGS = -O3


PROJ_INCLUDE_DIR = ../../include
DATATYPE_INCLUDE_DIR = ../datatype/include
OPS_INCLUDE_DIR = ../ops/include
HANDLE_INCLUDE_DIR = ../dataflow_handle/include
TRANSFORMER_OP_INCLUDE_DIR = include
TABLE_INCLUDE_DIR = ../table/include
FINGERPRINT_INCLUDE_DIR = ../fingerprint/include
NATIVE_OPS_INCLUDE_DIR = ../ops/set_native_ops/include

TRANSFORMER_OP_OBJS = objs/norm_ops.o objs/attn_misc_ops.o objs/mlp_misc_ops.o objs/loss_misc_ops.o 


all: ${TRANSFORMER_OP_OBJS}

objs/norm_ops.o: src/norm_ops.c
	${CC} ${CFLAGS} -I${TRANSFORMER_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${NATIVE_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<


objs/attn_misc_ops.o: src/attn_misc_ops.c
	${CC} ${CFLAGS} -I${TRANSFORMER_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${NATIVE_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<


objs/mlp_misc_ops.o: src/mlp_misc_ops.c
	${CC} ${CFLAGS} -I${TRANSFORMER_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${NATIVE_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<


objs/loss_misc_ops.o: src/loss_misc_ops.c
	${CC} ${CFLAGS} -I${TRANSFORMER_OP_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${NATIVE_OPS_INCLUDE_DIR} -I${HANDLE_INCLUDE_DIR} -I${FINGERPRINT_INCLUDE_DIR} -I${TABLE_INCLUDE_DIR} -c -o $@ $<


clean:
	rm -f objs/*.o
