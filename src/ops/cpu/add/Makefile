CC = gcc
CFLAGS = -O3 -fPIC

AVX2_INTRIN_FLAGS = -mavx2 -mf16c
AVX512_INTRIN_FLAGS = -mavx512f -mavx512fp16 -mavx512bf16

ADD_INCLUDE_DIR = include
OPS_INCLUDE_DIR = ../../include
PROJ_INCLUDE_DIR = ../../../../include

DATATYPE_PATH = ../../../datatype
DATATYPE_INCLUDE_DIR = ${DATATYPE_PATH}/include
DATATYPE_OBJ_DEPENDS = ${DATATYPE_PATH}/objs/dataflow_datatype.o ${DATATYPE_PATH}/objs/convert_datatype.o ${DATATYPE_PATH}/objs/convert_datatype_avx512.o

LIBS = -pthread


test/test_add: test/test_add.c objs/cpu_add_avx2.o objs/cpu_add_avx512.o ${DATATYPE_OBJ_DEPENDS}
	${CC} ${CFLAGS} -I${ADD_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} $^ -o $@ ${LIBS} 

objs/cpu_add_avx2.o: src/cpu_add_avx2.c
	${CC} ${CFLAGS} ${AVX2_INTRIN_FLAGS} -I${ADD_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR}  -c -o $@ $<

objs/cpu_add_avx512.o: src/cpu_add_avx512.c
	${CC} ${CFLAGS} ${AVX512_INTRIN_FLAGS} -I${ADD_INCLUDE_DIR} -I${OPS_INCLUDE_DIR} -I${DATATYPE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR}  -c -o $@ $<

clean:
	rm -f test/test_add objs/cpu_add_avx2.o objs/cpu_add_avx512.o