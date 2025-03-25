CC = gcc
CFLAGS = -O3 -fPIC

AVX2_INTRIN_FLAGS = -mavx2
AVX512_INTRIN_FLAGS = -mavx512f -mavx512fp16 -mavx512bf16

PROJ_INCLUDE_DIR = ../../include
DATATYPES_INCLUDE_DIR = include

MAIN_OBJS = objs/dataflow_datatype.o objs/solo_conversion.o
INTERNAL_OBJS = objs/convert_datatype.o objs/convert_datatype_avx512.o
ALL_OBJS = ${MAIN_OBJS} ${INTERNAL_OBJS}

EXECS = test/test_convert
DATATYPES_LIB_DEPENDS = -pthread


all: ${EXECS} ${ALL_OBJS}


test/test_convert: test/test_convert.c objs/dataflow_datatype.o objs/solo_conversion.o objs/convert_datatype.o objs/convert_datatype_avx512.o
	${CC} ${CFLAGS} -I${DATATYPES_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} $^ -o $@

objs/dataflow_datatype.o: src/dataflow_datatype.c
	${CC} ${CFLAGS} -I${DATATYPES_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -c -o $@ $<

objs/solo_conversion.o: src/solo_conversion.c
	${CC} ${CFLAGS} -I${DATATYPES_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -c -o $@ $<

objs/convert_datatype.o: src/convert_datatype.c
	${CC} ${CFLAGS} -I${DATATYPES_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -c -o $@ $<

objs/convert_datatype_avx512.o: src/convert_datatype_avx512.c
	${CC} ${CFLAGS} ${AVX512_INTRIN_FLAGS} -I${DATATYPES_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -c -o $@ $<

clean:
	rm -f ${EXECS} ${ALL_OBJS}
