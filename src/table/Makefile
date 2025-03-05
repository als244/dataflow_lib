CC = gcc
CFLAGS = -g -fPIC

PROJ_INCLUDE_DIR = ../../include

TABLE_SRC_DIR = src
TABLE_INCLUDE_DIR = include

TABLE_OBJ_DIR = objs

TABLE_TEST_DIR = test

EXECS = 

all: ${EXECS} ${TABLE_OBJ_DIR}/table.o

${TABLE_OBJ_DIR}/table.o: ${TABLE_SRC_DIR}/table.c
	${CC} ${CFLAGS} -I${TABLE_INCLUDE_DIR} -I${PROJ_INCLUDE_DIR} -c -o $@ $<


clean:
	rm -f ${TABLE_OBJ_DIR}/*.o ${EXECS}
