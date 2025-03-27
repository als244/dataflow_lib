#!/bin/bash

ncu -o profiling/flash_kernel/$1_seq_$2_seqlen --set detailed ./test_libflash3 $1 $2
