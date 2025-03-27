#!/bin/bash

nsys profile -t cuda,nvtx,cublas,osrt --gpu-metrics-devices=all --gpu-metrics-set=gh100 --force-overwrite true -o profiling/flash_lib_$1_seqs_$2_len ./test_libflash3 $1 $2
