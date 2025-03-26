#!/bin/bash

nsys profile -t cuda,nvtx,cublas,osrt --gpu-metrics-devices=0 --gpu-metrics-set=gh100 --force-overwrite true -o profiling/flash_lib_$1_seqs_$2_len ./test_flash_lib $1 $2
