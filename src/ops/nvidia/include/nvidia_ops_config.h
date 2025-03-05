#ifndef NVIDIA_OPS_CONFIG_H
#define NVIDIA_OPS_CONFIG_H

#include "dataflow_common.h"
#include "ops.h"
#include "dataflow_handle.h"
#include "cuda_dataflow_handle.h"

int swiglu_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


int rms_norm_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function);
int rms_norm_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


#endif