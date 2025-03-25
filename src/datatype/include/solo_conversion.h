#ifndef SOLO_CONVERION_H
#define SOLO_CONVERION_H

#include "dataflow_common.h"

float solo_bf16_to_fp32(uint16_t a);

uint16_t solo_fp32_to_bf16(float f);

float solo_fp16_to_fp32(uint16_t h);

uint16_t solo_fp32_to_fp16(float f);

uint16_t solo_fp16_to_bf16(uint16_t h);

uint16_t solo_bf16_to_fp16(uint16_t a);

#endif