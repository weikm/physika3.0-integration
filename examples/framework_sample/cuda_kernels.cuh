/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-10-18
 * @description: declaration of cuda kernel functions
 * @version    : 1.0
 */

#pragma once

#include <cuda_runtime.h>

__global__ void stepKernel(double dt, double gravity, size_t data_num, double* vel, double* pos);
