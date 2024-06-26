/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-10-18
 * @description: implementation of cuda kernel functions
 * @version    : 1.0
 */

#include "cuda_kernels.cuh"

__global__ void stepKernel(double dt, double gravity, size_t data_num, double* vel, double* pos)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= data_num)
        return;

    vel[tid] += gravity * dt;
    pos[tid] += vel[tid] * dt;
}