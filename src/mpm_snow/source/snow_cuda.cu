/**
 *
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-10
 * @version    : 1.0
 * @file       snow_cuda_interface.cu
 * @brief      Provides a set of functions callable from outside the CUDA kernel
 *
 * This file defines a series of functions, which are interfaces exposed from the snow simulation kernel written in CUDA.
 * They allow external code to interact with the CUDA kernel to perform operations such as setting parameters, getting positions, updating particle and mesh state, etc.
 * These functions are a key part of data processing and state updating during snow simulation, they handle the transfer of data from particles to mesh,
 * Update particle positions and velocities, and handle collisions between particles and meshes.
 *
 * 主要函数:
 * - SetupParam: Copies solver parameters to CUDA symbolic memory.
 * - getPositionPtr: Gets the particle's position and transfers it to device memory.
 * - setPointPosition: Set the particle's position.
 * - update: Performs the main update loop of the simulation, including particle-to-mesh transfer, collision detection, and mesh-to-particle transfer back.
 *
 * @dependencies: Depends on CUDA kernel functions defined in "snow_kernel.cuh".
 *
 * @note       : When using these functions, you need to ensure that the CUDA environment is properly initialized and that all device pointers point to valid device memory.
*                There is no error checking inside the function, the caller is responsible for handling CUDA errors and synchronization issues.
 *
 * @remark     : Care should be taken when modifying these functions because they interact directly with the underlying CUDA kernel.
 *               Understanding the CUDA kernel is a prerequisite for effective modification.
 */
#include "snow_kernel.cuh"

#include <iostream>

namespace Physika {
void SetupParam(SolverParam& params)
{
    cudaMemcpyToSymbol(sp, &params, sizeof(SolverParam));
}

void getPositionPtr(float* position, Point* dev_points, unsigned int num_particle)
{
    dim3 particleDims = int(ceil(num_particle / blockSize + 0.5f));
    getPositionToDevice<<<particleDims, blockSize>>>(( float3* )position, dev_points, num_particle);
}

void getVelocityPtr(float* velocity, Point* dev_points, unsigned int num_particle)
{
    dim3 particleDims = int(ceil(num_particle / blockSize + 0.5f));
    getVelocityToDevice<<<particleDims, blockSize>>>(( float3* )velocity, dev_points, num_particle);
}

void getPosAndVelPtr(float* position, float* velocity, Point* dev_points, unsigned int num_particle)
{
	dim3 particleDims = int(ceil(num_particle / blockSize + 0.5f));
    getVelocityPositionToDevice<<<particleDims, blockSize>>>(( float3* )position, ( float3* )velocity, dev_points, num_particle);
}

void setPointPosition(float* position, Point* dev_points, unsigned int num_particle)
{
    dim3 particleDims = int(ceil(num_particle / blockSize + 0.5f));
    setPointPosition<<<particleDims, blockSize>>>(( float3* )position, dev_points, num_particle);
}

void update(
    Point* device_points, 
    Grid*  device_grids, 
    float* extern_force, 
    float* height, 
    float  unit_height, 
    int    height_x_num, 
    int    height_z_num, 
    int    particle_num, 
    int    grid_num)
{
    dim3 particleDims = int(ceil(particle_num / blockSize + 0.5f));

    dim3 grid_blockDim(8, 8, 8);
    dim3 grid_gridDim((64 + grid_blockDim.x - 1) / grid_blockDim.x, (64 + grid_blockDim.y - 1) / grid_blockDim.y, (64 + grid_blockDim.z - 1) / grid_blockDim.z);

    for (int i = 0; i < 2; i++)
    {
        GridClear<<<grid_gridDim, grid_blockDim>>>(device_grids, grid_num);
        world2local<<<particleDims, blockSize>>>(device_points, particle_num);
        if (!first_time)
            P2G<<<particleDims, blockSize>>>(device_points, device_grids, (float3*)extern_force, particle_num);

        if (first_time)
        {
            initialTransfer<<<particleDims, blockSize>>>(device_points, device_grids, particle_num);
            computeVolumes<<<particleDims, blockSize>>>(device_points, device_grids, particle_num);
            first_time = false;
        }

        UpdateVelocityGrid<<<grid_gridDim, grid_blockDim>>>(device_grids, grid_num);

        bodyCollisions<<<grid_gridDim, grid_blockDim>>>(device_grids, grid_num, height, unit_height, height_x_num, height_z_num);

        updateDeformationGradient<<<particleDims, blockSize>>>(device_points, device_grids, particle_num);

        G2P<<<particleDims, blockSize>>>(device_points, device_grids, particle_num);

        local2world<<<particleDims, blockSize>>>(device_points, particle_num);

        particleBodyCollisions << <particleDims, blockSize >> > (device_points, particle_num, height, unit_height, height_x_num, height_z_num);

        cudaDeviceSynchronize();
    }
}

}  // namespace Physika