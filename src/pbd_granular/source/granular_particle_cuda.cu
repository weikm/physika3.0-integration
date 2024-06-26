/**
 * @file       : granular_kernel_interface.cuh
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-07
 * @description: This file defines a set of CUDA functions which serve as an interface between the CUDA kernels
 *               defined in 'granular_kernel.cuh' and the rest of the program. These functions provide the
 *               functionality necessary for executing various parts of the granular simulation, such as setting
 *               parameters, computing particle hashes, sorting particles, finding cell ranges, and executing the
 *               granular advection and constraint solving processes.
 *
 *               These functions are defined in the 'Physika' namespace and are externally visible, making them
 *               available for use throughout the program. The main purpose of this file is to allow the rest of the
 *               program to interact with the CUDA kernels without needing to deal directly with the intricacies of
 *               CUDA programming.
 *
 * @version    : 1.0
 * @note       : All CUDA functions are prefixed with a void return type and followed by their respective function
 *               bodies.
 * @dependencies: This file includes dependencies on several CUDA and Thrust libraries, as well as the
 *                'granular_kernel.cuh' file, which contains the CUDA kernels that these functions interface with.
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include "granular_kernel.cuh"

namespace Physika {

void getLastCudaErrorGranular(const char* errorMessage)
{
    // check cuda last error.
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cout << "getLastCudaErrorGranular() CUDA error : "
                  << errorMessage << " : "
                  << "(" << static_cast<int>(err) << ") "
                  << cudaGetErrorString(err) << ".\n";
    }
}

void setParameters(GranularSimulateParams* hostParams)
{
    setSimulationParams(hostParams);
}

void computeHashGranular(
    unsigned int* gridParticleHash,
    float*        pos,
    int           numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    // launch the kernel.
    calcParticlesHashKernelGranular<<<numBlocks, numThreads>>>(
        gridParticleHash,
        ( float4* )pos,
        numParticles);
}

void sortParticlesGranular(
    unsigned int* deviceGridParticleHash,
    unsigned int  numParticles,
    float*        devicePos,
    float*        deviceVel,
    float*        devicePredictedPos,
    float*        phase)
{
    thrust::device_ptr<float4> ptrPos(( float4* )devicePos);
    thrust::device_ptr<float4> ptrVel(( float4* )deviceVel);
    thrust::device_ptr<float4> ptrPredictedPos(( float4* )devicePredictedPos);
    thrust::device_ptr<float>  particlePhase(phase);
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(deviceGridParticleHash),
        thrust::device_ptr<unsigned int>(deviceGridParticleHash + numParticles),
        thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptrPredictedPos, particlePhase)));
}

void findCellRangeGranular(
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCell)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    // set all cell to empty.
    cudaMemset(cellStart, 0xffffffff, numCell * sizeof(unsigned int));

    unsigned int memSize = sizeof(unsigned int) * (numThreads + 1);
    findCellRangeKernelGranular<<<numBlocks, numThreads, memSize>>>(
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles);
}

void granularAdvection(float4*      position,
                       float4*      velocity,
                       float4*      predictedPos,
                       float3*      collisionForce,
                       float*       phase,
                       float*       height,
                       float        unit_height,
                       int          height_x_num,
                       int          height_z_num,
                       float        deltaTime,
                       unsigned int numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    cudaMemset(collisionForce, 0, numParticles * sizeof(float3)); 

    granularAdvect<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        collisionForce,
        phase,
        height,
        unit_height,
        height_x_num,
        height_z_num,
        deltaTime,
        numParticles);
    cudaDeviceSynchronize();
    getLastCudaErrorGranular("granularAdvect");
}

void solveDistanceConstrainGranluar(
    float4*       postion,
    float4*       velocity,
    float3*       deltaPos,
    float4*       predictedPos,
    float*        phase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    distanceConstrainGranluar<<<numBlocks, numThreads>>>(predictedPos, deltaPos, phase, cellStart, cellEnd, gridParticleHash, numParticles, numCells);
    getLastCudaErrorGranular("distanceConstrainGranluar");

    addDeltaPositionGranular<<<numBlocks, numThreads>>>(
        predictedPos,
        deltaPos,
        phase,
        numParticles);
    getLastCudaErrorGranular("addDeltaPositionGranular");
}

void updateVelAndPosGranular(
    float4*      position,
    float4*      velocity,
    float*       phase,
    float        deltaTime,
    unsigned int numParticles,
    float4*      predictedPos)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
    updateVelocityAndPositionGranular<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        phase,
        1.0f / deltaTime,
        numParticles);
    // cudaDeviceSynchronize();
}

void solverColisionConstrainGranular(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numCollisionParticles % numThreads != 0) ? (numCollisionParticles / numThreads + 1) : (numCollisionParticles / numThreads);

    solverCollisionConstrainGranular<<<numBlocks, numThreads>>>(
        position,
        predictedPos,
        moveDirection,
        moveDistance,
        particlePhase,
        collision_particle_id,
        numCollisionParticles);
}

}  // namespace Physika