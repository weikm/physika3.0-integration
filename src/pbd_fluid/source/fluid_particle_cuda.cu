/**
 * @file       : fluid_kernel_interface.cuh
 * @author     : Yuege Xiong (candybear0714@163.com)
 * @date       : 2023-11-22
 * @description: This file defines a set of CUDA functions which serve as an interface between the CUDA kernels
 *               defined in 'fluid_kernel.cuh' and the rest of the program. These functions provide the
 *               functionality necessary for executing various parts of the fluid simulation, such as setting
 *               parameters, computing particle hashes, sorting particles, finding cell ranges, and executing the
 *               fluid advection and constraint solving processes.
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
 *                'fluid_kernel.cuh' file, which contains the CUDA kernels that these functions interface with.
 */

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include "fluid_kernel.cuh"

namespace Physika {

void getLastCudaError(const char* errorMessage)
{
    // check cuda last error.
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cout << "getLastCudaError() CUDA error : "
                  << errorMessage << " : "
                  << "(" << static_cast<int>(err) << ") "
                  << cudaGetErrorString(err) << ".\n";
    }
}

void setParameters(PBDFluidSimulateParams* hostParams)
{
    setSimulationParams(hostParams);
}

void computeHash(
    unsigned int* gridParticleHash,
    float*        pos,
    int           numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    // launch the kernel.
    calcParticlesHashKernel<<<numBlocks, numThreads>>>(
        gridParticleHash,
        ( float4* )pos,
        numParticles);
}

void sortParticles(
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

void findCellRange(
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
    findCellRangeKernel<<<numBlocks, numThreads, memSize>>>(
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles);
}

void fluidAdvection(float4*      position,
                       float4*      velocity,
                       float4*      predictedPos,
                       float*       phase,
                       float        deltaTime,
                       unsigned int numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    fluidAdvect<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        phase,
        deltaTime,
        numParticles);
    cudaDeviceSynchronize();
    getLastCudaError("fluidAdvect");
}

void solveContactConstrain(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       phase,
    float        deltaTime,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    contactSDF<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        phase,
        deltaTime,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles,
        numCells);
}

void solveDistanceConstrain(
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

    distanceConstrain<<<numBlocks, numThreads>>>(predictedPos, deltaPos, phase, cellStart, cellEnd, gridParticleHash, numParticles, numCells);
    getLastCudaError("distanceConstrain");

    addDeltaPosition<<<numBlocks, numThreads>>>(
        predictedPos,
        deltaPos,
        phase,
        numParticles);
    getLastCudaError("addDeltaPosition");
}

void solveDensityConstrain(
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

    // calculate density and lagrange multiplier.
    //std::cout << "run here" << std::endl;
    calcLagrangeMultiplier<<<numBlocks, numThreads>>>(
        predictedPos,
        velocity,
        phase,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles,
        numCells);
    getLastCudaError("calcLagrangeMultiplier");

    // calculate delta position.
    calcDeltaPosition<<<numBlocks, numThreads>>>(
        predictedPos,
        velocity,
        deltaPos,
        phase,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles);
    getLastCudaError("calcDeltaPosition");

    // add delta position.
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
    addDeltaPosition<<<numBlocks, numThreads>>>(
        predictedPos,
        deltaPos,
        phase,
        numParticles);
    getLastCudaError("addDeltaPosition");
}

void updateVelAndPos(
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
    updateVelocityAndPosition<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        phase,
        1.0f / deltaTime,
        numParticles);
    // cudaDeviceSynchronize();
}

void solverColisionConstrain(
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

    solverCollisionConstrain<<<numBlocks, numThreads>>>(
        position,
        predictedPos,
        moveDirection,
        moveDistance,
        particlePhase,
        collision_particle_id,
        numCollisionParticles);
}

void add_surface_tension(
    float4*       velocity,
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    float         deltaTime,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    add_surfacetension<<<numBlocks, numThreads>>>(
        velocity,
        predictedPos,
        deltaPos,
        particlePhase,
        deltaTime,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles,
        numCells);
    getLastCudaError("add_surfacetension");

    add_adhesionforce<<<numBlocks, numThreads>>>(
        velocity,
        predictedPos,
        deltaPos,
        particlePhase,
        deltaTime,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles,
        numCells);
    getLastCudaError("add_adhesionforce");
}

}  // namespace Physika