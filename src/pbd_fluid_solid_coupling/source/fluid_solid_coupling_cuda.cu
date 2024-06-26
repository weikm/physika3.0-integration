/**
 * @file       : fluid_solid_coupling_cuda.cu
 * @author     : Yuhang Xu (mr.xuyh@qq.com)
 * @date       : 2023-08-17
 * @description: This file defines a set of CUDA functions which serve as an interface between the CUDA kernels
 *               defined in 'fluid_solid_kernel.cuh' and the rest of the program. These functions provide the
 *               functionality necessary for executing various parts of the fluid_solid simulation, such as setting
 *               parameters, computing particle hashes, sorting particles, finding cell ranges, and executing the
 *               fluid_solid advection and constraint solving processes.
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
 *                'fluid_solid_kernel.cuh' file, which contains the CUDA kernels that these functions interface with.
 */

#include "fluid_solid_coupling_kernel.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>



namespace Physika {

void getLastCudaErrorCoupling(const char* errorMessage)
{
    // check cuda last error.
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cout << "getLastCudaErrorCoupling() CUDA error : "
                  << errorMessage << " : "
                  << "(" << static_cast<int>(err) << ") "
                  << cudaGetErrorString(err) << ".\n";
    }
}

void setParameters(FluidSolidCouplingParams* hostParams)
{
    setSimulationParams(hostParams);
}

void computeHashCoupling(
    unsigned int* gridParticleHash,
    float*        pos,
    int           numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    // launch the kernel.
    calcCouplingParticlesHashKernel<<<numBlocks, numThreads>>>(
        gridParticleHash,
        ( float4* )pos,
        numParticles);
}

void sortCouplingParticles(
    unsigned int* deviceGridParticleHash,
    unsigned int  numParticles,
    float*        devicePos,
    float*        deviceVel,
    float*        device_radius_pos,
    float*        devicePredictedPos,
    int*        phase)
{
    thrust::device_ptr<float4> ptrPos(( float4* )devicePos);
    thrust::device_ptr<float4> ptrVel(( float4* )deviceVel);
    thrust::device_ptr<float3> ptr_radius_pos(( float3* )device_radius_pos);
    thrust::device_ptr<float4> ptrPredictedPos(( float4* )devicePredictedPos);
    thrust::device_ptr<int>    particlePhase(phase);
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(deviceGridParticleHash),
        thrust::device_ptr<unsigned int>(deviceGridParticleHash + numParticles),
        thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptr_radius_pos, ptrPredictedPos, particlePhase)));
}

void findCellRangeCoupling(
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
    findCellRangeCouplingKernel<<<numBlocks, numThreads, memSize>>>(
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles);
}

void particleAdvection(float4*      position,
                       float4*      velocity,
                       float4*      predictedPos,
                       float3*      collisionForce,
                       int*         phase,
                       float        deltaTime,
                       unsigned int numParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

    cudaMemset(collisionForce, 0, numParticles * sizeof(float3));  // 将内存全部初始化为0

    particleAdvect<<<numBlocks, numThreads>>>(
        position,
        velocity,
        predictedPos,
        collisionForce,
        phase,
        deltaTime,
        numParticles);
    cudaDeviceSynchronize();
    getLastCudaErrorCoupling("particleAdvect");
}

void solveDensityConstrainCoupling(
    int*          phase,
    float4*       postion,
    float4*       velocity,
    float3*       deltaPos,
    float4*       predictedPos,
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
    calcLagrangeMultiplierCoupling<<<numBlocks, numThreads>>>(
        phase,
        predictedPos,
        velocity,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles,
        numCells);
    getLastCudaErrorCoupling("calcLagrangeMultiplier");
    //cudaDeviceSynchronize();

    // calculate delta position.6
    calcDeltaPositionCoupling<<<numBlocks, numThreads>>>(
        phase,
        predictedPos,
        velocity,
        deltaPos,
        cellStart,
        cellEnd,
        gridParticleHash,
        numParticles);
    getLastCudaErrorCoupling("calcDeltaPosition");
    //cudaDeviceSynchronize();

    // add delta position.
    numThreads = 256;
    numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
    addDeltaPositionCoupling<<<numBlocks, numThreads>>>(
        postion,
        predictedPos,
        deltaPos,
        phase,
        numParticles);
    getLastCudaErrorCoupling("addDeltaPosition");
    //cudaDeviceSynchronize();
}

void solveShapeMatchingConstrain(
    int*    phase,
    float4* position,
    float4* predicted_pos,
    float3* delta_pos,
    //float*        invmasses,
    float3*      radius_pos,
    float3       rest_cm,
    const float  stiffness,
    const bool   allow_stretch,
    float4*      velocity,
    float        deltaTime,
    unsigned int num_particles)
{
    mat3*   R;
    float3* cm;
    cudaMalloc(( void** )&cm, sizeof(float3));
    cudaMalloc(( void** )&R, sizeof(mat3));

    calMassCenterMatrixR<<<1, 1>>>(
        phase,
        predicted_pos,
        cm,
        radius_pos,
        allow_stretch,
        num_particles,
        R);

    //mat3   _R;
    //float3 _cm;
    //cudaMemcpy(&_R, R, sizeof(mat3), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&_cm, cm, sizeof(float3), cudaMemcpyDeviceToHost);

    //int a = 25;

    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (num_particles % numThreads != 0) ? (num_particles / numThreads + 1) : (num_particles / numThreads);

    float invDeltaTime = 1.f / deltaTime;
    updateSolidVelocityAndPosition<<<numBlocks, numThreads>>>(
        phase,
        position,
        predicted_pos,
        delta_pos,
        cm,
        radius_pos,
        stiffness,
        velocity,
        invDeltaTime,
        R,
        num_particles);

    cudaFree(cm);
    cudaFree(R);
}

void solverColisionConstrainCoupling(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    int*          particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles)
{
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks  = (numCollisionParticles % numThreads != 0) ? (numCollisionParticles / numThreads + 1) : (numCollisionParticles / numThreads);

    solverCollisionConstrainCoupling<<<numBlocks, numThreads>>>(
        position,
        predictedPos,
        moveDirection,
        moveDistance,
        particlePhase,
        collision_particle_id,
        numCollisionParticles);
}

}  // namespace Physika