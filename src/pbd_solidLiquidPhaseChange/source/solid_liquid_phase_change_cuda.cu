/**
 * @file       : solid_liquid_phase_change_kernel.cuh
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @description: This file defines a set of CUDA functions which serve as an interface between the CUDA kernels
 *               defined in 'solid_liquid_phase_change_kernel.cuh' and the rest of the program. These functions provide
 *               the functionality necessary for executing various parts of the solid-liquid phase change simulation,
 *               such as setting parameters, computing particle properties, sorting particles, finding neighbor cells,
 *               and executing the temperature transfer and constraint management processes.
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
 *                'solid_liquid_phase_change_kernel.cuh' file, which contains the CUDA kernels that these functions
 *                interface with.
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include "solid_liquid_phase_change_kernel.cuh"

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

void setParameters(SolidLiquidPhaseChangeParams* hostParams)
{
    setSimulationParams(hostParams);
}

void computeHash(
	unsigned int *gridParticleHash,
	float *pos,
	int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	// launch the kernel.
	calcParticlesHashKernel << <numBlocks, numThreads >> > (
		gridParticleHash,
		(float4*)pos,
		numParticles);
}


void sortParticlesAll(
	unsigned int* deviceGridParticleHash,
	unsigned int numParticles,
	float* devicePos,
	float* deviceTem,
	float* deviceVel,
	int* deviceType,
	int* deviceInitId2,
	float* devicePredictedPos
)
{
	thrust::device_ptr<float4> ptrPos((float4*)devicePos);	
	thrust::device_ptr<float> ptrTem((float*)deviceTem);//temperature
	thrust::device_ptr<float4> ptrVel((float4*)deviceVel);
	thrust::device_ptr<float4> ptrPredictedPos((float4*)devicePredictedPos);
	thrust::device_ptr<int> ptrType((int*)deviceType);
	thrust::device_ptr<int> ptrInitId2((int*)deviceInitId2);
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(deviceGridParticleHash),
		thrust::device_ptr<unsigned int>(deviceGridParticleHash + numParticles),
		thrust::make_zip_iterator(thrust::make_tuple(ptrPos,  ptrVel, ptrPredictedPos, ptrTem, ptrType, ptrInitId2)));
}


void sortParticlesForSolid(
	unsigned int* m_deviceGridParticleHash_solid,
	unsigned int numParticles,
	float* deviceInitPos,
	int* deviceInitId1
)
{
	thrust::device_ptr<float4> ptrInitPos((float4*)deviceInitPos);//rest pos
	thrust::device_ptr<int> ptrInitId1((int*)deviceInitId1);
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(m_deviceGridParticleHash_solid),
		thrust::device_ptr<unsigned int>(m_deviceGridParticleHash_solid + numParticles),
		thrust::make_zip_iterator(thrust::make_tuple(ptrInitPos,ptrInitId1)));
}

void findCellRange(
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCell)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	// set all cell to empty.
	cudaMemset(cellStart, 0xffffffff, numCell * sizeof(unsigned int));

	unsigned int memSize = sizeof(unsigned int) * (numThreads + 1);
	findCellRangeKernel << <numBlocks, numThreads, memSize >> > (
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
}

void setInitIdToNow(
	int* deviceInitId1,
	int* deviceInitId2,
	int* deviceInitId2Now,
	int* deviceInitId2Rest,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	setInitId2Now << <numBlocks, numThreads >> > (
		deviceInitId1,
		deviceInitId2,
		deviceInitId2Now,
		deviceInitId2Rest,
		numParticles);
}

void fluidAdvection(
	int* type,
	float4 *position,
	float* temperature,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles,
	float3* external_force)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	advect<< < numBlocks, numThreads >> > (
		type,
		position,
		temperature,
		velocity,
		predictedPos,
		deltaTime,
		numParticles,
		external_force);
	cudaDeviceSynchronize();
}

void densityConstraint(
	int* type,
	float4 *position,
	float4 *velocity,
	float* temperature,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	// calculate density and lagrange multiplier.
	calcLagrangeMultiplier << <numBlocks, numThreads >> > (
		type,
		predictedPos,
		velocity,
		temperature,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles,
		numCells);
	getLastCudaError("calcLagrangeMultiplier");
	//cudaDeviceSynchronize();
	
	// calculate delta position.
	calcDeltaPosition << <numBlocks, numThreads >> > (
		type,
		predictedPos,
		velocity,
		temperature,
		deltaPos,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	getLastCudaError("calcDeltaPosition");
	//cudaDeviceSynchronize();

	// add delta position.
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	addDeltaPosition << <numBlocks, numThreads >> > (
		type,
		predictedPos,
		deltaPos,
		numParticles);
	getLastCudaError("addDeltaPosition");
	//cudaDeviceSynchronize();
}

void updateTemperature(
	int* type,
	float4* position,
	float* temperature,
	float* deltaTem,
	float* latentTem,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	   
	// calculate delta temperature.
	calcDeltaTem << <numBlocks, numThreads >> > (
		type,
		position,
		temperature,
		deltaTem,
		latentTem,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	getLastCudaError("calcDeltaTem");
	//cudaDeviceSynchronize();

	// add delta temperature.
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	addDeltaTem << <numBlocks, numThreads >> > (
		type,
		temperature,
		deltaTem,
		numParticles);
	getLastCudaError("addDeltaTem");
	//cudaDeviceSynchronize();
}

void solidFluidPhaseChange(
	int* type,
	int* initId,
	int* initId2Rest,
	float4* initPos,
	float4* position,
	float* temperature,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles)	
{
	//is_solidify[0] = false;

	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	   
	// phase change
	managePhaseChange << <numBlocks, numThreads >> > (
		type,
		initId,
		initId2Rest,
		initPos,
		position,
		temperature,		
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	getLastCudaError("managePhaseChange");
	//cudaDeviceSynchronize();

}

void solveDistanceConstrain(
	int* type,
	int* initId,
	int* initId2Now,
	float4* predictedPos,
	float4* initPos,
	float3* deltaPos,	
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles
)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	distanceConstran << <numBlocks, numThreads >> > (
		type,
		initId,
		initId2Now,
		predictedPos, initPos,deltaPos, cellStart, cellEnd, gridParticleHash, numParticles);
	getLastCudaError("distanceConstrain");


	addDeltaPosition << <numBlocks, numThreads >> > (
		type,
		predictedPos,
		deltaPos,
		numParticles);
	getLastCudaError("addDeltaPosition");

}

void updateVelAndPos(
int* type,
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	updateVelocityAndPosition << <numBlocks, numThreads >> > (
		type,
		position,
		velocity,
		predictedPos,
		1.0f / deltaTime,
		numParticles);
	//cudaDeviceSynchronize();
}

void applyXSPHViscosity(
int* type,
	float4 *velocity,
	float4 *position,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	applyXSPH << <numBlocks, numThreads >> > (
	type,
	position, velocity, cellStart, cellEnd, gridParticleHash, numParticles);
	getLastCudaError("applyXSPHViscosity");
}

}  // namespace Physika