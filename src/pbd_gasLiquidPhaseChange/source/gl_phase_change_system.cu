#include "gl_phase_change_system.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "device_launch_parameters.h"
#include <memory>

#include "Util/CudaFunction.cuh"
namespace Physika {
BoilingSystem::BoilingSystem(
        std::shared_ptr<ThermoParticles>& fluidModel,
        std::shared_ptr<ThermoParticles>& boundariesModel,
		std::shared_ptr<ThermoParticles>& heatenModel,
        std::shared_ptr<ThermalPBFSolver>& solver,
        const float3 spaceSize,
        const float cellLength,
        const float smoothRadius,
        const float dt,
        const float mass,
        const float rho0,
        const float rhoBoundary,
        const float dynamicVisc,
        const float3 gravity,
		const float surfaceTensionIntensity,
		const float airPressure,
        const int3 cellSize
)
    :
	m_fluids(std::move(fluidModel)),
	m_boundaries(std::move(boundariesModel)),
	m_solver(std::move(solver)),
	m_heaten(std::move(heatenModel)),
    cellStartFluid(cellSize.x * cellSize.y * cellSize.z + 1),
    cellStartBoundary(cellSize.x * cellSize.y * cellSize.z + 1),
	cellStartHeaten(cellSize.x* cellSize.y* cellSize.z + 1),
    m_spaceSize(spaceSize), m_smoothRadius(smoothRadius), m_cellLength(cellLength),
    m_dt(dt), m_rho0(rho0), m_rhoBoundary(rhoBoundary),
    m_gravity(gravity), m_dynamicVisc(dynamicVisc), m_surfaceTensionIntensity(surfaceTensionIntensity), m_airPressure(airPressure),
    m_cellSize(cellSize), bufferInt(max(particlesSize(), cellSize.x* cellSize.y* cellSize.z + 1)), m_latenHeat(2500.0f), m_startTemp(20.0f)
    {
		// Init boundary particles 
		neighborSearch(m_boundaries, cellStartBoundary);
		neighborSearch(m_heaten, cellStartHeaten);
		// compute MASS/PSI for boundary particles
		boundaryMassSampling();
		// fill MASS/VOLUME for fluid particles
		thrust::fill(thrust::device, m_fluids->getMassPtr(), m_fluids->getMassPtr() + m_fluids->getCapacity(), mass);
		// fluids neighbor search
		//neighborSearch(m_fluids, cellStartFluid);
		thrust::fill(thrust::device, m_fluids->getValidPtr(), m_fluids->getValidPtr() + m_fluids->size(), 1);
		thrust::fill(thrust::device, m_fluids->getHumidityPtr(), m_fluids->getHumidityPtr() + m_fluids->size(), 1.0f);
		// Single Step 
		//solveStep();
    }

/*
    Compute MASS/PSI for Boundary particles 
*/
static inline __device__ void contributeBoundaryKernel(float* sum_kernel, const int i, const int cellID, float3* pos, uint32_t* cellStart, const int3 cellSize, const float radius)
{
	if (cellID == (cellSize.x * cellSize.y * cellSize.z)) return;
	auto j = cellStart[cellID];
	const auto end = cellStart[cellID + 1];
	while (j < end)
	{
		*sum_kernel += cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		++j;
	}
	return;
}
__global__ void computeHeatenBoundaryMass_CUDA(float* mass, float3* pos, const int num, uint32_t* cellStart, const int3 cellSize, const float cellLength, const float rhoB, const float radius)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	const auto cellPos = make_int3(pos[i] / cellLength);
	for (auto m = 0; m < 27; ++m)
	{
		const auto cellID = particlePos2cellIdx(cellPos + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		contributeBoundaryKernel(&mass[i], i, cellID, pos, cellStart, cellSize, radius);
	}
	mass[i] = rhoB / fmaxf(EPSILON, mass[i]);
	return;
}

void BoilingSystem::boundaryMassSampling()
{
	computeHeatenBoundaryMass_CUDA <<<(m_boundaries->size() - 1) / block_size + 1, block_size >>> (
		m_boundaries->getMassPtr(), m_boundaries->getPosPtr(),m_boundaries->size(),
		cellStartBoundary.addr(), m_cellSize, m_cellLength,m_rhoBoundary, m_smoothRadius);
	computeHeatenBoundaryMass_CUDA << <(m_heaten->size() - 1) / block_size + 1, block_size >> > (
		m_heaten->getMassPtr(), m_heaten->getPosPtr(), m_heaten->size(),
		cellStartHeaten.addr(), m_cellSize, m_cellLength, m_rhoBoundary, m_smoothRadius);
}

/*
    Neighbor search. 
*/
__global__ void mapParticles2CellsBoil_CUDA(uint32_t* particles2cells, float3* pos, const float cellLength, const int3 cellSize, const int num)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i] / cellLength), cellSize);
	return;
}

__global__ void countingInCellBoil_CUDA(uint32_t* cellStart, uint32_t* particle2cell, const int num)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	atomicAdd(&cellStart[particle2cell[i]], 1);
	return;
}

void BoilingSystem::neighborSearch(const std::shared_ptr<ThermoParticles> &particles, DataArray<uint32_t> &cellStart)
{
	int num = particles->size();
    // Linear curve
	mapParticles2CellsBoil_CUDA <<<(num - 1) / block_size + 1, block_size >>> (
		particles->getP2Cell(), particles->getPosPtr(), m_cellLength, m_cellSize, num);

	thrust::device_ptr<float3> ptrPos(particles->getPosPtr());
	thrust::device_ptr<float3> ptrVel(particles->getVelPtr());
	thrust::device_ptr<float> ptrTem(particles->getTempPtr());
	thrust::device_ptr<float> ptrLatent(particles->getLatentPtr());
	thrust::device_ptr<float> ptrHumidity(particles->getHumidityPtr());
	thrust::device_ptr<int>   ptrType(particles->getTypePtr());

	checkCudaErrors(cudaMemcpy(bufferInt.addr(), particles->getP2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice)); 
	thrust::sort_by_key(
		thrust::device,
		bufferInt.addr(),
		bufferInt.addr() + num,
		thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptrTem, ptrLatent, ptrHumidity, ptrType)));
	
	// fill cell info
	thrust::fill(thrust::device, cellStart.addr(), cellStart.addr() + m_cellSize.x * m_cellSize.y * m_cellSize.z + 1, 0);
	countingInCellBoil_CUDA <<<(num - 1) / block_size + 1, block_size >>> (
		cellStart.addr(), particles->getP2Cell(), num);
	// compute prefix sum
	thrust::exclusive_scan(thrust::device, cellStart.addr(), cellStart.addr() + m_cellSize.x * m_cellSize.y * m_cellSize.z + 1, cellStart.addr());
	return;
}

/*
    Single step, return time consuming
*/

float BoilingSystem::solveStep()
{
    if(steps == 0) 
		neighborSearch(m_fluids, cellStartFluid);
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	neighborSearch(m_fluids, cellStartFluid);

	if(steps > 1000)
		m_solver->thermalStep(m_fluids, m_boundaries, m_heaten, cellStartFluid, cellStartBoundary, cellStartHeaten,
		m_spaceSize, m_cellSize, m_cellLength, m_smoothRadius,
		m_dt, m_rho0, m_rhoBoundary, m_dynamicVisc, m_gravity, m_surfaceTensionIntensity, m_airPressure);
	else
		m_solver->normalStep(m_fluids, m_boundaries, m_heaten, cellStartFluid, cellStartBoundary, cellStartHeaten,
			m_spaceSize, m_cellSize, m_cellLength, m_smoothRadius,
			m_dt, m_rho0, m_rhoBoundary, m_dynamicVisc, m_gravity, m_surfaceTensionIntensity, m_airPressure);

	checkCudaErrors(cudaDeviceSynchronize());
	float milliseconds;
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
    steps += 1;
    resetExternalForce();
	return milliseconds;
}
void BoilingSystem::resetExternalForce()
{
    thrust::fill(thrust::device, m_fluids->getExternalForcePtr(), m_fluids->getExternalForcePtr() + m_fluids->size(), make_float3(0.0f));
}
bool BoilingSystem::setExternalForce(const std::vector<float3>& ex_acc_source)
{
    if (ex_acc_source.size() < m_fluids->size() * 3)
    {
        return false;
	}
    thrust::device_ptr<float3> ex_force_tar(m_fluids->getExternalForcePtr());
    thrust::copy(ex_acc_source.begin(), ex_acc_source.begin() + m_fluids->size(), ex_force_tar);
    return true;
}

bool BoilingSystem::setVelocity(const std::vector<float3>& ex_vel_source)
{
    if (ex_vel_source.size() < m_fluids->size())
    {
        std::cout << "Set velocity error: size mismatching.\n";
        return false;
    }
    thrust::device_ptr<float3> ex_vel_tar(m_fluids->getVelPtr());
    thrust::copy(ex_vel_source.begin(), ex_vel_source.begin() + m_fluids->size(), ex_vel_tar);
    float3 temp = ex_vel_tar[0];
    std::cout << temp.y << "\n";
    return true;
}
}
