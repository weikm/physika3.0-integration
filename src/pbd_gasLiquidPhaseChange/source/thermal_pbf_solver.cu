#include "thermal_pbf_solver.hpp"

#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <memory>

#include "thermal_pbf_kernel.cuh"

namespace Physika {
void ThermalPBFSolver::normalStep(std::shared_ptr<ThermoParticles>& fluids,
	const std::shared_ptr<ThermoParticles>& boundaries,
	const std::shared_ptr<ThermoParticles>& heaten_boundaries,
	const DataArray<uint32_t>& cellStartFluid,
	const DataArray<uint32_t>& cellStartBoundary,
	const DataArray<uint32_t>& cellStartHeaten,
	float3 spaceSize, int3 cellSize, float cellLength, float smoothRadius,
	float dt, float rho0, float rhoB, float visc, float3 G,
	float surfaceTensionIntensity, float airPressure) {
	auto sph_fluids = static_cast<std::shared_ptr<SPHParticles>>(fluids);
	auto sph_boundaries = static_cast<std::shared_ptr<SPHParticles>>(boundaries);
	if (!posLastInitialized) {
        predict(sph_fluids, dt, spaceSize);
		initializePosLast(fluids->getPos());
	}
	// step 1: update local neighborhood
	updateNeighborhood(fluids);
	// step 2: iteratively correct position
	project(fluids, boundaries,
		cellStartFluid, cellStartBoundary,
		rho0, cellSize, spaceSize, cellLength, smoothRadius, m_maxIter);
	// step 3: calculate velocity
	thrust::transform(thrust::device,
		fluids->getPosPtr(), fluids->getPosPtr() + fluids->size(),
		fluidPosLast.addr(),
		fluids->getVelPtr(),
		[dt]__host__ __device__(const float3 & lhs, const float3 & rhs) { return (lhs - rhs) / dt; }
	);
	// step 4: apply non-pressure forces (gravity, XSPH viscosity and surface tension)
	diffuse(fluids, cellStartFluid, cellSize,
		cellLength, rho0, smoothRadius, m_xSPH_c);
	if (surfaceTensionIntensity > EPSILON || airPressure > EPSILON)
		handleSurface(sph_fluids, sph_boundaries,
			cellStartFluid, cellStartBoundary,
			rho0, rhoB, cellSize, cellLength, smoothRadius,
			dt, surfaceTensionIntensity, airPressure);

	force(sph_fluids, dt, G);
	// step 5: predict position for next timestep
	predict(sph_fluids, dt, spaceSize);
}

// KageStep: preserve original step util kageStep solver compeleted.

void ThermalPBFSolver::thermalStep(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
	const std::shared_ptr<ThermoParticles>& heaten_boundaries,
	const DataArray<uint32_t>& cellStartFluid, 
	const DataArray<uint32_t>& cellStartBoundary, 
	const DataArray<uint32_t>& cellStartHeaten,
	float3 spaceSize, int3 cellSize, float cellLength, float smoothRadius, 
	float dt, float rho0, float rhoB, float visc, float3 G,
	float surfaceTensionIntensity, float airPressure) {
	// Position-based Fluids need the position in last timestep to calculate velocity.
	// If it is not provided, use the current position as the history position in the next timestep.

	auto sph_fluids = static_cast<std::shared_ptr<SPHParticles>>(fluids);
	auto sph_boundaries = static_cast<std::shared_ptr<SPHParticles>>(boundaries);

	unsigned int num = fluids->size();
	if (!posLastInitialized) {
		initializePosLast(fluids->getPos());
	}
	// step 1: update local neighborhood
	updateNeighborhood(fluids);
	// step 2: iteratively correct position
	project(fluids, boundaries,
		cellStartFluid, cellStartBoundary,
		rho0, cellSize, spaceSize, cellLength, smoothRadius, m_maxIter);
	// step 3: calculate velocity

	thrust::transform(thrust::device,
		fluids->getPosPtr(), fluids->getPosPtr() + fluids->size(),
		fluidPosLast.addr(),
		fluids->getVelPtr(),
		[dt]__host__ __device__(const float3 & lhs, const float3 & rhs) { return (lhs - rhs) / dt; }
	);
	//gaseousVelocities << <(num - 1) / block_size + 1, block_size >> > (fluids->getPosPtr(), fluids->getTypePtr(), num);

	// step 4: apply non-pressure forces (gravity, XSPH viscosity and surface tension)
	diffuse(fluids, cellStartFluid, cellSize,
		cellLength, rho0, smoothRadius, m_xSPH_c);
	/*
	* Surface tension and other force
	*/
	if (surfaceTensionIntensity > EPSILON || airPressure > EPSILON)
		handleSurface(sph_fluids, sph_boundaries,
			cellStartFluid, cellStartBoundary,
			rho0, rhoB, cellSize, cellLength, smoothRadius,
			dt, surfaceTensionIntensity, airPressure);

	force(sph_fluids, dt, G);
    convectionForce(fluids, heaten_boundaries, dt, G, cellStartFluid, cellStartHeaten, cellSize, cellLength, smoothRadius);
	bubbleExternalForce(dt, fluids, heaten_boundaries, cellStartFluid, cellStartHeaten, rho0, G, cellSize, spaceSize, cellLength, smoothRadius);
	// step *: heat transfer
	// heat_transfer(dt, fluids, boundaries, cellStartFluid, cellStartBoundary, rhoB, cellSize, cellLength, smoothRadius);
	heatTransfer(dt, fluids, heaten_boundaries, cellStartFluid, cellStartHeaten, rhoB, cellSize, cellLength, smoothRadius);
	massTransferVaporization(fluids, heaten_boundaries, cellStartFluid, cellStartHeaten, rho0, 3610.0f, cellSize, spaceSize, cellLength, smoothRadius);
	// mass_transfer2(fluids, heaten_boundaries, cellStartFluid, cellStartHeaten, rho0, 3610.0f, cellSize, spaceSize, cellLength, smoothRadius);
	if (sph_fluids->check_size()) {
		printf(" new particle size: %d", sph_fluids->size());
	}

	massTransferCondensation(dt, fluids, boundaries, cellStartFluid, cellStartBoundary, rho0, 3610.0f, cellSize, spaceSize, cellLength, smoothRadius);
	
	// step 5: predict position for next timestep
	predict(sph_fluids, dt, spaceSize);

}

/*
* PBF solver members
*/
void ThermalPBFSolver::predict(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize)
{
	checkCudaErrors(cudaMemcpy(fluidPosLast.addr(), fluids->getPosPtr(), sizeof(float3) * fluids->size(), cudaMemcpyDeviceToDevice));
	advect(fluids, dt, spaceSize);
}

void ThermalPBFSolver::updateNeighborhood(const std::shared_ptr<SPHParticles>& particles)
{
	const int num = particles->size();
	checkCudaErrors(cudaMemcpy(bufferInt.addr(), particles->getP2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, bufferInt.addr(), bufferInt.addr() + num, fluidPosLast.addr());
	return;
}

void ThermalPBFSolver::diffuse(std::shared_ptr<ThermoParticles>& fluids, const DataArray<uint32_t>& cellStartFluid,
	int3 cellSize, float cellLength, float rho0,
	float smoothRaidus, float visc)
{
	int num = fluids->size();
	XSPHViscosityTP_CUDA << <(num - 1) / block_size + 1, block_size >> > (fluids->getVelPtr(), fluids->getPosPtr(), fluids->getTypePtr(),
		fluids->getMassPtr(), num, cellStartFluid.addr(), cellSize,
		cellLength, smoothRaidus, visc, rho0);
}

int ThermalPBFSolver::project(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
	const  DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
	float rho0, int3 cellSize, float3 spaceSize, float cellLength,
	float smoothRaidus, int maxIter)
{
	int num = fluids->size();
	auto iter = 0;
	while (iter < maxIter) {
		// step 1: compute lambda
		// use bufferFloat as lambda
		computeDensityLambdaTP_CUDA << <(num - 1) / block_size + 1, block_size >> > (fluids->getDensityPtr(), bufferFloat.addr(), fluids->getTypePtr(),
			fluids->getPosPtr(), fluids->getMassPtr(), fluids->size(), cellStartFluid.addr(), cellSize,
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
			cellLength, smoothRaidus, rho0, m_relaxation);
		// step 2: compute Delta pos for density correction
		// use bufferFloat3 as Delta pos
		computeDeltaPosTP_CUDA << <(num - 1) / block_size + 1, block_size >> > (bufferFloat3.addr(),
			fluids->getPosPtr(), boundaries->getPosPtr(), bufferFloat.addr(), fluids->getTypePtr(),
			fluids->getMassPtr(), boundaries->getMassPtr(), num,
			cellStartFluid.addr(), cellStartBoundary.addr(), cellSize,
			cellLength, smoothRaidus, rho0);
		// step 3: update pos
		thrust::transform(thrust::device,
			fluids->getPosPtr(), fluids->getPosPtr() + num,
			bufferFloat3.addr(),
			fluids->getPosPtr(),
			thrust::plus<float3>());
		enforceBoundary_CUDA << <(num - 1) / block_size + 1, block_size >> >
			(fluids->getPosPtr(), num, spaceSize);

		++iter;
	}
	return iter;
}

void ThermalPBFSolver::force(std::shared_ptr<SPHParticles>& fluids, float dt, float3 G)
{
	// add external force. 
    externalForces_Cuda<<<((fluids->size()) - 1) / block_size + 1, block_size>>>(fluids->getVelPtr(), fluids->getExternalForcePtr(), G, dt, fluids->size());
}
void ThermalPBFSolver::advect(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize)
{
    fluids->advect(dt);
    enforceBoundaryAdvect_CUDA<<<((fluids->size()) - 1) / block_size + 1, block_size>>>(fluids->getPosPtr(), fluids->getVelPtr(), fluids->size(), spaceSize);
}

////////////////////////////////////////////////////////////// Surface /////////////////////////////////////////////////////////////
void ThermalPBFSolver::handleSurface(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries, const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rho0, float rhoB, int3 cellSize, float cellLength, float radius, float dt, float surfaceTensionIntensity, float airPressure)
{
    surfaceDetection(surfaceBufferFloat3, fluids, boundaries, cellStartFluid, cellStartBoundary, rho0, rhoB, cellSize, cellLength, radius);
    applySurfaceEffects(fluids, surfaceBufferFloat3, cellStartFluid, rho0, cellSize, cellLength, radius, dt, surfaceTensionIntensity, airPressure);
}

__device__ auto contributeColorGrad_fluid(float3& numerator, float& denominator, const int i, float3* pos, float* mass, int j, const uint32_t cellEnd, const float radius, const float rho0) -> void
{
    while (j < cellEnd)
    {
        numerator += mass[j] / rho0 * cubic_spline_kernel_gradient(pos[i] - pos[j], radius);
        denominator += mass[j] / rho0 * cubic_spline_kernel(length(pos[i] - pos[j]), radius);
        ++j;
    }
    return;
}

__device__ void contributeColorGrad_boundary(float3& numerator, float& denominator, float3* pos_i, float3* pos, float* mass, int j, const uint32_t cellEnd, const float radius, const float rhoB)
{
    while (j < cellEnd)
    {
        numerator += mass[j] / rhoB * cubic_spline_kernel_gradient(*pos_i - pos[j], radius);
        denominator += mass[j] / rhoB * cubic_spline_kernel(length(*pos_i - pos[j]), radius);
        ++j;
    }
    return;
}

__global__ void computeColorGrad_CUDA(float3* colorGrad, float3* posFluid, float* massFluid, const int num, uint32_t* cellStartFluid, const int3 cellSize, float3* posBoundary, float* massBoudnary, uint32_t* cellStartBoundary, const float cellLength, const float radius, const float rho0, const float rhoB)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num)
        return;
    auto c_g         = make_float3(0.0f);
    auto denominator = 0.0f;
#pragma unroll
    for (auto m = 0; m < 27; __syncthreads(), ++m)
    {
        const auto cellID = particlePos2cellIdx(
            make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
        if (cellID == (cellSize.x * cellSize.y * cellSize.z))
            continue;
        contributeColorGrad_fluid(c_g, denominator, i, posFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius, rho0);
        contributeColorGrad_boundary(c_g, denominator, &posFluid[i], posBoundary, massBoudnary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius, rhoB);
    }

    colorGrad[i] = c_g / fmaxf(EPSILON, denominator);
    return;
}

void ThermalPBFSolver::surfaceDetection(DataArray<float3>& colorGrad, const std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries, const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rho0, float rhoB, int3 cellSize, float cellLength, float radius)
{
    computeColorGrad_CUDA<<<(fluids->size() - 1) / block_size + 1, block_size>>>(colorGrad.addr(),
                                                                                 fluids->getPosPtr(),
                                                                                 fluids->getMassPtr(),
                                                                                 fluids->size(),
                                                                                 cellStartFluid.addr(),
                                                                                 cellSize,
                                                                                 boundaries->getPosPtr(),
                                                                                 boundaries->getMassPtr(),
                                                                                 cellStartBoundary.addr(),
                                                                                 cellLength,
                                                                                 radius,
                                                                                 rho0,
                                                                                 rhoB);
    return;
}

__device__ void contributeSurfaceTensionAndAirPressure(float3& a, const int i, float3* pos, float* mass, float3* color_grad, int j, const uint32_t cellEnd, const float radius, const float rho0, const float color_energy_coefficient, const float airPressure)
{
    while (j < cellEnd)
    {
        // surface tension
        a += 0.25f * mass[j] / (rho0 * rho0) * color_energy_coefficient
             * (dot(color_grad[i], color_grad[i]) + dot(color_grad[j], color_grad[j]))
             * surface_tension_kernel_gradient(pos[i] - pos[j], radius);
        // air pressure
        a += airPressure * mass[j] / (rho0 * rho0)
             * cubic_spline_kernel_gradient(pos[i] - pos[j], radius)
             /*following terms disable inner particles*/
             * length(color_grad[i]) / fmaxf(EPSILON, length(color_grad[i]));
        ++j;
    }
    return;
}

__global__ void surfaceTensionAndAirPressure_CUDA(float3* vel, float3* pos_fluid, float* mass_fluid, float3* color_grad, const int num, uint32_t* cellStart, const int3 cellSize, const float cellLength, const float radius, const float dt, const float rho0, const float color_energy_coefficient, const float airPressure)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num)
        return;
    auto a = make_float3(0.0f);
#pragma unroll
    for (auto m = 0; m < 27; __syncthreads(), ++m)
    {
        const auto cellID = particlePos2cellIdx(
            make_int3(pos_fluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
        if (cellID == (cellSize.x * cellSize.y * cellSize.z))
            continue;
        contributeSurfaceTensionAndAirPressure(a, i, pos_fluid, mass_fluid, color_grad, cellStart[cellID], cellStart[cellID + 1], radius, rho0, color_energy_coefficient, airPressure);
    }
    vel[i] += a * dt;
    return;
}
void ThermalPBFSolver::applySurfaceEffects(std::shared_ptr<SPHParticles>& fluids, const DataArray<float3>& colorGrad, const DataArray<uint32_t>& cellStartFluid, float rho0, int3 cellSize, float cellLength, float radius, float dt, float surfaceTensionIntensity, float airPressure)
{
    int num = fluids->size();
    surfaceTensionAndAirPressure_CUDA<<<(num - 1) / block_size + 1, block_size>>>(fluids->getVelPtr(),
                                                                                  fluids->getPosPtr(),
                                                                                  fluids->getMassPtr(),
                                                                                  colorGrad.addr(),
                                                                                  num,
                                                                                  cellStartFluid.addr(),
                                                                                  cellSize,
                                                                                  cellLength,
                                                                                  radius,
                                                                                  dt,
                                                                                  rho0,
                                                                                  surfaceTensionIntensity,
                                                                                  airPressure);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* HEAT TRANSFER:
* Heat transfer between particles. 
* Humidity evaluation and diffusion
* Dew Point cal
*/

// Heat conduction
__device__ void contributHeatConduction_fluids(float& a, const int i, float3* pos, int* type_fluid,
										float conductivity_i,  float* mass, float* density, float* temperature,
										const float conductivity_fluid, const float conductivity_gas,
										int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) 
	{
		float conductivity_j = conductivity_fluid;
		if (type_fluid[j] == 0) conductivity_j = conductivity_gas;
		float con = 4 * conductivity_i * conductivity_j / (conductivity_i + conductivity_j);
		float density_clamp = clamp(density[j], 0.1f, 1.05f);
		float vol_j = mass[j] / density_clamp;
		auto d_pos = pos[i] - pos[j];
		auto diff = d_pos / (dot(d_pos, d_pos) + 0.01f * smoothRaidus * smoothRaidus);
		float kernel_num = abs(dot(diff, cubic_spline_kernel_gradient(pos[i] - pos[j], smoothRaidus)));
		a += (temperature[j] - temperature[i]) * con * vol_j * kernel_num;
		++j;
	}
}

__device__ void contributHeatConduction_boundaries(float& a, const float3 pos_i, int* typeFluid, const float temperature_i,
												   float3* pos, float* mass, float density, float* temperature,
												   float conductivityFluid, float conductivityBoundary,  
												   int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) 
	{
		auto con = 4 * conductivityFluid * conductivityBoundary / (conductivityBoundary + conductivityFluid) * typeFluid[j];
		auto vol_j = mass[j] / density;
		
		auto diff = (pos_i - pos[j]) / (dot((pos_i - pos[j]), (pos_i - pos[j])) + 0.01 * smoothRaidus * smoothRaidus);
		a += con * vol_j * (temperature[j] - temperature_i) * abs(dot(diff, cubic_spline_kernel_gradient(pos_i - pos[j], smoothRaidus)));
		++j;
	}
}

__global__ void computeConduction_CUDA(float dt, float3* posFluid, float3* posBoundary, int* typeFluid, float* bufferFloat, 
	float conductivityFluid, float conductivityBoundary, float* temperatureFluid, float* temperatureBoundary, 
	float* massFluid, float* massBoundary, float* densityFluid, float densityBoundary, 
	const int num, uint32_t* cellStartFluid, uint32_t* cellStartBoundary, 
	const int3 cellSize, const float cellLength, const float smoothRaidus)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	float d_temp = 0.0f;
	float conductivity_gas = conductivityFluid * 0.04;
	float conductivity_i = conductivityFluid;

	// environment temperature for gasoues particles
	//if (typeFluid[i] == 0) {
	//	conductivity_i = conductivity_gas;
	//	float con = 30 * conductivity_i * conductivityFluid / (conductivity_i + conductivityFluid);
	//	float density_clamp = 1.0f;
	//	float vol_j = massFluid[i] / density_clamp;
	//	auto d_pos = make_float3(0.4*smoothRaidus);
	//	auto diff = d_pos / (dot(d_pos, d_pos) + 0.01f * smoothRaidus * smoothRaidus);
	//	float kernel_num = abs(dot(diff, cubic_spline_kernel_gradient(d_pos, smoothRaidus)));
	//	d_temp += (20.0f - temperatureFluid[i]) * con * vol_j * kernel_num;
	//}
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributHeatConduction_fluids(d_temp, i, posFluid, typeFluid,
			conductivity_i, massFluid, densityFluid, temperatureFluid,
			conductivityFluid, conductivity_gas,
			cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
		contributHeatConduction_boundaries(d_temp, posFluid[i], typeFluid, temperatureFluid[i],
			posBoundary, massBoundary, densityBoundary, temperatureBoundary,
			conductivityFluid, conductivityBoundary, 
			cellStartBoundary[cellID], cellStartBoundary[cellID + 1], smoothRaidus);
	}
	bufferFloat[i] = d_temp * dt * 0.00025f; 
	// 4000.0f: a coefficient related to density, heat capacity and temperature
	return;
}

__global__ void updateTemperature_CUDA(float* tem_ptr, float* latent_heat, float* d_tem,  
										float* mass, const float heat_capacity, const int num) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	__syncthreads();
	auto tem_temp = tem_ptr[i] + d_tem[i];
	auto latent_temp = 0.0f;
	if (tem_temp >= 130.0f) {
		tem_ptr[i] = 130.0f;
	}
	else {
		tem_ptr[i] = tem_temp;
	}
}

void ThermalPBFSolver::heatTransfer(float dt, std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
									const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rhoBoundary,
									int3 cellSize, float cellLength, float smoothRadius) {
	int num = fluids->size();
	//computeDensity(fluids_base, boundaries_base, cellStartFluid, cellStartBoundary, cellSize, cellLength, smoothRadius);
	computeConduction_CUDA <<<(num - 1) / block_size + 1, block_size >>> (dt, fluids->getPosPtr(), boundaries->getPosPtr(), fluids->getTypePtr(),
						bufferFloat.addr(), this->m_conductivityFluid, this->m_conductivityBoundary,
						fluids->getTempPtr(), boundaries->getTempPtr(), fluids->getMassPtr(), boundaries->getMassPtr(),
						fluids->getDensityPtr(), rhoBoundary, num, cellStartFluid.addr(), cellStartBoundary.addr(),
						cellSize, cellLength, smoothRadius);
	updateTemperature_CUDA<<<(num - 1) / block_size + 1, block_size >>>
		(fluids->getTempPtr(), fluids->getLatentPtr(), bufferFloat.addr(), fluids->getMassPtr(), 400.0f, num);
}

/*
* EXTERNAL FORCE FOR LIQUID PARTICLES: gravity and buoyancy
*/
__device__ void fluidTemperatureGradient(float3& a, const int i, float3* pos, 
										float* mass, float* density, float* temperature,
										int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd)
	{
		float3 d_pos = pos[i] - pos[j];
		a += temperature[j] * mass[j] / 1.0f * cubic_spline_kernel_gradient(pos[i] - pos[j], smoothRaidus);
		++j;
	}
}
__device__ void fluidTemperatureGradientBound(float3& a, float3 pos_i, float3* pos_bound,
	float* mass_bound, float* temperature_bound,
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd)
	{
		float3 d_pos = pos_i - pos_bound[j];
		a += 0.9f * temperature_bound[j] * mass_bound[j] / 1.0f * cubic_spline_kernel_gradient(pos_i - pos_bound[j], smoothRaidus);
		++j;
	}
}

__device__ void fluidConvectionForceFluid(float3& a, const int i, float3* pos,
	float* mass, float* density, float* temperature,
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd)
	{
		float3 d_pos = pos[i] - pos[j];
		auto dT = clamp((temperature[j] - temperature[i]), -20.0f, 20.0f);
		a += dT //* 1.0f
			* cubic_spline_kernel_gradient(pos[i] - pos[j], smoothRaidus);
		++j;
	}
}
__device__ void fluidConvectionForceBoundary(float3& a, float3 pos_i, float temperature_i, float3* pos_bound,
	float* mass_bound, float* temperature_bound,
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd)
	{
		float3 d_pos = pos_i - pos_bound[j];
		auto dT = clamp((temperature_bound[j] - temperature_i), -20.0f, 20.0f);
		a += 0.2f * dT //* 1.0f 
			* cubic_spline_kernel_gradient(pos_i - pos_bound[j], smoothRaidus);
		++j;
	}
}

__device__ void fluidAveTemperature(float& a, const int i, float3* pos,
	float* mass, float* density, float* temperature,
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd)
	{
		float3 d_pos = pos[i] - pos[j];
		a += temperature[j] * mass[j] / 1.0f * cubic_spline_kernel(length(pos[i] - pos[j]), smoothRaidus);
		++j;
	}
}

__global__ void fluidExternalForce_CUDA(float dt, float ref_temp, int* type, float3* posFluid, float3* velocity, float3* colorGrad,
	float* temperatureFluid, float* massFluid, float* densityFluid, float3 gravity,
	const int num, uint32_t* cellStartFluid, const int3 cellSize,
	const float cellLength, const float smoothRaidus) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num ) return;
	float3 GT = gravity;
	float ave_temp = 0.0f;
	__syncthreads();
	if (type[i] == 0) return;
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		fluidTemperatureGradient(GT, i, posFluid,
			massFluid, densityFluid, temperatureFluid,
			cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
		fluidAveTemperature(ave_temp, i, posFluid, massFluid, densityFluid,
			temperatureFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
	}
	float surface_n = length(colorGrad[i]) / fmaxf(EPSILON, length(colorGrad[i]));
	float temperature_i = temperatureFluid[i];
	float rho_i = densityOfFluidsAtTemperature(1.0f, temperature_i, 0);
	//if (i == 5000)
	//	printf("tempi��%f, ave_temp\n", temperature_i, ave_temp);
	ave_temp = clamp(ave_temp, temperature_i - 2, temperature_i + 2);
	float rho_T = densityOfFluidsAtTemperature(1.0f, ave_temp, 0);
	if(length(colorGrad[i]) < 1.5f)
		velocity[i] += (-GT.y / abs(EPSILON + GT.y)) * 0.005f * (ave_temp - temperature_i)* gravity * dt / 1.0f ;
	else
		velocity[i] += 0.1f * gravity * dt;
	//if(type[i] == 1)
	//	GT = (1 - 0.01 * (temperatureFluid[i] - ref_temp)) * gravity;
	//velocity[i] += GT * dt;
	return;
}

__global__ void fluidConvectionForce_CUDA(float dt, float ref_temp, int* type, float3* posFluid, float3* posBoundary,
	float3* velocity, float* temperatureFluid, float* temperature_boundary,
	float* massFluid, float* mass_boundary, float* densityFluid, float3* colorGradient, float3 gravity,
	const int num, uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float smoothRaidus) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	if (type[i] == 0) return;
	__syncthreads();
	auto acc = make_float3(0.0f);
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		fluidConvectionForceFluid(acc, i, posFluid,
			massFluid, densityFluid, temperatureFluid,
			cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
		fluidConvectionForceBoundary(acc, posFluid[i], temperatureFluid[i], posBoundary, mass_boundary, temperature_boundary,
			cellStartBoundary[cellID], cellStartBoundary[cellID + 1], smoothRaidus);
	}
	//if(colorGradient[i].y > -0.3f && colorGradient[i].y < 1.0f)
	//	velocity[i] += acc * make_float3(0.0f, -0.00001f, 0.0f) * dt;
	if(length(colorGradient[i]) < 1.5f)
		velocity[i] += acc * make_float3(0.0f, -0.00001f, 0.0f) * dt;

	return;
}

void ThermalPBFSolver::convectionForce(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries, float dt, float3 G, 
									const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, int3 cellSize, float cellLength, float smoothRadius) {
	const auto dv = dt * G;
	int num = fluids->size();	
    float3* colorGrad = surfaceBufferFloat3.addr();
	
	fluidConvectionForce_CUDA << <(num - 1) / block_size + 1, block_size >> > 
		(dt, 0, fluids->getTypePtr(),
		fluids->getPosPtr(), boundaries->getPosPtr(),
		fluids->getVelPtr(), 
		fluids->getTempPtr(), boundaries->getTempPtr(),
		fluids->getMassPtr(), boundaries->getMassPtr(),
		fluids->getDensityPtr(), colorGrad,
		G, num, 
		cellStartFluid.addr(), cellStartBoundary.addr(), cellSize, cellLength, smoothRadius
	);
}

/*
* EXTERNAL FORCE FOR GASEOUS PARTICLES.
* Force list:
* Gravity(down), buoyancy(up) 
* surface tension(gas interior)
* drag force(viscosity like)
* Be like:
*	  buoyancy give bubble particles a force to rise. 
*     small bubbles(less gaseous particles) have high surface tension(high average acc due to tension), preventing them from rising.
*     big bubbles(more gaseous particles) have less surface tension(low average acc due to tension), starting to rising.
*	  drag force will make gaseous particles to move following the liquid flow.
*/
void ThermalPBFSolver::bubbleExternalForce(float dt, std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
	const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
	float rho0, float3 G, int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius) {
	auto num = fluids->size();
	// tension (cohension)
	auto fluid_vel = fluids->getVelPtr();
	auto fluid_rho = fluids->getDensityPtr();
	auto fluid_type = fluids->getTypePtr();
	auto fluid_pos = fluids->getPosPtr();
	auto fluid_mass = fluids->getMassPtr();
	auto bound_type = boundaries->getTypePtr();
	auto bound_pos = boundaries->getPosPtr();

	liquidVaporColorGradient_CUDA << < (num - 1) / block_size + 1, block_size >> > (
		bufferFloat3.addr(),
		fluid_pos, fluid_mass, fluid_rho,
		fluid_type, rho0, num, cellStartFluid.addr(),
		cellSize, cellLength, smoothRadius
		);
	
	copyFloat3Norm << < (num - 1) / block_size + 1, block_size >> >(
		outColorGradient.addr(),
		bufferFloat3.addr(),
		num
	);

	liquidVaporTension_CUDA << < (num - 1) / block_size + 1, block_size >> > (
		dt, fluid_vel, bufferFloat3.addr(),
		fluid_mass, fluid_pos, fluid_type, num, rho0,
		cellStartFluid.addr(), cellSize, cellLength, smoothRadius);
	
	artificialCohension_CUDA << < (num - 1) / block_size + 1, block_size >> > 
		(dt, fluid_vel, fluid_rho, bufferFloat3.addr(),
		fluid_pos, bound_pos, fluid_type, bound_type, num,
		cellStartFluid.addr(), cellStartBoundary.addr(),
		cellSize, cellLength,
		smoothRadius, rho0);

	artificialBuoyancy_CUDA << < (num - 1) / block_size + 1, block_size >> > 
		(dt, fluid_vel, fluid_rho,
		fluid_pos, fluid_type, G, num,
		cellStartFluid.addr(), cellSize, cellLength,
		smoothRadius, rho0);

	liquidVaporDrag_CUDA << < (num - 1) / block_size + 1, block_size >> > 
		(dt, bufferFloat3.addr(), fluid_vel, fluid_rho, fluid_mass,
		fluid_pos, fluid_type,
		num, cellStartFluid.addr(), cellSize, cellLength,
		smoothRadius, rho0);

	thrust::transform(thrust::device,
		fluids->getVelPtr(), fluids->getVelPtr() + num,
		bufferFloat3.addr(),
		fluids->getVelPtr(),
		thrust::plus<float3>()
	);

}

/*
* Mass transfer : vaporization
* Necessary condition 1: Enough Latent heat 2260kj/kg.
* Necessary condition 2: Near a type-0 particle
* Vaporization should produce one more particle (unsolved)
* latent heat should be smoothed 
*/

// OLD
__device__ 
void findNucleationToTrans_fluid(bool& trans, const int i, float3* pos,
	int* type, 
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) {
		auto distance = length(pos[i] - pos[j]);
		if (type[j] == 0 && distance <= smoothRaidus * 0.7f) {
			trans = true;
			return;
		}
		j++;
	}
}
__device__ 
void findNucleationToTrans_boundary(bool& trans, float3 pos_i, float3* pos_boundary,
	int* type, 
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) {
		auto distance = length(pos_i - pos_boundary[j]);
		if (type[j] == 0 && distance <= smoothRaidus * 0.7f) {
			trans = true;
			return;
		}
		++j;
	}
}
__global__
void vaporization_CUDA(int* type_ptr, float* latent_ptr, float* tem_ptr,
	// const para
	int* type_temp, int* type_boundary, 
	const float trans_heat, 
	float3* posFluid, float3* posBoundary,
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float smoothRaidus, const int num) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= num) return;
	if (type_temp[i] != 1)
		return;
	bool trans = false;

	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		bool trans_temp = false;
		findNucleationToTrans_fluid(trans_temp, i, posFluid, 
									type_temp, 
									cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
		trans = trans_temp || trans;
		findNucleationToTrans_boundary(trans_temp, posFluid[i], posBoundary, 
									   type_boundary, 
									   cellStartBoundary[cellID], cellStartBoundary[cellID + 1], smoothRaidus);
		trans = trans_temp || trans;
	}
	// tranfer 
	if (trans) {
		if (tem_ptr[i] > 100.0f) {
			latent_ptr[i] += (tem_ptr[i] - 100.0f) * 4.0f;
			tem_ptr[i] = 100.0f;
		}
		if (latent_ptr[i] > trans_heat) {
			type_ptr[i] = 0;
			latent_ptr[i] = 0.0f;
		}
		// printf("trans id: , %d\n", i);
	}
}

// NEW
__global__
void vaporizationGasNucleation_CUDA(int* valid_fluid, int* type_ptr, float* latent_ptr, float* tem_ptr,
	size_t* index_counter, 
	// const para
	const float trans_heat,
	float3* posFluid, float3* posBoundary,
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float smoothRadius, const int num){
	
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type_ptr[i] != 0) return;

	// grab latent heat
	auto latent_heat = (tem_ptr[i] - 100.0f) * 600.0f;
	tem_ptr[i] = 100.0f;
	latent_ptr[i] += latent_heat;

	if (latent_ptr[i] > trans_heat) {
		latent_ptr[i] -= trans_heat;
		int index = atomicAdd(index_counter, 1);
		posFluid[index] = make_float3(posFluid[i].x, posFluid[i].y + 0.5f * smoothRadius, posFluid[i].z);
		tem_ptr[index] = 100.0f;
		type_ptr[index] = 0;
		latent_ptr[index] = 0.0f;
		valid_fluid[index] = 1;
	}
}

__device__
void absorbHeatFromFluid(float& get_heat, float3 pos_i, float3* posFluid, int* type_temp, float* tem_ptr,
	unsigned int j, unsigned int cellEnd, float smoothRadius) {
	while (j < cellEnd) {
		auto distance = length(pos_i - posFluid[j]);
		if (type_temp[j] == 1 && distance <= smoothRadius) {
			if (tem_ptr[j] > 100.0f) {
				get_heat += (tem_ptr[j] - 100.0f) * 600.0f;
				tem_ptr[j] = 100.0f;
			}
		}
		j++;
	}
}

__global__
void vaporizationBoundNucleation_CUDA(int* valid_fluid, int* type_fluid, int* type_bound,
	float* latent_fluid, float* latent_bound, float* tem_ptr, 
	size_t* index_counter, 
	// const para
	const float trans_heat,
	float3* posFluid, float3* posBoundary,
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float smoothRadius, const int num) {

	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type_bound[i] != 0) return;
	float get_heat = 0.0f;
	auto pos_i = posBoundary[i];
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posBoundary[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		absorbHeatFromFluid(get_heat, pos_i, posFluid, type_fluid, tem_ptr, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRadius);
	}
	latent_bound[i] += get_heat;

	if (latent_bound[i] >= trans_heat) {
		size_t index = atomicAdd(index_counter, 1);
		int dir = index % 3;
		posFluid[index] = make_float3(pos_i.x + (1 - dir) * 0.25f * smoothRadius, pos_i.y + 0.5f * smoothRadius, (dir - 1) * 0.25f * smoothRadius +pos_i.z);
		tem_ptr[index] = 100.0f;
		type_fluid[index] = 0;
		latent_fluid[index] = 0.0f;
		valid_fluid[index] = 1;
		latent_bound[i] -= trans_heat;
	}
}

void ThermalPBFSolver::massTransferVaporization(
	std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
	const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
	float rho0, float trans_heat, 
	int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius
) {
	auto num_fluid = fluids->size();
	size_t particle_index = num_fluid;
	size_t* index_ptr = fluids->getIndexPtr();

	// fluid attributes pointer
	auto type_fluid = fluids->getTypePtr();
	auto pos_fluid = fluids->getPosPtr();
	auto tem_fluid = fluids->getTempPtr();
	auto latent_fluid = fluids->getLatentPtr();
	auto vel_fluid = fluids->getVelPtr();
	auto valid_fluid = fluids->getValidPtr();
	// boundary attributes pointer
	auto type_bound = boundaries->getTypePtr();
	auto latent_bound = boundaries->getLatentPtr();
	auto pos_bound = boundaries->getPosPtr();
	auto tem_bound = boundaries->getTempPtr();
	

	vaporizationBoundNucleation_CUDA << <(num_fluid - 1) / block_size + 1, block_size >> >
		(valid_fluid, type_fluid, type_bound,
			latent_fluid, latent_bound,
			tem_fluid,
			index_ptr,
			// const var
			trans_heat, 
			pos_fluid, pos_bound,
			cellStartFluid.addr(), cellStartBoundary.addr(),
			cellSize, cellLength, smoothRadius, num_fluid);

	vaporizationGasNucleation_CUDA <<<(num_fluid - 1) / block_size + 1, block_size >>>
		(valid_fluid, type_fluid, latent_fluid, tem_fluid,
		index_ptr,
		// const var
		trans_heat, pos_fluid, pos_bound,
		cellStartFluid.addr(), cellStartBoundary.addr(), 
		cellSize, cellLength, smoothRadius, boundaries->size());

	//condensationForGasoues_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
}
/*
* Mass transfer : condensation
* Necessary condition 1 : type - 0 particle's latent heat
*/
__global__
void condensationForGasoues_CUDA(
	// indicate attributes
	int* valid_fluid, int* type_fluid, int* type_bound,
	// phys attributes
	float* latent_fluid, float* tem_ptr,
	float3* posFluid, float3* posBoundary, float3* velFluid,
	// const phys para
	const float trans_heat, const float heat_capacity,
	// const neigh para
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float  smoothRadius, const int num
) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type_fluid[i] != 0) return;

	__syncthreads();

	float tem_i = tem_ptr[i];
	if (tem_i < 100.0f) {
		latent_fluid[i] += (tem_ptr[i] - 100.0f) * heat_capacity;
		tem_ptr[i] = 100.0f;
	}
	// condensation: remove the gas particle, set valid flag to 0
	if (latent_fluid[i] < trans_heat) {
		valid_fluid[i] = 0;
	}
}
/*
* Weighting Humidity in air by fluids particle
*/
__device__
void humiditySpread(float& dRH, int& neigh_num,
	const float3 pos_i, const float humidity_i, const float3* posFluid, const float* humidity, const float* mass,
	const float coefficient, const float rho0,
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) {
		auto x_ij = pos_i - posFluid[j];
		if (smoothRaidus > length(x_ij) ){
			dRH += coefficient * (humidity[j] - humidity_i) * mass[j] * cubic_spline_kernel_laplacian(length(x_ij), smoothRaidus) / rho0;
			neigh_num++;
		}
		j++;
	}
}

__global__ 
void humidityEvaluate_CUDA(
	const int* type_fluid, const float* massFluid, const float3* posFluid, const float3* posBound,
	float* humidityFluid, float* humidityBound, 
	// sim para
	const float dt,
	// const neigh para
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float  smoothRadius, const int num
) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type_fluid[i] != 0) return;
	__syncthreads();
	int neigh_fluid_num = 0;
	float ambient_air_ratio = 0.4f;
	float humidity_i = humidityFluid[i];
	float dRH = 0.0f;
	float3 pos_i = posFluid[i];
	float mass_i = massFluid[i];
	float diffusion_coefficient = 1.4f;
	float virtual_humidity = 0.3f;
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		// Humidity spread and neighbor cal 
		humiditySpread(dRH, neigh_fluid_num, pos_i, humidity_i, posFluid, humidityFluid, massFluid, diffusion_coefficient, 1.0f,
			cellStartFluid[cellID], cellStartFluid[cellID + 1],smoothRadius);
	}
	if (neigh_fluid_num > 30) {
		humidityFluid[i] = 1.0f;
	}
	else {
		
		dRH += diffusion_coefficient * (virtual_humidity - humidity_i) * mass_i *
			cubic_spline_kernel_laplacian(ambient_air_ratio * smoothRadius, smoothRadius) / 1.0f;
		humidityFluid[i] += dRH * dt;
	}
	
}

__device__ 
void findNearestBound(int& bound_index,const float3 pos_i, const float3* posBound, 
	int j, const uint32_t cellEnd, const float smoothRaidus) {
	float dis_near = 0.7 * smoothRaidus;
	while (j < cellEnd) {
		auto x_ij = pos_i - posBound[j];
		if (dis_near > length(x_ij)) {
			dis_near = length(x_ij);
			bound_index = j;
		}
		j++;
	}
}

__global__
void condensationOnWall_CUDA(
	int* valid, 
	const int* type, const float3* posFluid, const float3* posBound,
	float* humidityFluid, float* humidityBound, float* tem_fluid,
	// 
	const float temperature,
	// const neigh para
	uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float  smoothRadius, const int num
) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= num || type[i] == 1) return;

	int bound_index = -1;
	if (tem_fluid[i] > calDewPoint(temperature, humidityFluid[i]))
		return;
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		// Humidity spread and neighbor cal 
		findNearestBound(bound_index, posFluid[i], posBound, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], smoothRadius);
	}
	if (bound_index > 0) {
		valid[i] = 0;
		atomicAdd(humidityBound + bound_index, 1.0f);
	}
}

__global__ 
void coalesceOnWall_CUDA(size_t* index_counter, const float3* posBound, float*humidtyFluid, float* humidityBound,
	float3* posFluid, int* type_fluid, float* tem_fluid, float* latent_fluid, int* valid_fluid,
	const float densityRatio, const float dt,
	uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float  smoothRadius, const int num) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto pos_i = posBound[i]; 
	//	auto dRH = 0.0f;
	//#pragma unroll
	//	for (auto m = 0; m < 27; __syncthreads(), ++m)
	//	{
	//		const auto cellID = particlePos2cellIdx(
	//			make_int3(posBound[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
	//		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
	//		
	//	}
	//	humidityBound[i] += dRH * dt;
	if (humidityBound[i] > densityRatio) {
		humidityBound[i] -= densityRatio;
		size_t index = atomicAdd(index_counter, 1);
		posFluid[index] = make_float3(pos_i.x , pos_i.y - 0.5f * smoothRadius, pos_i.z);
		tem_fluid[index] = 20.0f;
		type_fluid[index] = 1;
		latent_fluid[index] = 0.0f;
		valid_fluid[index] = 1;
		humidtyFluid[index] = 1.0f;
	}
}

__global__
void resetValidFlag_CUDA(int* valid, const int num, const int new_num) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	if (i >= new_num) {
		valid[i] = 0;
	}
	else {
		valid[i] = 1;
	}
}

void ThermalPBFSolver::massTransferCondensation(float dt,
	std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
	const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
	float rho0, float trans_heat,
	int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius
) {
	auto num_fluid = fluids->size();
	auto num_bound = boundaries->size();

	// fluid attributes pointer
	auto type_fluid = fluids->getTypePtr();
	auto pos_fluid = fluids->getPosPtr();
	auto tem_fluid = fluids->getTempPtr();
	auto latent_fluid = fluids->getLatentPtr();
	auto vel_fluid = fluids->getVelPtr();
	auto valid_fluid = fluids->getValidPtr();
	auto humidity_fluid = fluids->getHumidityPtr();
	auto mass_fluid = fluids->getMassPtr();
	// boundary attributes pointer
	auto type_bound = boundaries->getTypePtr();
	auto latent_bound = boundaries->getLatentPtr();
	auto pos_bound = boundaries->getPosPtr();
	auto tem_bound = boundaries->getTempPtr();
	auto humidity_bound = boundaries->getHumidityPtr();

	// This function evaluates every gaseous particle's humidity
	// humidity of particles around liquid will be set to 100%
	// otherwise humidity will diffuse to air.
	humidityEvaluate_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
	(type_fluid, mass_fluid, pos_fluid, pos_bound, humidity_fluid, humidity_bound, dt,
		cellStartFluid.addr(), cellStartBoundary.addr(), cellSize, cellLength, smoothRadius, num_fluid);

	// This function calculate dew point at every gasoues particle,
	// particles near boundary below dew point will condensation,
	// boundary particles will receive the mass.
	condensationOnWall_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> > 
		(valid_fluid, type_fluid, pos_fluid, pos_bound, humidity_fluid, humidity_bound, tem_fluid,
		100.0f, cellStartFluid.addr(), cellStartBoundary.addr(), cellSize, cellLength, smoothRadius, num_fluid);
	 
	// This function will find colden gaseous particles under water to trans, mark them and record num of particles needed to remove.
	// Set Valid flag to 0 as soon as removing.
	if (scene_type != 3) {
		condensationForGasoues_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(valid_fluid, type_fluid, type_bound, // indicate attrs
				latent_fluid, tem_fluid,  // phys attrs
				pos_fluid, pos_bound, vel_fluid,
				-trans_heat, 3600000.0f, // phys paras
				cellStartFluid.addr(), cellStartBoundary.addr(), // search cells
				cellSize, cellLength, smoothRadius, num_fluid // search paras
				);
	}
	// Execute a prefix sum to get new indices for particles 
	int last_valid_flag = 0;
	checkCudaErrors(cudaMemcpy(&last_valid_flag, valid_fluid + num_fluid - 1, sizeof(int), cudaMemcpyDeviceToHost));

	thrust::exclusive_scan(thrust::device, valid_fluid, valid_fluid + num_fluid + 1, valid_fluid);

	int res_prefix = 0;
	checkCudaErrors(cudaMemcpy(&res_prefix, valid_fluid + num_fluid - 1, sizeof(int), cudaMemcpyDeviceToHost));


	int new_num = res_prefix;
	if (last_valid_flag) {
		new_num = res_prefix + 1;
	}

	if (new_num != num_fluid) {
		// Sort all arrays: position, velocity, temperature, type, latent heat
		checkCudaErrors(cudaMemcpy(bufferFloat3.addr(), pos_fluid, sizeof(float3) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat3.addr(), pos_fluid, valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferFloat3.addr(), vel_fluid, sizeof(float3) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat3.addr(), vel_fluid, valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferFloat.addr(), outColorGradient.addr(), sizeof(float) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat.addr(), outColorGradient.addr(), valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferFloat.addr(), latent_fluid, sizeof(float) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat.addr(), latent_fluid, valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferFloat.addr(), tem_fluid, sizeof(float) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat.addr(), tem_fluid, valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferFloat.addr(), humidity_fluid, sizeof(float) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferFloat.addr(), humidity_fluid, valid_fluid, num_fluid);

		checkCudaErrors(cudaMemcpy(bufferInt2.addr(), type_fluid, sizeof(int) * num_fluid, cudaMemcpyDeviceToDevice));
		reassignAttrutes_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> >
			(bufferInt2.addr(), type_fluid, valid_fluid, num_fluid);

		fluids->reduce_size(num_fluid - new_num);
	}
	// resete valid attr
	resetValidFlag_CUDA << < (num_fluid - 1) / block_size + 1, block_size >> > (valid_fluid, num_fluid, new_num);
	
	num_fluid = fluids->size();
	size_t particle_index = num_fluid;
	size_t* index_ptr = fluids->getIndexPtr();

	coalesceOnWall_CUDA << < (num_bound - 1) / block_size + 1, block_size >> >
		(index_ptr, pos_bound, humidity_fluid, humidity_bound, pos_fluid, type_fluid, tem_fluid, latent_fluid, valid_fluid,
			5.0f, dt, cellStartBoundary.addr(), cellSize, cellLength, smoothRadius, num_bound);
	int check_res = fluids->check_size();
}

}