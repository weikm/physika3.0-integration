#pragma once
#ifndef _THERMAL_KERNELS_CUH_
#define _THERMAL_KERNELS_CUH_
#include <cuda_runtime.h>
#include <cstdio>

#include "Util/CudaFunction.cuh"
#define _TYPE_ACCESS_ type[i]
/*
* Reassign attrbutes to their new index
*/
template<class T>
__global__
void reassignAttrutes_CUDA(
	T* src, T* tar,
	const int* prefix_res, const int num) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || prefix_res[i] == prefix_res[i + 1]) return;
	unsigned int new_id = prefix_res[i];
	tar[new_id] = src[i];
}

__global__
void copyFloat3Norm(float* des, float3* src, int num) {
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	des[index] = length(src[index]);
}

static inline __device__ void modifyMassAndDensity(float& mass, float& density, int type) {
	if (type == 0) {
		mass = mass / 2;
		density = density / 2;
	}
}

static inline __device__ void contributeXSPHViscosityTP(float3* a, const int i, float3* pos, float3* vel,
	float* mass, int j/*cellStart*/, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) {
		*a += mass[j] * (vel[j] - vel[i]) * cubic_spline_kernel(length(pos[i] - pos[j]), smoothRaidus);
		++j;
	}
	return;
}
/// <summary>
/// 
/// </summary>
/// <param name="vel"> velocity </param>
/// <param name="pos"> position </param>
/// <param name="mass"> mass of fluids </param>
/// <param name="num"> size </param>
/// <param name="cellStart"> neighbor query </param>
/// <param name="cellSize"> neighbor query </param>
/// <param name="cellLength"> neighbor query </param>
/// <param name="smoothRaidus"></param>
/// <param name="visc"> viscousity coefficient </param>
/// <param name="rho0"> rest density </param>
/// <returns></returns>
__global__ void XSPHViscosityTP_CUDA(float3* vel, float3* pos, int*type,
	float* mass, const int num, uint32_t* cellStart, const int3 cellSize,
	const float cellLength, const float smoothRaidus, const float visc, const float rho0) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto a = make_float3(0.0f);

	// different mass and density
	auto mass_i = mass[i];
	auto rest_rho = rho0;
	modifyMassAndDensity(mass_i, rest_rho, _TYPE_ACCESS_);

	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m) {
		const auto cellID = particlePos2cellIdx(make_int3(pos[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
			cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeXSPHViscosityTP(&a, i, pos, vel, mass, cellStart[cellID], cellStart[cellID + 1], smoothRaidus);
	}

	vel[i] += visc * a / rest_rho;
	return;
}


static inline __device__ void contributeDeltaPosTP_fluid(float3& a, const int i, float3* pos, float* lambda, float mass, int j, const uint32_t cellEnd, const float smoothRaidus)
{
	while (j < cellEnd)
	{
		a += mass * (lambda[i] + lambda[j]) * cubic_spline_kernel_gradient(pos[i] - pos[j], smoothRaidus);
		++j;
	}
	return;
}

static inline __device__ void contributeDeltaPosTP_boundary(float3& a, const float3 pos_i, float3* pos, const float lambda_i, float* mass, int j, const uint32_t cellEnd, const float smoothRaidus)
{
	while (j < cellEnd)
	{
		a += mass[j] * (lambda_i)*cubic_spline_kernel_gradient(pos_i - pos[j], smoothRaidus);
		++j;
	}
	return;
}
/// <summary>
/// 
/// </summary>
/// <param name="deltaPos"></param>
/// <param name="posFluid"></param>
/// <param name="posBoundary"></param>
/// <param name="lambda"></param>
/// <param name="massFluid"></param>
/// <param name="massBoundary"></param>
/// <param name="num"></param>
/// <param name="cellStartFluid"></param>
/// <param name="cellStartBoundary"></param>
/// <param name="cellSize"></param>
/// <param name="cellLength"></param>
/// <param name="smoothRaidus"></param>
/// <param name="rho0"></param>
/// <returns></returns>
__global__ void computeDeltaPosTP_CUDA(float3* deltaPos, float3* posFluid, float3* posBoundary, float* lambda, int* type,
	float* massFluid, float* massBoundary, const int num, uint32_t* cellStartFluid, uint32_t* cellStartBoundary, const int3 cellSize,
	const float cellLength, const float smoothRaidus, const float rho0)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto dp = make_float3(0.0f);
	// different mass and density
	auto mass_i = massFluid[i];
	auto rest_rho = rho0;
	modifyMassAndDensity(mass_i, rest_rho, _TYPE_ACCESS_);

	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDeltaPosTP_fluid(dp, i, posFluid, lambda, mass_i, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
		contributeDeltaPosTP_boundary(dp, posFluid[i], posBoundary, lambda[i], massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], smoothRaidus);
	}

	deltaPos[i] = dp / rest_rho;
	return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// two-phase density estimation fluids (ignore mass difference, buoyancy ni kakeru)
static inline __device__ void contributeDensityLambdaTP_F(float& density, float3& gradientSum, float& sampleLambda, const float3 pos_i, float3* pos,
	float mass, int j, const uint32_t cellEnd, const bool rho0, const float smoothRaidus)
{
	while (j < cellEnd)
	{
		density += mass * cubic_spline_kernel(length(pos_i - pos[j]), smoothRaidus);
		const auto gradient = -mass * cubic_spline_kernel_gradient(pos_i - pos[j], smoothRaidus) / rho0;
		gradientSum -= gradient;
		sampleLambda += dot(gradient, gradient);
		++j;
	}
	return;
}

// two-phase density estimation boundary
static inline __device__ void contributeDensityLambdaTP_B(float& density, float3& gradientSum, float& sampleLambda, const float3 pos_i, float3* pos,
	float* mass, int j, const uint32_t cellEnd, const bool rho0, const float smoothRaidus)
{
	while (j < cellEnd)
	{
		density += mass[j] * cubic_spline_kernel(length(pos_i - pos[j]), smoothRaidus);
		const auto gradient = -mass[j] * cubic_spline_kernel_gradient(pos_i - pos[j], smoothRaidus) / rho0;
		gradientSum -= gradient;
		sampleLambda += dot(gradient, gradient);
		++j;
	}
	return;
}
/// <summary>
/// 
/// </summary>
/// <param name="density"></param>
/// <param name="lambda"></param>
/// <param name="posFluid"></param>
/// <param name="massFluid"></param>
/// <param name="num"></param>
/// <param name="cellStartFluid"></param>
/// <param name="cellSize"></param>
/// <param name="posBoundary"></param>
/// <param name="massBoundary"></param>
/// <param name="cellStartBoundary"></param>
/// <param name="cellLength"></param>
/// <param name="smoothRaidus"></param>
/// <param name="rho0"></param>
/// <param name="relaxation"></param>
/// <returns></returns>
__global__ void computeDensityLambdaTP_CUDA(float* density, float* lambda, int* type,
	float3* posFluid, float* massFluid, const int num, uint32_t* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, uint32_t* cellStartBoundary,
	const float cellLength, const float smoothRaidus, const float rho0, const float relaxation)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto gradientSum = make_float3(0.0f);
	auto sampleLambda = 0.0f;
	auto den = 0.0f;
	// different mass and density
	auto mass_i = massFluid[i];
	auto rest_rho = rho0;
	modifyMassAndDensity(mass_i, rest_rho, _TYPE_ACCESS_);

	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDensityLambdaTP_F(den, gradientSum, sampleLambda, posFluid[i], posFluid, mass_i, cellStartFluid[cellID], cellStartFluid[cellID + 1], rest_rho, smoothRaidus);
		contributeDensityLambdaTP_B(den, gradientSum, sampleLambda, posFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], rest_rho, smoothRaidus);
	}

	density[i] = den;
	lambda[i] = (den > rest_rho) ?
		(-(den / rest_rho - 1.0f) / (dot(gradientSum, gradientSum) + sampleLambda + EPSILON))
		: 0.0f;
	lambda[i] *= relaxation;
	return;
}

__global__ void externalForces_Cuda(float3* vel, float3* ex_force, float3 G, const float dt, const int num) 
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num)
        return;
    vel[i] += (G + ex_force[i]) * dt;
}

// Enforce boundary
__global__ void enforceBoundary_CUDA(float3* pos, const int num, const float3 spaceSize)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num)
        return;
    if (pos[i].x <= spaceSize.x * .00f)
    {
        pos[i].x = spaceSize.x * .00f;
    }
    if (pos[i].x >= spaceSize.x * .99f)
    {
        pos[i].x = spaceSize.x * .99f;
    }
    if (pos[i].y <= spaceSize.y * .00f)
    {
        pos[i].y = spaceSize.y * .00f;
    }
    if (pos[i].y >= spaceSize.y * .99f)
    {
        pos[i].y = spaceSize.y * .99f;
    }
    if (pos[i].z <= spaceSize.z * .00f)
    {
        pos[i].z = spaceSize.z * .00f;
    }
    if (pos[i].z >= spaceSize.z * .99f)
    {
        pos[i].z = spaceSize.z * .99f;
    }
    return;
}

__global__ void enforceBoundaryAdvect_CUDA(float3* pos, float3* vel, const int num, const float3 spaceSize)
{
    const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num)
        return;
    if (pos[i].x <= spaceSize.x * .00f)
    {
        pos[i].x = spaceSize.x * .00f;
        vel[i].x = fmaxf(vel[i].x, 0.0f);
    }
    if (pos[i].x >= spaceSize.x * .99f)
    {
        pos[i].x = spaceSize.x * .99f;
        vel[i].x = fminf(vel[i].x, 0.0f);
    }
    if (pos[i].y <= spaceSize.y * .00f)
    {
        pos[i].y = spaceSize.y * .00f;
        vel[i].y = fmaxf(vel[i].y, 0.0f);
    }
    if (pos[i].y >= spaceSize.y * .99f)
    {
        pos[i].y = spaceSize.y * .99f;
        vel[i].y = fminf(vel[i].y, 0.0f);
    }
    if (pos[i].z <= spaceSize.z * .00f)
    {
        pos[i].z = spaceSize.z * .00f;
        vel[i].z = fmaxf(vel[i].z, 0.0f);
    }
    if (pos[i].z >= spaceSize.z * .99f)
    {
        pos[i].z = spaceSize.z * .99f;
        vel[i].z = fminf(vel[i].z, 0.0f);
    }
    return;
}

// Artificial Cohension to make gaseous particles to attract each other.
static inline __device__ void gaseousCohension(float3& acc, float* density, float3* posFluid, int* type, const float3 pos_i,
												int j, const uint32_t cellEnd,  const float smoothRaidus) {
	while (j < cellEnd)
	{
		// acc += cohension_coefficient * mass_i * density[j] * (pos_i - posFluid[j]);
		auto x_ij = pos_i - posFluid[j];
		if(type[j] == 0 && length(x_ij) <= smoothRaidus)
			acc += 220.0f * (pos_i - posFluid[j]);
		++j;
	}
	return;
}
static inline __device__ void gaseousCohension_bound(float3& acc, float* density, float3* posBound, int* type, const float3 pos_i,
												int j, const uint32_t cellEnd,  const float smoothRaidus) {
	while (j < cellEnd)
	{
		// acc += cohension_coefficient * mass_i * density[j] * (pos_i - posFluid[j]);
		auto x_ij = pos_i - posBound[j];
		if(type[j] == 0 && length(x_ij) <= smoothRaidus)
			acc += 2000.0f * x_ij;
		++j;
	}
	return;
}

__global__ void artificialCohension_CUDA(const float dt, float3* velocity, float* density, float3* color_gradient,
														float3* posFluid, float3* posBound, int* type, int* type_bound,
														const int num, uint32_t* cellStartFluid, uint32_t* cellStartBound,
														const int3 cellSize, const float cellLength, const float smoothRadius, 
														const float rho0) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type[i] == 1) return;
	if (length(color_gradient[i]) < 0.3f) return;
	float3 acc = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		gaseousCohension(acc, density, posFluid, type, posFluid[i], cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRadius);
		gaseousCohension_bound(acc, density, posBound, type_bound, posFluid[i], cellStartBound[cellID], cellStartBound[cellID + 1], smoothRadius);
	}
	velocity[i] -= acc * dt;
	return;
}

static inline __device__
void countNearborNum(int& count, const float3 pos_i, float3* pos, int j, const uint32_t cellEnd, float smoothRadius) {
	while (j < cellEnd) {
		if (length(pos_i - pos[j]) < smoothRadius) {
			++count;
		}
		++j;
	}
}
/*
* Buoyancy for gaseous particles trapped in liquid

* @param[in] in parameter to read only
* @param[in,out] in_out parameter to read and write
* @param[out] out parameter to write only
*/
__global__ void artificialBuoyancy_CUDA(const float dt, float3* velocity, float* density,
	float3* posFluid, int* type, const float3 G,
	const int num, uint32_t* cellStartFluid, const int3 cellSize,
	const float cellLength, const float smoothRaidus, const float rho0) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num || type[i] == 1) return;
	float3 acc = make_float3(0.0f);
	const int max_count = 50;
	const int min_count = 3;
	int nCount = 0;
	//
	__syncthreads();
	#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		countNearborNum(nCount, posFluid[i], posFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRaidus);
	}
	auto den = clamp(density[i], rho0 / 2, rho0);
	float ratio = 0.0f;
	if (nCount > 20)
		ratio = 1.0f;
	else
		ratio = (float)nCount / 30.0f + 0.25f;
	auto vel_i = velocity[i] - (0.05f * G + 4.0f * G * ratio) * dt;
	if (nCount < 10) {
		vel_i = 0.95f * vel_i;
	}
	velocity[i] = vel_i;
	return;
}

/*
* Surface tension between liquid and vapor
* Color gradients needed 
* 
* @param[in] in parameter to read only
* @param[in,out] in_out parameter to read and write
* @param[out] out parameter to write only
*/

static inline __device__ void computeLiquidVaporColorGradient(float3& color_gradient, float3* posFluid, int* type, 
															float* mass, const int i, const float rho0,
															int j, const uint32_t cellEnd, const float smoothRaidus){
	while (j < cellEnd) {
		color_gradient += mass[j] / rho0 * (type[j] - type[i]) * cubic_spline_kernel_gradient(posFluid[i] - posFluid[j], smoothRaidus);
		j++;
	}
}

/*
* Cal color gradient to identify liquid-vapor interface
*
* @param[in] in parameter to read only
* @param[in,out] in_out parameter to read and write
* @param[out] out parameter to write only
*/
__global__ void liquidVaporColorGradient_CUDA(float3* bufferColorGrad, float3* posFluid, float* massFluid, float* density,
															int* type, float rho0, int num, uint32_t* cellStartFluid, 
															const int3 cellSize, const float cellLength, const float smoothRadius) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto color_grad = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		computeLiquidVaporColorGradient(color_grad, posFluid, type, massFluid, i, rho0,
										cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRadius);
	}
	bufferColorGrad[i] = color_grad;
}

static inline __device__
void liquidVaporTension(float3& diver_color, float3* color_gradient, float3* posFluid, int* type,
						float* mass, const int i, const float rho0,
						int j, const uint32_t cellEnd, const float smoothRaidus) {
	while (j < cellEnd) {
		diver_color += (type[j] - type[i]) * mass[j] / rho0 * 
			dot((color_gradient[j] - color_gradient[i]), cubic_spline_kernel_gradient(posFluid[i] - posFluid[j], smoothRaidus));
		j++;
	}
}
/*
* Tension on liquid-vapor interface
*
* @param[in] in parameter to read only
* @param[in,out] in_out parameter to read and write
* @param[out] out parameter to write only
*/
__global__ void liquidVaporTension_CUDA(const float dt, float3* vevlocity, float3* bufferColorGrad,
	float* mass, float3* posFluid, int* type, const int num, const float rho0,
	uint32_t* cellStartFluid, const int3 cellSize, const float cellLength, const float smoothRadius) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;
	auto color_grad_i = bufferColorGrad[i];
	auto color_mag = fmaxf(length(color_grad_i), EPSILON);
	auto normal = color_grad_i / color_mag;
	auto diver_color = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		liquidVaporTension(diver_color, bufferColorGrad, posFluid, type, mass, i, rho0, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRadius);
	}
	vevlocity[i] -= 0.05f * diver_color * normal * dt;
}

// Vapor - liquid drag force
static inline __device__
void liquidVaporDragKernel(float3& drag_acc, const int i, float3* posFluid, float3* vel, int* type, float* massFluid, const float rho0, 
							int j, const uint32_t cellEnd, const float smoothRadius) {
	while (j < cellEnd) {			
		auto d_vel = vel[i] - vel[j];
		auto x_ij = posFluid[i] - posFluid[j];
		auto PI_ij = dot(d_vel, x_ij) / (length(x_ij) + EPSILON * smoothRadius * smoothRadius);
		int type_ij = type[i] - type[j];
		if (PI_ij > 0 && type_ij != 0) {
			drag_acc += massFluid[j] * PI_ij * cubic_spline_kernel_gradient(x_ij, smoothRadius) / rho0 / 1.001f;
		}
		j++;
	}
}
/*
* Drag force between vapor and liquid
*
* @param[in] dt: simulation time step
* @param[in] density: input particles density
* @param[in] mass: input particles mass
* @param[in] velocity: input particles velocity
* @param[in,out] deltaV: acceleration casued by drag force 
*/
__global__ void liquidVaporDrag_CUDA(const float dt, float3* deltaV, float3* velocity, float* density, float* mass,
	float3* posFluid, int* type,
	const int num, uint32_t* cellStartFluid,
	const int3 cellSize, const float cellLength, const float smoothRadius,
	const float rho0) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num) return;

	auto drag_acc = make_float3(0.0f);
	auto drag_coefficent = 8.0f;
	if (type[i] == 1)
		drag_coefficent = 3.0f;
	__syncthreads();

#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		liquidVaporDragKernel(drag_acc, i, posFluid, velocity, type, mass, rho0, cellStartFluid[cellID], cellStartFluid[cellID + 1], smoothRadius);
	}
	deltaV[i] = drag_coefficent * drag_acc * dt;

}

/*
* Real water density at a temperature(approximant). 
*
* @param[in] rho0: rest density
* @param[in] temprature: fluid temperature
* @param[in] temprature0: reference temperature (0 centigrade for water)
*/
static inline __device__ float densityOfFluidsAtTemperature(float rho0, float temperature, float temperature0) {
	float dT = temperature - temperature0;
	float coefficient = -0.0006654094 + 0.00006041581 * dT + 0.000003846269 * dT * dT;
	
	return rho0 / (1 + coefficient * (temperature - temperature0));
}
/*
*  deterative water density Curvature at a temperature(approximant).
*
* @param[in] rho0: rest density
* @param[in] temprature: fluid temperature
* @param[in] temprature0: reference temperature (0 centigrade for water)
*/
static inline __device__ float deterativeFluidsDensityCurvature(const float rho0, const float temperature, const float temperature0) {
	float dT = temperature - temperature0;
	float coefficient = 0.00006041581 * dT + 0.000007692538 * dT;

	return coefficient;
}

/*
*  Dew point at a temperature and humidity(approximant).
*
* @param[in] temperature: environment temperature  
* @param[in] humidity: environment humidity
*/
static inline __device__ float calDewPoint(const float temperature, const float humidity) {
	float dew_point = 0.0f;
	clamp(temperature, 0.0f, 65.0f);
	clamp(humidity, 1.0f, 100.0f);
	float b = 17.67;
	float c = 243.5;
	float vapor_pressure = log(humidity) + (17.67 * temperature / (243.5 + temperature));
	dew_point = 243.5 * vapor_pressure / (17.67 - vapor_pressure);
	return dew_point;
}

static inline __device__ float calWaterBoilingPoint(const float air_temperature, const float air_pressure_magnitude, const float basic_point) {

}
#endif