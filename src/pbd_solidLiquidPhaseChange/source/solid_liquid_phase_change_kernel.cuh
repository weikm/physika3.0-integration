/**
 * @file       : solid_liquid_phase_change_kernel.cuh
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @description: This file contains the declaration of CUDA functions for solid-liquid phase change simulation. The main functionality
 *               of this file revolves around the GPU accelerated physics simulation of solid-liquid phase change phenomena. It includes
 *               definitions for setting up and managing simulation parameters, performing calculations related to the phase change
 *               process, updating particle states, handling phase change interactions, and other related operations.
 *
 *               This file interfaces with the 'pbd_solidLiquidPhaseChange' module and specifically uses the solid_liquid_phase_change_params.hpp for
 *               the parameters required in the simulation. The CUDA kernels defined in this file are expected to be
 *               called by the host-side (CPU) code.
 *
 * @dependencies: This file depends on CUDA runtime headers and the solid_liquid_phase_change_params.hpp from the 'pbd_solidLiquidPhaseChange' module.
 *
 * @version    : 1.0
 */
#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>
#include <math_constants.h>

#include "solid_liquid_phase_change_params.hpp"
#include "helper_math.hpp"

using namespace cooperative_groups;
namespace Physika {

// forward declaration
__constant__ SolidLiquidPhaseChangeParams g_params;

/**
 * @brief setup :the SolidLiquidPhaseChange system parameters
 * @param[in] params :the SolidLiquidPhaseChange system parameters
 */
__host__ void setSimulationParams(SolidLiquidPhaseChangeParams* hostParam)
{
    cudaMemcpyToSymbol(g_params, hostParam, sizeof(SolidLiquidPhaseChangeParams));
}

/**
 * @brief      This CUDA device function calculates the inverse square root of a given number using the fast inverse square root method.
 *
 * @param[in]  number  The input number for which the inverse square root is to be calculated
 *
 * @return     The inverse square root of the input number
 */
__device__
inline float Q_rsqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5f;

	x2 = number * 0.5f;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);	
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));
	return y;
}

/**
 * @brief calculate the poly6 kernel value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel value
 */
__device__ float wPoly6(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > g_params .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = g_params .m_sph_radius_squared - lengthSquared;
    return g_params .m_poly6_coff * iterm * iterm * iterm;
}

/**
 * @brief calculate the poly6 kernel gradient value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel gradient value
 */
__device__
    float3
    wSpikyGrad(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3      ret           = { 0.0f, 0.0f, 0.0f };
    if (lengthSquared > g_params .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float       iterm  = g_params .m_sph_radius - length;
    float       coff   = g_params .m_spiky_grad_coff * iterm * iterm / length;
    ret.x              = coff * r.x;
    ret.y              = coff * r.y;
    ret.z              = coff * r.z;
    return ret;
}

/**
 * @brief calculate which grid the particle belongs to
 * @param[in] p the position of the particle
 * @return the grid position
 */
__device__
    int3
    calcGridPosKernel(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - g_params .m_world_origin.x) / g_params .m_cell_size.x);
    gridPos.y = floor((p.y - g_params .m_world_origin.y) / g_params .m_cell_size.y);
    gridPos.z = floor((p.z - g_params .m_world_origin.z) / g_params .m_cell_size.z);
    return gridPos;
}

/**
 * @brief      This CUDA device function calculates the gradient of the spiky kernel function for a given position vector.
 * @param[in]  r  The position vector for which the spiky gradient is to be calculated
 * @return     The gradient of the spiky kernel function at the given position
 */
__device__
float SpikyGrad(const float3& r)
{
	const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
	if (lengthSquared > g_params.m_sph_radius_squared || lengthSquared <= 0.00000001f)
		return 0.0;
	const float length = 1.f / Q_rsqrt(lengthSquared);
	float iterm = g_params.m_sph_radius - length;
	float coff = g_params.m_spiky_grad_coff * iterm * iterm / length;
	return coff;
}

/**
 * @brief calculate hash of grid positions (clamp them within cell boundary)
 * @param[in] gridPos the grid position
 * @return the hash value of given grid
 */
__device__ unsigned int calcGridHashKernel(int3 gridPos)
{
    gridPos.x = gridPos.x & (g_params .m_grid_size.x - 1);
    gridPos.y = gridPos.y & (g_params .m_grid_size.y - 1);
    gridPos.z = gridPos.z & (g_params .m_grid_size.z - 1);
    return gridPos.z * g_params .m_grid_size.x * g_params .m_grid_size.y + gridPos.y * g_params .m_grid_size.x + gridPos.x;
}

/**
 * @brief calculate the hash value of each particle
 *
 * @param[out] gridParticleHash :the hash value of each particle
 * @param[in] pos :the position of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void calcParticlesHashKernel(
    unsigned int* gridParticleHash,
    float4*       pos,
    unsigned int  numParticles)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    volatile float4 curPos    = pos[index];
    int3            gridPos   = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z));
    unsigned int    hashValue = calcGridHashKernel(gridPos);
    gridParticleHash[index]   = hashValue;
}

/**
 * @brief sort the particles based on their hash value
 *
 * @param[out] gridParticleHash the hash value of each particle
 * @param[out] gridParticleIndex the index of each particle
 * @param[in] numParticles the number of particles
 */
__global__ void findCellRangeKernel(
    unsigned int* cellStart,         // output: cell start index
    unsigned int* cellEnd,           // output: cell end index
    unsigned int* gridParticleHash,  // input: sorted grid hashes
    unsigned int  numParticles)
{
    thread_block                   cta = this_thread_block();
    extern __shared__ unsigned int sharedHash[];
    unsigned int                   index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int                   hashValue;

    if (index < numParticles)
    {
        hashValue                   = gridParticleHash[index];
        sharedHash[threadIdx.x + 1] = hashValue;

        // first thread in block must load neighbor particle hash
        if (index > 0 && threadIdx.x == 0)
            sharedHash[0] = gridParticleHash[index - 1];
    }

    sync(cta);

    if (index < numParticles)
    {
        if (index == 0 || hashValue != sharedHash[threadIdx.x])
        {
            cellStart[hashValue] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
            cellEnd[hashValue] = index + 1;
    }
}

/**
 * @brief      This CUDA global function performs advection of particles based on their velocity and external forces.
 *
 * @param      type           An array representing the type of each particle
 * @param      position       An array of 4D vectors representing the current position of each particle
 * @param      Temperature    An array representing the temperature of each particle
 * @param      velocity       An array of 4D vectors representing the velocity of each particle
 * @param      predictedPos   An array of 4D vectors to store the predicted position of each particle after advection
 * @param      deltaTime      The time step for advection
 * @param      numParticles   The total number of particles
 */
__global__
void advect(
	int * type,
	float4 *position,
	float* Temperature,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles,
	float3* external_force)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
				
		float3 new_gravity = make_float3(0.0f, (10.8f / log((100.0f + 0.02f) / 0.02f) * log((100.0f + 0.02f) / (100.0f + 0.02f - float(Temperature[index]))) - 9.8f), 0.0f);
		float3 newVel = make_float3(velocity[index]);
		float3 newPos = make_float3(position[index]);
		if(g_params.m_is_convect)
			newVel += deltaTime * new_gravity;
		else
			newVel += deltaTime * g_params.m_gravity;

		newVel += deltaTime * external_force[index];
		
		newPos += deltaTime * newVel;

		// collision with walls.
		if (newPos.x < g_params.m_world_box_corner1.x + g_params.m_particle_radius)
			newPos.x = g_params.m_world_box_corner1.x + g_params.m_particle_radius;
		if (newPos.x > g_params.m_world_box_corner2.x - g_params.m_particle_radius)
			newPos.x = g_params.m_world_box_corner2.x - g_params.m_particle_radius;

		if (newPos.y < g_params.m_world_box_corner1.y + g_params.m_particle_radius)
			newPos.y = g_params.m_world_box_corner1.y + g_params.m_particle_radius;
		if (newPos.y > g_params.m_world_box_corner2.y - g_params.m_particle_radius)
			newPos.y = g_params.m_world_box_corner2.y - g_params.m_particle_radius;
		
		if (newPos.z < g_params.m_world_box_corner1.z + g_params.m_particle_radius)
			newPos.z = g_params.m_world_box_corner1.z + g_params.m_particle_radius;
		if (newPos.z > g_params.m_world_box_corner2.z - g_params.m_particle_radius)
			newPos.z = g_params.m_world_box_corner2.z - g_params.m_particle_radius;
	

		if(type[index]==0)//boundary particle
		{
			predictedPos[index] = make_float4(make_float3(position[index]), 1.0f);
		}
		else
			predictedPos[index] = make_float4(newPos, 1.0f);
	
}

/**
 * @brief      This CUDA global function calculates the Lagrange multiplier for each particle based on its predicted position, velocity, type, and neighboring particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      predictedPos       An array of 4D vectors representing the predicted position of each particle after advection
 * @param      velocity           An array of 4D vectors representing the velocity of each particle
 * @param      Temperature        An array representing the temperature of each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 * @param      numCells           The total number of cells in the grid
 */
__global__
void calcLagrangeMultiplier(
	int * type,
	float4 *predictedPos,
	float4 *velocity,
	float* Temperature,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	// calculate current particle's density and lagrange multiplier.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	
	float3 readVel = make_float3(velocity[index]);
	float3 curPos = make_float3(predictedPos[index]);
	int3 gridPos = calcGridPosKernel(curPos);

	float beta;
	if(g_params.m_is_convect)
		beta = 1.5f - 1.0f / (1.0f + pow(1.0f + 12.0f/100.0f, 100.0f/2.0f - Temperature[index]));
	else
		beta=1.0f;
	float new_inv_rest_density=1.0f/(g_params.m_rest_density*beta);

	float density = 0.0f;
	float gradSquaredSum_j = 0.0f;
	float gradSquaredSumTotal = 0.0f;
	float3 curGrad, gradSum_i = { 0.0f,0.0f,0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				// empty cell.
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = predictedPos[i];
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					density += wPoly6(r);
					curGrad = wSpikyGrad(r);
					curGrad.x *= new_inv_rest_density;
					curGrad.y *= new_inv_rest_density;
					curGrad.z *= new_inv_rest_density;

					gradSum_i.x += curGrad.x;
					gradSum_i.y += curGrad.y;
					gradSum_i.y += curGrad.y;
					if (i != index)
						gradSquaredSum_j += curGrad.x * curGrad.x + curGrad.y * curGrad.y + curGrad.z * curGrad.z;
				}
			}
		}
	}
	gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;
	
	// density constraint.
	predictedPos[index].w = density;
	float constraint = density * new_inv_rest_density - 1.0f;
	float lambda = -(constraint) / (gradSquaredSumTotal + g_params.m_lambda_eps);
	velocity[index] = {readVel.x, readVel.y, readVel.z, lambda};
	
}

/**
 * @brief      This CUDA global function calculates the change in position (delta position) for each particle based on its predicted position, velocity, type, and neighboring particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      predictedPos       An array of 4D vectors representing the predicted position of each particle after advection
 * @param      velocity           An array of 4D vectors representing the velocity of each particle
 * @param      Temperature        An array representing the temperature of each particle
 * @param      deltaPos           An array of 3D vectors to store the change in position (delta position) for each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 */
__global__
void calcDeltaPosition(
	int * type,
	float4 *predictedPos,
	float4 *velocity,
	float* Temperature,
	float3 *deltaPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;


	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);

	float beta;
	if(g_params.m_is_convect)
		beta = 1.5f - 1.0f / (1.0f + pow(1.0f + 12.0f/100.0f, 100.0f/2.0f - Temperature[index]));
	else
		beta=1.0f;
	float new_inv_rest_density=1.0f/(g_params.m_rest_density*beta);

	float curLambda = readVel.w;
	float3 deltaP = { 0.0f, 0.0f, 0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = predictedPos[i];
					float neighbourLambda = velocity[i].w;
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					float corrTerm = wPoly6(r) * g_params.m_one_div_wPoly6;
					float coff = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
					float3 grad = wSpikyGrad(r);
					deltaP += coff * grad;
					
				}
			}
		}
	}

	
	float3 ret = {deltaP.x * new_inv_rest_density, deltaP.y * new_inv_rest_density,
		deltaP.z * new_inv_rest_density };	
	deltaPos[index] = ret;
	
}

/**
 * @brief      This CUDA global function calculates the change in temperature (delta temperature) for each particle based on its current position, type, and neighboring particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      Position           An array of 4D vectors representing the current position of each particle
 * @param      Temperature        An array representing the temperature of each particle
 * @param      deltaTem           An array to store the change in temperature (delta temperature) for each particle
 * @param      latentTem          An array representing the latent heat of each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 */
__global__
void calcDeltaTem(
	int* type,
	float4* Position,
	float* Temperature,
	float* deltaTem,
	float* latentTem,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	float4 readPos = Position[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);

	float deltaT = 0.0;

#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = Position[i];
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					deltaT += (Temperature[index] - Temperature[i]) * SpikyGrad(r) * 0.02;

				}
				if(g_params.m_radiate)
				{
					int suface_area=endIndex-startIndex;

					//if(suface_area<=1 && type[index]==1)
					//type[index]=2;

					if(suface_area>50)
						suface_area=0;
					else
						suface_area=50-suface_area;

					deltaT += -(Temperature[index]-80.0f) * suface_area * 0.000006;
				}
			}
		}
	}

	if (Temperature[index] < 20 && Temperature[index] + deltaT >= 20)
		latentTem[index] += deltaT;
	else if (Temperature[index] > 20 && Temperature[index] + deltaT <= 20)
		latentTem[index] -= deltaT;
	else {
		latentTem[index] = 0.0f;
	}


	if (latentTem[index] ==0.0f || latentTem[index]>4)
	{
		deltaTem[index] = deltaT;
		latentTem[index] = 0.0f;
	}
	else
	{
		deltaTem[index] = 0.0f;
	}
	
}

/**
 * @brief      This CUDA global function manages the phase change for each particle based on its type, initial position, current position, temperature, and neighboring particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      initId             An array representing the initial ID of each particle
 * @param      initId2Rest        An array representing the mapping from initial ID to the index in the sorted particle arrays
 * @param      initPos            An array of 4D vectors representing the initial position of each particle
 * @param      Position           An array of 4D vectors representing the current position of each particle
 * @param      Temperature        An array representing the temperature of each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 */
__global__
void managePhaseChange(
	int* type,
	int* initId,
	int* initId2Rest,
	float4* initPos,
	float4* Position,
	float* Temperature,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	float4 readPos = Position[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };	
	int3 gridPos = calcGridPosKernel(curPos);
	
	//solid change into fluid
	if(type[index]==1)//this particle is solid
	{
		if(Temperature[index]>g_params.m_melt_tem)//melt
		{
			type[index]=2;
		}

	}
	else//fluid change into solid
	{
		if(Temperature[index]<g_params.m_solidify_tem)//solidify
		{
#pragma unroll 3
			for (int z = -1; z <= 1; ++z)
			{
#pragma unroll 3
				for (int y = -1; y <= 1; ++y)
				{
#pragma unroll 3
					for (int x = -1; x <= 1; ++x)
					{
						int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
						unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
						unsigned int startIndex = cellStart[neighbourGridIndex];
						if (startIndex == 0xffffffff)
							continue;
						unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
						for (unsigned int i = startIndex; i < endIndex; ++i)
						{							
								float4 readNeighbour = Position[i];
								float3 neighbour = { readNeighbour.x, readNeighbour.y, readNeighbour.z };

								float3 r = curPos - neighbour;
								float r_len = sqrt(dot(r, r));

								if (type[i] == 1 && r_len < 0.6 * 1.0)
								{
									type[index] = 1;
									float4 r4 = { r.x, r.y, r.z, 1.f };
									initPos[initId2Rest[initId[index]]] = initPos[initId2Rest[initId[i]]] + r4;
									
								}								
						
						}
					}
				}
			}		

		}		
	}	
}

/**
 * @brief      This CUDA global function adds the change in position (delta position) to the predicted position for each particle.
 *
 * @param      type           An array representing the type of each particle
 * @param      predictedPos   An array of 4D vectors representing the predicted position of each particle after advection
 * @param      deltaPos       An array of 3D vectors representing the change in position (delta position) for each particle
 * @param      numParticles   The total number of particles
 */
__global__
void addDeltaPosition(
	int * type,
	float4 *predictedPos,
	float3 *deltaPos,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	
	float3 readPos = make_float3(predictedPos[index]);
	//if(type[index]==0) { deltaPos[index] = { 0.0f, 0.0f, 0.0f };}
	readPos += deltaPos[index];

	predictedPos[index] = { readPos.x, readPos.y, readPos.z, 1.f };
	
}

/**
 * @brief      This CUDA global function updates the mapping from initial ID to the current ID for each particle.
 *
 * @param      InitId1       An array representing the first part of the initial ID of each particle
 * @param      InitId2       An array representing the second part of the initial ID of each particle
 * @param      InitId2Now    An array to store the updated mapping from initial ID to the current ID for each particle
 * @param      initId2Rest   An array representing the mapping from initial ID to the index in the sorted particle arrays
 * @param      numParticles  The total number of particles
 */
__global__
void setInitId2Now(
	int* InitId1,
	int* InitId2,
	int* InitId2Now,
	int* initId2Rest,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	
	InitId2Now[InitId2[index]] = index;
	initId2Rest[InitId1[index]] = index;



}

/**
 * @brief      This CUDA global function adds the change in temperature (delta temperature) to the current temperature for each particle.
 *
 * @param      type         An array representing the type of each particle
 * @param      Temperature  An array representing the temperature of each particle
 * @param      deltaTem     An array representing the change in temperature (delta temperature) for each particle
 * @param      numParticles The total number of particles
 */
__global__
void addDeltaTem(
	int * type,
	float* Temperature,
	float* deltaTem,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles|| type[index]==0)//boundary particle
		return;
	Temperature[index]+= deltaTem[index];
	if(Temperature[index]>99)
		Temperature[index]=99;
	else if(Temperature[index]<-100)
		Temperature[index]=-100;	
}

/**
 * @brief      This CUDA global function applies distance constraints to the particles based on their type, initial ID, current ID, predicted position, initial position, change in position, and neighboring particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      initId             An array representing the initial ID of each particle
 * @param      initId2Now         An array representing the current ID mapped from the initial ID for each particle
 * @param      predictedPos       An array of 4D vectors representing the predicted position of each particle after advection
 * @param      initPos            An array of 4D vectors representing the initial position of each particle
 * @param      deltaPos           An array of 3D vectors representing the change in position (delta position) for each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 */
__global__
void distanceConstran(
	int* type,
	int* initId,
	int* initId2Now,
	float4* predictedPos,
	float4* initPos,
	float3* deltaPos,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	if (type[initId2Now[initId[index]]] != 1)
		return;

	float4 readPos = initPos[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);	

	float4 readNowPos = predictedPos[initId2Now[initId[index]]];
	float3 nowPos = { readNowPos.x, readNowPos.y, readNowPos.z };
	float invMass1 = readNowPos.w;

	float3 deltaP = make_float3(0.f);

	int num_neighbor = 0;

#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					if (type[initId2Now[initId[i]]] != 1)
						continue;
					float4 readNeighbour = predictedPos[initId2Now[initId[i]]];
					float3 neighbour = { readNeighbour.x, readNeighbour.y, readNeighbour.z };

					float4 readNeighbourInit = initPos[i];
					float3 neighbour_init = { readNeighbourInit.x, readNeighbourInit.y, readNeighbourInit.z };

					float invMass2 = readNeighbour.w;
					float wSum = invMass1 + invMass2;
					float3 r = nowPos - neighbour;
					float r_len = sqrt(dot(r, r));
					float3 grad = r / r_len;

					//count the init distance
					float3 rest_r = curPos - neighbour_init;
					float rest_len = sqrt(dot(rest_r, rest_r));

					float3 corr = make_float3(0.0f);
					

					corr = grad * (rest_len - r_len) / wSum;
					num_neighbor++;
					if (sqrt(corr.x * corr.x + corr.y * corr.y + corr.z * corr.z) > 0.6 * 0.01)
					{
						deltaP += invMass1 * corr;
						
					}

				}
			}
		}
	}
	deltaPos[initId2Now[initId[index]]] =3.5*deltaP/num_neighbor;
}

/**
 * @brief      This CUDA global function updates the velocity and position of each particle based on their type, current position, predicted position, inverse delta time, and total number of particles.
 *
 * @param      type           An array representing the type of each particle
 * @param      position       An array of 4D vectors representing the current position of each particle
 * @param      velocity       An array of 4D vectors representing the current velocity of each particle
 * @param      predictedPos   An array of 4D vectors representing the predicted position of each particle after advection
 * @param      invDeltaTime   The inverse of the time step size
 * @param      numParticles   The total number of particles
 */
__global__
void updateVelocityAndPosition(
	int * type,
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float invDeltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles || type[index]==0)//boundary particle
		return;

	float4 oldPos = position[index];
	float4 newPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
	posDiff *= invDeltaTime;
	velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
	position[index] = { newPos.x, newPos.y, newPos.z, newPos.w };
	
}

/**
 * @brief      This CUDA global function applies the eXtended Smoothed Particle Hydrodynamics (XSPH) method to update the velocity of each particle based on their type, position, neighboring particles, and total number of particles.
 *
 * @param      type               An array representing the type of each particle
 * @param      position           An array of 4D vectors representing the current position of each particle
 * @param      velocity           An array of 4D vectors representing the current velocity of each particle
 * @param      cellStart          An array representing the start index of each cell in the grid
 * @param      cellEnd            An array representing the end index of each cell in the grid
 * @param      gridParticleHash   An array representing the hash value of each particle's grid cell
 * @param      numParticles       The total number of particles
 */
__global__
void applyXSPH(
	int * type,
	float4* position,
	float4* velocity,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;


	
	float3 pos = make_float3(position[index]);
	float3 vel = make_float3(velocity[index]);
	int3 gridPos = calcGridPosKernel(pos);

	float3 avel = make_float3(0.f);
	float3 nVel = make_float3(0.f);
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float3 neighbour = make_float3(position[i]);
					float3 neighbourVel = make_float3(velocity[i]);
					avel += (neighbourVel - vel) * wPoly6(pos - neighbour);
					if (i == index) continue;
					float3 x_ij = { pos.x - neighbour.x, pos.y - neighbour.y, pos.z - neighbour.z };
					float len = length(x_ij);
					x_ij /= len;
					if (len <= 0.7f * g_params.m_sph_radius)
						nVel += 1.f/60  * g_params.m_rest_density * 1 * 1 * x_ij * cos(len * 3.36f / g_params.m_sph_radius);
				}
			}
		}
	}
	vel += nVel;
	velocity[index] = make_float4(vel, 0.f);

	
}
}  // namespace Physika