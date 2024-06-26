#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream> 

#include "cutil.h"
#include "cutil_math.h"

#include "debug.h"


#include "kernel.cuh"
#include "kernel_host.cuh"
#include "cudaHeaders.cuh"
#include "common_defs.h"

#include "SVD3.cuh"

#define SCAN_BLOCKSIZE			512
#define LUT_SIZE_CUDA	100000

ParticleParams fcudaParams;



__constant__ ParticleParams		simData;
__constant__ bufList			fbuf;			// GPU Particle buffers (unsorted)
__constant__ bufList			ftemp;			// GPU Particle buffers (sorted)
__constant__ uint				gridActive;


// * * * * * CUDA Setting * * * * * //

void cudaInit(int argc, char** argv)
{
	CUT_DEVICE_INIT(argc, argv);

	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);

	// 输出CUDA设备信息
	printf("-- CUDA Device Info --\n");
	printf("Name:       %s\n", p.name);
	printf("Capability: %d.%d\n", p.major, p.minor);
	printf("Global Mem: %d MB\n", p.totalGlobalMem / 1000000);
	printf("Shared/Blk: %d\n", p.sharedMemPerBlock);
	printf("Regs/Blk:   %d\n", p.regsPerBlock);
	printf("Warp Size:  %d\n", p.warpSize);
	printf("Mem Pitch:  %d\n", p.memPitch);
	printf("Thrds/Blk:  %d\n", p.maxThreadsPerBlock);
	printf("Const Mem:  %d\n", p.totalConstMem);
	printf("Clock Rate: %d\n", p.clockRate);
	std::cout << std::endl;
};

void cudaExit(int argc, char** argv)
{
	CUT_EXIT(argc, argv);
}

inline int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void AllocateThreads(int numParticles, int blockSize, int& numBlocks, int& numThreads)
{
	numThreads = min(blockSize, numParticles);
	numBlocks = iDivUp(numParticles, numThreads);
}

// ================================ //


// * * * * * DataTransfer * * * * * //

void TransferDataToCUDA(float* position, float* velocity, float* acceleration, float* pressure, float* density, int* type, int* explosion, int* lock, bool* active, bool* render, bool* rotate, float* center, float* smooth_radius, int* grid_ver_idx, float* concentration, bool* mix,float* eforce)
{
	const int number = fcudaParams.num_particle;

	CUDA_SAFE_CALL(cudaMemcpy(fbuf._position, position, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(fbuf._external_force, eforce, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._velocity, velocity, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._acceleration, acceleration, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._pressure, pressure, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._density, density, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._type, type, number * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._explosion, explosion, number * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._lock, lock, number * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._active, active, number * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._render, render, number * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._rotate, rotate, number * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._center, center, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._smooth_radius, smooth_radius, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._concentration, concentration, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix, mix, number * sizeof(bool), cudaMemcpyHostToDevice));
		const int mc_grid_ver_num = fcudaParams.grid_ver_number;
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.grid_ver_idx, grid_ver_idx, mc_grid_ver_num * sizeof(int), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
}

void TransferDataFromCUDA(float* position, float* velocity, float* acceleration, float* pressure, float* density, int* type, bool* render, float* particle_radius, float* concentration, float* scalar_field, float* color,
	float* grid_scalar_field, float* grid_color_field)
{
	const int number = fcudaParams.num_particle;
	
	if (position != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(position, fbuf._position, number * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	if (velocity != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(velocity, fbuf._velocity, number * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	if (acceleration != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(acceleration, fbuf._acceleration, number * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	if (pressure != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(pressure, fbuf._pressure, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (density != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(density, fbuf._density, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (type != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(type, fbuf._type, number * sizeof(int), cudaMemcpyDeviceToHost));
	if (render != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(render, fbuf._render, number * sizeof(bool), cudaMemcpyDeviceToHost));
	if (particle_radius != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(particle_radius, fbuf._particle_radius, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (concentration != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(concentration, fbuf._concentration, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (scalar_field != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(scalar_field, fbuf._surface_scalar_field, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (color != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(color, fbuf._color, number * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	
	const int mc_grid_ver_num = fcudaParams.grid_ver_number;
	if (grid_scalar_field != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(grid_scalar_field, fbuf.scalar_field_value_grid, mc_grid_ver_num * sizeof(float), cudaMemcpyDeviceToHost));
	if (grid_color_field != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(grid_color_field, fbuf.color_field_grid, mc_grid_ver_num * sizeof(float) * 3, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
}

void TransferSumDataFromCUDA(int& number, float* result)
{
	if (number < fcudaParams.particle_blocks)
	{
		number = 0;
		printf("sum error: number is not enough");
		return;
	}
	number = fcudaParams.particle_blocks;
	if (result != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(result, fbuf.sum_result, fcudaParams.particle_blocks * sizeof(float), cudaMemcpyDeviceToHost));
}

void TransferGridDataFromCUDA(uint* grid_particle_cellindex, uint* num_particle_grid, uint* grid_particle_table, uint* next_particle_index)
{
	int number = fcudaParams.num_particle;
	if (grid_particle_cellindex != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(grid_particle_cellindex, fbuf.particle_grid_cellindex, number * sizeof(uint), cudaMemcpyDeviceToHost));
	if (next_particle_index != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(next_particle_index, fbuf.next_particle_index, number * sizeof(uint), cudaMemcpyDeviceToHost));

	int grid_number = fcudaParams.grid_number;
	if (num_particle_grid != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(num_particle_grid, fbuf.num_particle_grid, grid_number * sizeof(uint), cudaMemcpyDeviceToHost));
	if (grid_particle_table != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(grid_particle_table, fbuf.grid_particle_table, grid_number * sizeof(uint), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
}

// bubble data transfer //
//void TransferBubbleDataFromCUDA(float* position, float* particle_radius, int* type)
//{
//	const int number = NumBubble;
//	const int dst = simData.max_num_particle - number;
//
//	if (position != 0x0)
//		CUDA_SAFE_CALL(cudaMemcpy(position + dst, fbuf._position + dst, number * sizeof(float) * 3, cudaMemcpyDeviceToHost));
//	if (type != 0x0)
//		CUDA_SAFE_CALL(cudaMemcpy(type + dst, fbuf._type + dst, number * sizeof(int), cudaMemcpyDeviceToHost));
//	if (particle_radius != 0x0)
//		CUDA_SAFE_CALL(cudaMemcpy(particle_radius + dst, fbuf._particle_radius + dst, number * sizeof(float), cudaMemcpyDeviceToHost));
//
//	cudaDeviceSynchronize();
//}

// mf data transfer //

void TransferMFDataToCUDA(float* mforce, float* restmass, float* mrest_density, float* mpressure, float* alpha_advanced, int* phase, float* mix_velocity, float* rest_mass_k)
{
	const int number = fcudaParams.num_particle;
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._force, mforce, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._rest_mass, restmass, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix_rest_density, mrest_density, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix_pressure, mpressure, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._alpha_advanced, alpha_advanced, MAX_PHASE_NUMBER * number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._phase, phase, number * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix_velocity, mix_velocity, number * sizeof(float) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._rest_mass_k, rest_mass_k, number * sizeof(float) * 3, cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
}

void TransferMFDataFromCUDA(float* alpha_advanced, float* restmass, float* eff_V, float* _delta_mass, float* _delta_mass_k, float* _delta_alpha, float* _mix_density, float* _mix_velocity)
{
	int number = fcudaParams.num_particle;
	if (alpha_advanced != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(alpha_advanced, fbuf._alpha_advanced, MAX_PHASE_NUMBER * number * sizeof(float), cudaMemcpyDeviceToHost));
	if (restmass != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(restmass, fbuf._rest_mass, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (eff_V != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(eff_V, fbuf._eff_V, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (_mix_density != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(_mix_density, fbuf._mix_density, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (_mix_velocity != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(_mix_velocity, fbuf._mix_velocity, 3 * number * sizeof(float), cudaMemcpyDeviceToHost));
	/*
	if (_delta_mass != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(_delta_mass, fbuf._delta_mass, number * sizeof(float), cudaMemcpyDeviceToHost));
	if (_delta_mass_k != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(_delta_mass_k, fbuf._delta_mass_k, MAX_PHASE_NUMBER * number * sizeof(float), cudaMemcpyDeviceToHost));
	if (_delta_alpha != 0x0)
		CUDA_SAFE_CALL(cudaMemcpy(_delta_alpha, fbuf._delta_alpha, MAX_PHASE_NUMBER * number * sizeof(float), cudaMemcpyDeviceToHost));*/

	cudaDeviceSynchronize();
}

// gui apply data transfer //

void TransferApplyDataToCUDA(bool* active, float* rest_mass, float*  mix_rest_density, int current_num, int* lock)
{
	const int number = fcudaParams.num_particle;

	CUDA_SAFE_CALL(cudaMemcpy(fbuf._active, active, current_num * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._rest_mass, rest_mass, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix_rest_density, mix_rest_density, number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._lock, lock, number * sizeof(int), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
}

void TransferLockDataToCUDA(int* lock)
{
	const int number = fcudaParams.num_particle;

	CUDA_SAFE_CALL(cudaMemcpy(fbuf._lock, lock, number * sizeof(int), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
}

void TransferVelocityDataToCUDA(float* mix_velocity, float* acceleration)
{
	const int number = fcudaParams.num_particle;

	CUDA_SAFE_CALL(cudaMemcpy(fbuf._mix_velocity, mix_velocity, 3 * number * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._acceleration, acceleration, 3 * number * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
}

// ================================ //


// * * * * * Memory Management * * * * * //

void SetupParticlesCUDA(int numParticles, int3 grid_resolution, int grid_number, float kernel_self, int container_size, int3 mc_grid_resolution, int mc_grid_number, int max_num)
{
	fcudaParams.num_particle = numParticles;
	fcudaParams.grid_resolution = grid_resolution;
	fcudaParams.grid_number = grid_number;
	fcudaParams.kernel_self = kernel_self;
	fcudaParams.container_size = container_size;
	
	// change:
	fcudaParams.max_num_particle = max_num;
	fcudaParams.grid_scan_max = grid_resolution - make_int3(3, 3, 3);

	// hardcode
	fcudaParams.theta1 = 0.5f;
	fcudaParams.theta2 = 0.2f;

	int cell = 0;
	for (int y = -1; y < 2; y++)
		for (int z = -1; z < 2; z++)
			for (int x = -1; x < 2; x++)
				fcudaParams.grid_search_offset[cell++] = x + fcudaParams.grid_resolution.x * (y + fcudaParams.grid_resolution.y * z);

	fcudaParams.mc_grid_resolution = mc_grid_resolution;
	fcudaParams.mc_grid_ver_resolution = make_int3(mc_grid_resolution.x + 1, mc_grid_resolution.y + 1, mc_grid_resolution.z + 1);
	fcudaParams.grid_ver_number = (mc_grid_resolution.x + 1) * (mc_grid_resolution.y + 1) * (mc_grid_resolution.z + 1);
	fcudaParams.mc_grid_number = mc_grid_number;
	
	cell = 0;
	for (int y = -4; y < 5; y++)
		for (int z = -4; z < 5; z++)
			for (int x = -4; x < 5; x++)
				fcudaParams.mc_grid_search_offset[cell++] = x + fcudaParams.mc_grid_resolution.x * (y + fcudaParams.mc_grid_resolution.y * z);

	// change:
	int block_size = 512;
	AllocateThreads(fcudaParams.num_particle, block_size, fcudaParams.particle_blocks, fcudaParams.particle_threads);
	AllocateThreads(fcudaParams.grid_number, block_size, fcudaParams.grid_blocks, fcudaParams.grid_threads);
	AllocateThreads(fcudaParams.mc_grid_number, block_size, fcudaParams.mc_grid_blocks, fcudaParams.mc_grid_threads);
	AllocateThreads(fcudaParams.grid_ver_number, block_size, fcudaParams.grid_ver_blocks, fcudaParams.grid_ver_threads);

	fcudaParams.size_Points = fcudaParams.particle_blocks * fcudaParams.particle_threads;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._position, fcudaParams.size_Points * sizeof(float) * 3));
    CUDA_SAFE_CALL(cudaMalloc(( void** )&fbuf._external_force, fcudaParams.size_Points * sizeof(float) * 3));
    CUDA_SAFE_CALL(cudaMalloc(( void** )&fbuf._render_pos, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._velocity, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._acceleration, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._density, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._pressure, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._type, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._phase, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._explosion, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._lock, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._active, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._render, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._rotate, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._noted, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._test, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mix, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._vorticity, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._vorticity_force, fcudaParams.size_Points * sizeof(float) * 3));



	// temp container
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._container, fcudaParams.size_Points * sizeof(neighbor_value_container) * container_size));

	// peridynamics
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._tensor_K, fcudaParams.size_Points * sizeof(float) * 9));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._tensor_F, fcudaParams.size_Points * sizeof(float) * 9));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._SdV, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._dV, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._eff_V, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._smooth_radius, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._particle_radius, fcudaParams.size_Points * sizeof(float)));

	// muti-phase
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mix_velocity, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._phase_velocity, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._drift_velocity, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._force, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._pressure_force, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._visc_force, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._phase_force, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mix_density, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mix_pressure, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._alpha, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._delta_alpha, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._alpha_advanced, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._alpha_sum, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._viscosity, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._rest_mass, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mix_rest_density, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._lambda, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._delta_mass, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._delta_mass_k, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._rest_mass_k, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._color, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._concentration, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._sf_k, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._surface_tension, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._mass_fraction, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._nabla_c, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float) * 3));

	// bubbles
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.inter_alpha, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubble_volume, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubble, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubble_type, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.attached_id, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.idxToListIdx, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubbleList, MAX_BUBBLE_NUM * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubblePosList, MAX_BUBBLE_NUM * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.bubbleNum, sizeof(int)));

	CUDA_SAFE_CALL(cudaMemset(fbuf.bubbleList, UNDEF_INT, sizeof(int) * MAX_BUBBLE_NUM));
	CUDA_SAFE_CALL(cudaMemset(fbuf.bubbleNum, 0, sizeof(int)));

	// surface tracking
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._ave_position, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._weighted_position, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._surface_scalar_field, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._C, fcudaParams.size_Points * sizeof(float) * 9));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._G, fcudaParams.size_Points * sizeof(float) * 9));

	// rigid
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._center, fcudaParams.size_Points * sizeof(float) * 3));

	// pressure-bound
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._pre_velocity, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._acceleration_p, fcudaParams.size_Points * sizeof(float) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._rest_volume, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._volume, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._source_term, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._diag_term, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._residual, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._bound_phi, fcudaParams.size_Points * sizeof(float)));

	// data structure
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.next_particle_index, fcudaParams.size_Points * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_grid_cellindex, fcudaParams.size_Points * sizeof(uint)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_neighbor_number, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.neighbor_search_index, fcudaParams.size_Points * sizeof(uint)));

	fcudaParams.size_Grids = fcudaParams.grid_blocks * fcudaParams.grid_threads;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_particle_table, fcudaParams.size_Grids * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.num_particle_grid, fcudaParams.size_Grids * sizeof(uint)));

	// marching cube grid
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.mc_next_particle_index, fcudaParams.size_Points * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_mc_grid_cellindex, fcudaParams.size_Points * sizeof(uint)));

	fcudaParams.size_McGrids = fcudaParams.mc_grid_blocks * fcudaParams.mc_grid_threads;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.mc_grid_particle_table, fcudaParams.size_McGrids * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.num_particle_mc_grid, fcudaParams.size_McGrids * sizeof(uint)));

	fcudaParams.size_GridVers = fcudaParams.grid_ver_blocks * fcudaParams.grid_ver_threads;
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.scalar_field_value_grid, fcudaParams.size_GridVers * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.color_field_grid, fcudaParams.size_GridVers * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_ver_idx, fcudaParams.size_GridVers * sizeof(int)));

	// sum function
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.sum_result, fcudaParams.particle_blocks * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.min_data, fcudaParams.size_Points * MAX_PHASE_NUMBER * sizeof(float)));

	// change:------------
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf._inv_dens, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.particle_grid_index, fcudaParams.size_Points * sizeof(uint)));
	printf("max_num_particle: %d\n--------------", fcudaParams.max_num_particle);
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.sorted_grid_map, fcudaParams.max_num_particle * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_offset, fcudaParams.size_Grids * sizeof(uint)));

	// prefix sums auxiliary buffers
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = grid_number;
	int numElem2 = (int)(numElem1 / blockSize) + 1;
	int numElem3 = (int)(numElem2 / blockSize) + 1;
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.aux_array1, numElem2 * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.aux_scan1, numElem2 * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.aux_array2, numElem3 * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.aux_scan2, numElem3 * sizeof(uint)));

	// temp buffers
    printf("nnnn:%d\n", fcudaParams.size_Points);
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._position, fcudaParams.size_Points * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMalloc(( void** )&ftemp._external_force, fcudaParams.size_Points * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._mix_velocity, fcudaParams.size_Points * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._drift_velocity, fcudaParams.size_Points * sizeof(float3)* MAX_PHASE_NUMBER));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._force, fcudaParams.size_Points * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._alpha, fcudaParams.size_Points * sizeof(float)* MAX_PHASE_NUMBER));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._alpha_advanced, fcudaParams.size_Points * sizeof(float)* MAX_PHASE_NUMBER));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._delta_alpha, fcudaParams.size_Points * sizeof(float)* MAX_PHASE_NUMBER));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._delta_mass_k, fcudaParams.size_Points * sizeof(float)* MAX_PHASE_NUMBER));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._alpha_sum, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._lambda, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._mix_pressure, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._mix_density, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._inv_dens, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._mix_rest_density, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._viscosity, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._rest_mass, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._particle_radius, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._bound_phi, fcudaParams.size_Points * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._type, fcudaParams.size_Points * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._active, fcudaParams.size_Points * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp.particle_grid_cellindex, fcudaParams.size_Points * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp.particle_grid_index, fcudaParams.size_Points * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._tensor_K, fcudaParams.size_Points * sizeof(tensor)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ftemp._tensor_F, fcudaParams.size_Points * sizeof(tensor)));

	//-------------------------

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));

	CUDA_SAFE_CALL(cudaMemset(fbuf._test, false, sizeof(bool) * fcudaParams.size_Points));

	
	
	cudaDeviceSynchronize();
}

void SetParametersCUDA(float3 boundary_min, float3 boundary_max, float3 grid_boundary_offset, float3 gravity, float mass, float rest_density,
	float gas_constant, float viscosity, float damp, float simscale, float smooth_radius, float particle_radius, float grid_radius, bool _explicit, int test_index, float3 center,
	float smoothing_factor, float k_n, float k_r, float k_s, int neighbor_threshold, float surface_tension_factor, float mc_grid_radius)
{
	fcudaParams.boundary_min = boundary_min;
	fcudaParams.boundary_max = boundary_max;
	fcudaParams.grid_boundary_offset = grid_boundary_offset;
	fcudaParams.rotate_center = center;
	fcudaParams.gravity = gravity;

	fcudaParams.mass = mass;
	fcudaParams.rest_density = rest_density;
	fcudaParams.gas_constant = gas_constant;
	fcudaParams.viscosity = viscosity;
	fcudaParams.surface_tension_factor = surface_tension_factor;
	fcudaParams.damp = damp;
	fcudaParams.simscale = simscale;
	fcudaParams.smooth_radius = smooth_radius;
	fcudaParams.particle_radius = particle_radius;
	fcudaParams.grid_radius = grid_radius;

	fcudaParams.mc_grid_radius = mc_grid_radius;

	fcudaParams._explicit = _explicit;

	fcudaParams.test_index = test_index;

	fcudaParams.smoothing_factor = smoothing_factor;
	fcudaParams.diag_k_n = k_n;
	fcudaParams.diag_k_r = k_r;
	fcudaParams.diag_k_s = k_s;
	fcudaParams.search_num_threshold = neighbor_threshold;

	fcudaParams.poly6kern = 315.0f / (64.0f * 3.141592 * pow(smooth_radius, 9));
	fcudaParams.spikykern = -45.0f / (3.141592 * pow(smooth_radius, 6));
	fcudaParams.lapkern = 45.0f / (3.141592 * pow(smooth_radius, 6));

	fcudaParams.CubicSplineKern = 8.0f / (3.141592 * pow(smooth_radius, 3));
	fcudaParams.GradCubicSplineKern = 48.0f / (3.141592 * pow(smooth_radius, 4));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));
	cudaDeviceSynchronize();
}

void SetupMarchingCubeCUDA(int3 mc_grid_resolution, int mc_grid_number, float mc_grid_radius)
{
	fcudaParams.mc_grid_resolution = mc_grid_resolution;
	fcudaParams.mc_grid_ver_resolution = make_int3(mc_grid_resolution.x + 1, mc_grid_resolution.y + 1, mc_grid_resolution.z + 1);
	fcudaParams.grid_ver_number = (mc_grid_resolution.x + 1) * (mc_grid_resolution.y + 1) * (mc_grid_resolution.z + 1);
	fcudaParams.mc_grid_number = mc_grid_number;
	fcudaParams.mc_grid_radius = mc_grid_radius;

	int cell = 0;
	for (int y = -4; y < 5; y++)
		for (int z = -4; z < 5; z++)
			for (int x = -4; x < 5; x++)
				fcudaParams.mc_grid_search_offset[cell++] = x + fcudaParams.mc_grid_resolution.x * (y + fcudaParams.mc_grid_resolution.y * z);


	CUDA_SAFE_CALL(cudaFree(fbuf.mc_grid_particle_table));
	CUDA_SAFE_CALL(cudaFree(fbuf.num_particle_mc_grid));

	CUDA_SAFE_CALL(cudaFree(fbuf.scalar_field_value_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.color_field_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.grid_ver_idx));

	AllocateThreads(fcudaParams.mc_grid_number, 256, fcudaParams.mc_grid_blocks, fcudaParams.mc_grid_threads);
	AllocateThreads(fcudaParams.grid_ver_number, 256, fcudaParams.grid_ver_blocks, fcudaParams.grid_ver_threads);

	fcudaParams.size_McGrids = fcudaParams.mc_grid_blocks * fcudaParams.mc_grid_threads;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.mc_grid_particle_table, fcudaParams.size_McGrids * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.num_particle_mc_grid, fcudaParams.size_McGrids * sizeof(uint)));

	fcudaParams.size_GridVers = fcudaParams.grid_ver_blocks * fcudaParams.grid_ver_threads;
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.scalar_field_value_grid, fcudaParams.size_GridVers * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.color_field_grid, fcudaParams.size_GridVers * sizeof(float3)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&fbuf.grid_ver_idx, fcudaParams.size_GridVers * sizeof(int)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));

	cudaDeviceSynchronize();
}

void SetGeneralParametersCUDA(float gas_constant)
{
	fcudaParams.gas_constant = gas_constant;
	printf("gas_constant: %f\n", fcudaParams.gas_constant);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));

	cudaDeviceSynchronize();
}

void SetMcParametersCUDA(float itp_radius, float anisotropic_radius, float k_n, float k_r, float k_s)
{
	fcudaParams.itp_radius = itp_radius;
	fcudaParams.anisotropic_radius = anisotropic_radius;
	fcudaParams.diag_k_n = k_n;
	fcudaParams.diag_k_r = k_r;
	fcudaParams.diag_k_s = k_s;
	printf("mc_grid_radius: %f, itp_radius: %f, anisotropic_radius: %f\n", fcudaParams.mc_grid_radius, fcudaParams.itp_radius, fcudaParams.anisotropic_radius);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));

	cudaDeviceSynchronize();
}

// mf memory setting //

void SetMFParametersCUDA(float* density, float* mass, float* visc, int phase_number, float tau, float sigma, bool miscible, float3* phase_color)
{
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		fcudaParams.phase_density[fcount] = *(density+fcount);
		fcudaParams.phase_mass[fcount] = *(mass+fcount);
		fcudaParams.phase_visc[fcount] = *(visc+fcount);
		fcudaParams.phase_color[fcount] = *(phase_color+fcount);
		printf("density: %f, mass: %f, visc: %f\n", *(density + fcount), *(mass + fcount), *(visc + fcount));
	}
	fcudaParams.phase_number = phase_number;
	fcudaParams.tau = tau;
	fcudaParams.sigma = sigma;
	fcudaParams.miscible = miscible;


	CUDA_SAFE_CALL(cudaMemcpyToSymbol(simData, &fcudaParams, sizeof(ParticleParams)));
	cudaDeviceSynchronize();
}

void ClearParticlesCUDA()
{
	CUDA_SAFE_CALL(cudaFree(fbuf._position));
	CUDA_SAFE_CALL(cudaFree(fbuf._velocity));
	CUDA_SAFE_CALL(cudaFree(fbuf._acceleration));
	CUDA_SAFE_CALL(cudaFree(fbuf._density));
	CUDA_SAFE_CALL(cudaFree(fbuf._pressure));
	CUDA_SAFE_CALL(cudaFree(fbuf.particle_grid_cellindex));
	CUDA_SAFE_CALL(cudaFree(fbuf.next_particle_index));
	CUDA_SAFE_CALL(cudaFree(fbuf.neighbor_search_index));
	CUDA_SAFE_CALL(cudaFree(fbuf.particle_neighbor_number));
	CUDA_SAFE_CALL(cudaFree(fbuf._type));
	CUDA_SAFE_CALL(cudaFree(fbuf._phase));
	CUDA_SAFE_CALL(cudaFree(fbuf._active));
	CUDA_SAFE_CALL(cudaFree(fbuf._render));
	CUDA_SAFE_CALL(cudaFree(fbuf._rotate));
	CUDA_SAFE_CALL(cudaFree(fbuf._noted));
	CUDA_SAFE_CALL(cudaFree(fbuf._test));
	CUDA_SAFE_CALL(cudaFree(fbuf._mix));
	CUDA_SAFE_CALL(cudaFree(fbuf._vorticity));
	CUDA_SAFE_CALL(cudaFree(fbuf._vorticity_force));

	// temp container
	CUDA_SAFE_CALL(cudaFree(fbuf._container));

	// peridynamics
	CUDA_SAFE_CALL(cudaFree(fbuf._tensor_K));
	CUDA_SAFE_CALL(cudaFree(fbuf._tensor_F));
	CUDA_SAFE_CALL(cudaFree(fbuf._SdV));
	CUDA_SAFE_CALL(cudaFree(fbuf._dV));
	CUDA_SAFE_CALL(cudaFree(fbuf._eff_V));
	CUDA_SAFE_CALL(cudaFree(fbuf._smooth_radius));
	CUDA_SAFE_CALL(cudaFree(fbuf._particle_radius));

	// muti-phase
	CUDA_SAFE_CALL(cudaFree(fbuf._mix_velocity));
	CUDA_SAFE_CALL(cudaFree(fbuf._phase_velocity));
	CUDA_SAFE_CALL(cudaFree(fbuf._drift_velocity));
	CUDA_SAFE_CALL(cudaFree(fbuf._force));
	CUDA_SAFE_CALL(cudaFree(fbuf._pressure_force));
	CUDA_SAFE_CALL(cudaFree(fbuf._visc_force));
	CUDA_SAFE_CALL(cudaFree(fbuf._phase_force));
	CUDA_SAFE_CALL(cudaFree(fbuf._mix_density));
	CUDA_SAFE_CALL(cudaFree(fbuf._mix_pressure));
	CUDA_SAFE_CALL(cudaFree(fbuf._alpha));
	CUDA_SAFE_CALL(cudaFree(fbuf._delta_alpha));
	CUDA_SAFE_CALL(cudaFree(fbuf._alpha_advanced));
	CUDA_SAFE_CALL(cudaFree(fbuf._alpha_sum));
	CUDA_SAFE_CALL(cudaFree(fbuf._viscosity));
	CUDA_SAFE_CALL(cudaFree(fbuf._rest_mass));
	CUDA_SAFE_CALL(cudaFree(fbuf._mix_rest_density));
	CUDA_SAFE_CALL(cudaFree(fbuf._lambda));
	CUDA_SAFE_CALL(cudaFree(fbuf._delta_mass));
	CUDA_SAFE_CALL(cudaFree(fbuf._delta_mass_k));
	CUDA_SAFE_CALL(cudaFree(fbuf._rest_mass_k));
	CUDA_SAFE_CALL(cudaFree(fbuf._color));
	CUDA_SAFE_CALL(cudaFree(fbuf._concentration));
	CUDA_SAFE_CALL(cudaFree(fbuf._sf_k));
	CUDA_SAFE_CALL(cudaFree(fbuf._surface_tension));
	CUDA_SAFE_CALL(cudaFree(fbuf._mass_fraction));
	CUDA_SAFE_CALL(cudaFree(fbuf._nabla_c));

	// bubbles
	CUDA_SAFE_CALL(cudaFree(fbuf.inter_alpha));
	CUDA_SAFE_CALL(cudaFree(fbuf.bubble_volume));
	CUDA_SAFE_CALL(cudaFree(fbuf.bubble));
	CUDA_SAFE_CALL(cudaFree(fbuf.bubble_type));
	CUDA_SAFE_CALL(cudaFree(fbuf.attached_id));

	// surface tracking
	CUDA_SAFE_CALL(cudaFree(fbuf._ave_position));
	CUDA_SAFE_CALL(cudaFree(fbuf._weighted_position));
	CUDA_SAFE_CALL(cudaFree(fbuf._surface_scalar_field));
	CUDA_SAFE_CALL(cudaFree(fbuf._C));
	CUDA_SAFE_CALL(cudaFree(fbuf._G));


	// pressure-bound
	CUDA_SAFE_CALL(cudaFree(fbuf._pre_velocity));
	CUDA_SAFE_CALL(cudaFree(fbuf._acceleration_p));
	CUDA_SAFE_CALL(cudaFree(fbuf._rest_volume));
	CUDA_SAFE_CALL(cudaFree(fbuf._volume));
	CUDA_SAFE_CALL(cudaFree(fbuf._source_term));
	CUDA_SAFE_CALL(cudaFree(fbuf._diag_term));
	CUDA_SAFE_CALL(cudaFree(fbuf._residual));
	CUDA_SAFE_CALL(cudaFree(fbuf._bound_phi));

	// data structure
	CUDA_SAFE_CALL(cudaFree(fbuf.grid_particle_table));
	CUDA_SAFE_CALL(cudaFree(fbuf.num_particle_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.grid_search_offset));

	//CUDA_SAFE_CALL(cudaFree(fbuf.neighbortable));
	//CUDA_SAFE_CALL(cudaFree(fbuf.neighbordist));

	// sum function
	CUDA_SAFE_CALL(cudaFree(fbuf.sum_result));
	CUDA_SAFE_CALL(cudaFree(fbuf.min_data));

	// mcube structure
	CUDA_SAFE_CALL(cudaFree(fbuf.mc_grid_particle_table));
	CUDA_SAFE_CALL(cudaFree(fbuf.num_particle_mc_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.mc_grid_search_offset));

	CUDA_SAFE_CALL(cudaFree(fbuf.mc_next_particle_index));
	CUDA_SAFE_CALL(cudaFree(fbuf.particle_mc_grid_cellindex));

	CUDA_SAFE_CALL(cudaFree(fbuf.scalar_field_value_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.color_field_grid));
	CUDA_SAFE_CALL(cudaFree(fbuf.grid_ver_idx));

	// change:
	CUDA_SAFE_CALL(cudaFree(fbuf._inv_dens));
	CUDA_SAFE_CALL(cudaFree(fbuf.sorted_grid_map));
	CUDA_SAFE_CALL(cudaFree(fbuf.grid_offset));

	CUDA_SAFE_CALL(cudaFree(fbuf.aux_array1));
	CUDA_SAFE_CALL(cudaFree(fbuf.aux_scan1));
	CUDA_SAFE_CALL(cudaFree(fbuf.aux_array2));
	CUDA_SAFE_CALL(cudaFree(fbuf.aux_scan2));

	CUDA_SAFE_CALL(cudaFree(ftemp._position));
	CUDA_SAFE_CALL(cudaFree(ftemp._mix_velocity));
	CUDA_SAFE_CALL(cudaFree(ftemp._inv_dens));
	CUDA_SAFE_CALL(cudaFree(ftemp._drift_velocity));
	CUDA_SAFE_CALL(cudaFree(ftemp._force));
	CUDA_SAFE_CALL(cudaFree(ftemp._alpha));
	CUDA_SAFE_CALL(cudaFree(ftemp._alpha_advanced));
	CUDA_SAFE_CALL(cudaFree(ftemp._delta_alpha));
	CUDA_SAFE_CALL(cudaFree(ftemp._delta_mass_k));
	CUDA_SAFE_CALL(cudaFree(ftemp._alpha_sum));
	CUDA_SAFE_CALL(cudaFree(ftemp._lambda));
	CUDA_SAFE_CALL(cudaFree(ftemp._mix_pressure));
	CUDA_SAFE_CALL(cudaFree(ftemp._mix_density));
	CUDA_SAFE_CALL(cudaFree(ftemp._mix_rest_density));
	CUDA_SAFE_CALL(cudaFree(ftemp._viscosity));
	CUDA_SAFE_CALL(cudaFree(ftemp._rest_mass));
	CUDA_SAFE_CALL(cudaFree(ftemp._particle_radius));
	CUDA_SAFE_CALL(cudaFree(ftemp._bound_phi));
	CUDA_SAFE_CALL(cudaFree(ftemp._type));
	CUDA_SAFE_CALL(cudaFree(ftemp._active));
	CUDA_SAFE_CALL(cudaFree(ftemp.particle_grid_cellindex));
	CUDA_SAFE_CALL(cudaFree(ftemp.particle_grid_index));
	CUDA_SAFE_CALL(cudaFree(ftemp._tensor_K));
	CUDA_SAFE_CALL(cudaFree(ftemp._tensor_F));
}

// ==================================== //


// * * * * * SubStep Function * * * * * //

void AllocateGridCUDA(int numParticles)
{
	CUDA_SAFE_CALL(cudaMemset(fbuf.grid_particle_table, GRID_UNDEF, sizeof(uint) * fcudaParams.grid_number));
	CUDA_SAFE_CALL(cudaMemset(fbuf.num_particle_grid, 0, sizeof(uint) * fcudaParams.grid_number));


	AllocateGrid << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: AllocateGrid: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// peridynamics computing function //

void ComputeTensorKCUDA(int numParticles)
{
	ComputeTensorK << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeTensorK: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeTensorFCUDA(float time_step, int numParticles)
{
	ComputeTensorF << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeTensorF: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// muti-phase sph //

void ComputeMFDensityCUDA(int numParticles)
{
	CUDA_SAFE_CALL(cudaMemset(fbuf.particle_neighbor_number, 0, sizeof(int) * fcudaParams.num_particle));
	ComputeMFDensity << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeMFDensity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
	/*
	ComputeMFBoundPressure << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeMFBoundPressure: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();*/
}

void ApplyAlphaCUDA(int numParticles)
{
	ApplyAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ApplyAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeDriftVelocityCUDA(int numParticles)
{
	ComputeDriftVelocity << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeDriftVelocity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeAlphaAdvanceCUDA(float time_step, int numParticles)
{
	ComputeAlphaAdvance << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeAlphaAdvance: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeCorrectionCUDA(int numParticles)
{
	ComputeCorrection << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeCorrection: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeCorrectionVolumeCUDA(float time_step, int numParticles)
{
	ComputeCorrectionVolume << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeCorrectionVolume: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void AlphaBanlanceUsingMassFractionCUDA(float time_step, int numParticles)
{
	AlphaBanlanceUsingMassFraction << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: AlphaBanlanceUsingMassFraction: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeTDMCUDA(int numParticles)
{
	ComputeTDM << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeTDM: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void UpdateMFParticlesCUDA(float time_step, int numParticles)
{
	UpdateMFParticles << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: UpdateMFParticles: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

//  muti-phase in peridynamics //

void ComputeDriftVelocityPeridynamicsCUDA(int numParticles, float bound_vel, float factor)
{
	ComputeDriftVelocityPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, bound_vel, factor);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeDriftVelocityPeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeSdVPeridynamicsCUDA(int numParticles)
{
	ComputeSdVPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeSdVPeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeAlphaAdvancePeridynamicsCUDA(float time_step, int numParticles)
{
	ComputeAlphaAdvancePeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeAlphaAdvancePeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeTDMPeridynamicsCUDA(float time_step, int numParticles, float surface_factor)
{
	ComputeTDMPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step, surface_factor);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeTDMPeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeMFItemPeridynamicsCUDA(float time_step, int numParticles)
{
	/*
	ComputeMFSourceTermPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeSourceTermPeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	ComputeMFDiagElementPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeDiagElementPeridynamics: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	InitialiseMFPressure << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: InitialisePressure: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();*/
}

void MFPressureSolvePeridynamicsCUDA(float time_step, int numParticles, int numFluids)
{
	int iter = 0;
	float v_error;
	float eta = 0.1;
	float last_error = 1.0f;
	cudaError_t error;
	do {
		iter++;
		v_error = 0.0f;

		ComputeMFPressureAccelPeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: ComputeMFPressureAccelPeridynamics: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		MFPressureUpdatePeridynamics << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: MFPressureUpdatePeridynamics: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		//if (abs(v_error - last_error) / last_error<0.001 || iter>MAX_LOOPS)
			//break;
	} while (iter < 10);
}

// alpha transport //

void ComputeAlphaTransportCUDA(float time_step, int numParticles, float factor)
{
	int max_iter = 20;
	int l = 1;
	while (l <= max_iter)
	{
		ComputeAlphaTransportIter << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step, factor);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: ComputeAlphaTransportIter: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();
		/**/
		UpdateLambda << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: UpdateLambda: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();
		l += 1;
	}
}

void ContributeAlphaCorrectionCUDA(int numParticles)
{
	AlphaCorrection << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ContributeAlphaCorrection: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// particle bound //

void ComputePhiParticleBoundCUDA(int numParticles)
{
	ComputePhiParticleBound << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputePhiParticleBound: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeMFDensityParticleBoundCUDA(int numParticles, bool transferAlpha)
{
	ComputeMFDensityParticleBound << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, transferAlpha);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeMFDensityParticleBound: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ContributePressureCUDA(int numParticles)
{
	ContributePressure << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ContributePressure: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeBoundPressureCUDA(int numParticles)
{
	ComputeBoundPressure << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeBoundPressure: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void AdvanceParticleBoundCUDA(float time_step, int numParticles)
{
	AdvanceParticleBound << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: AdvanceParticleBound: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// generate mf particles //

void GenerateParticlesCUDA(float time_step, int begin, int N, int numParticles, int Generate_pos, float3 start_point, float GenFrameRate)
{
	GenerateParticles << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, begin, N, time_step, Generate_pos, start_point, GenFrameRate);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: GenerateParticles: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ChemicalReactionCUDA(int numParticles, float ReactionSpeed)
{
	ChemicalReaction << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, ReactionSpeed);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ChemicalReaction: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ConcentrationDecayCUDA(int numParticles)
{
	ConcentrationDecay << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ConcentrationDecay: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// rigid body //

void UpdateRigidBodyCUDA(float time_step, int numParticles, float omega)
{
	UpdateRigidBody << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step, omega);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: UpdateRigidBody: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void UpdateRigidBodyDriftCUDA(float time_step, int numParticles)
{
	UpdateRigidBodyDrift << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: UpdateRigidBodyDrift: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void RigidBodyTransitionCUDA(int numParticles)
{
	RigidBodyTransition << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: RigidBodyTransition: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void UpdateUpBoundCUDA(float time_step, int numParticles)
{
	UpdateUpBound << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: UpdateUpBound: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// density scalar field //
void GetAverageKernelPosCUDA(int numParticles)
{
	GetAverageKernelPos << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: GetAverageKernelPos: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputePosCovarianceCUDA(int numParticles)
{
	ComputePosCovariance << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputePosCovariance: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeShapeMatrixGCUDA(int numParticles)
{
	//ComputeShapeMatG << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	ComputeShapeMatGSVD << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeShapeMatG: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void CalDensityScalarFieldParticleCUDA(int numParticles)
{
	CalDensityScalarFieldParticle << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: CalDensityScalarFieldParticle: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// mcube grid //
void AllocateMcGridCUDA(int numParticles)
{
	CUDA_SAFE_CALL(cudaMemset(fbuf.mc_grid_particle_table, GRID_UNDEF, sizeof(uint) * fcudaParams.mc_grid_number));
	CUDA_SAFE_CALL(cudaMemset(fbuf.num_particle_mc_grid, 0, sizeof(uint) * fcudaParams.mc_grid_number));


	AllocateMcGrid << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: AllocateMcGrid: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ParticleScalarvalueToGridCUDA(int numMcGridver)
{
	ParticleScalarvalueToGrid << <fcudaParams.mc_grid_blocks, fcudaParams.mc_grid_threads >> > (fbuf, numMcGridver);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ParticleScalarvalueToGrid: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// reset particle //
void ResetParticleAttribCUDA(int numParticles)
{
	ResetParticles << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ResetParticles: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// secondary bubbles //
void ComputeBubbleAlphaCUDA(int numParticles)
{
	ComputeBubbleAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeBubbleAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void InsertBubbleParticleCUDA(int numParticles)
{
	InsertBubbleParticle << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: InsertBubbleParticle: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void UpdatePosBubbleCUDA(float time_step, int numParticles)
{
	UpdatePosBubble << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: UpdatePosBubble: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void DeleteBubbleParticleCUDA(int numParticles)
{
	DeleteBubbleParticle << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: DeleteBubbleParticle: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ResortBubbleList(int* target, float3* target_pos, float3* position, int* idxToListIdx, int& num_bubble)
{
	const int numParticles = fcudaParams.num_particle;
	int count1 = 0, count2 = 0;

	num_bubble = -1;

	CUDA_SAFE_CALL(cudaMemcpy(&num_bubble, &fbuf.bubbleNum[0], sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(idxToListIdx, fbuf.idxToListIdx, numParticles * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(target, fbuf.bubbleList, MAX_BUBBLE_NUM * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(target_pos, fbuf.bubblePosList, 3 * MAX_BUBBLE_NUM * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(position, fbuf._position, 3 * numParticles * sizeof(float), cudaMemcpyDeviceToHost));
	printf("num_bubble:%d\n", num_bubble);
	// rerange bubbleList
	while (count1 < num_bubble)
	{
		if (target[count2] != UNDEF_INT)
		{
			if (count1 != count2)
			{
				target[count1] = target[count2];
				target[count2] = UNDEF_INT;
				target_pos[count1] = target_pos[count2];
				idxToListIdx[target[count1]] = count1;
			}
			count1++;
			count2++;
		}
		else
		{
			count2++;
		}
	}
	
	for (int i = 0; i < num_bubble; i++)
	{
		int bubble_idx = numParticles - i - 1;
		position[bubble_idx] = target_pos[i];
	}
	
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.idxToListIdx, idxToListIdx, numParticles * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.bubbleList, target, MAX_BUBBLE_NUM * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf.bubblePosList, target_pos, 3 * MAX_BUBBLE_NUM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(fbuf._position, position, 3 * numParticles * sizeof(float), cudaMemcpyHostToDevice));/**/
}

// surface tension //
void ComputeSurfaceTensionCUDA(int numParticles)
{
	ComputeSfParticlePhase << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeSfParticlePhase: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	ComputeSurfaceTensionParticle << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeSurfaceTensionParticle: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

// vorticity force //
void ComputeVorticityCUDA(int numParticles)
{
	ComputeVorticityParticle << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeVorticityParticle: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	ComputeForceVorticity << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeForceVorticity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}




// change:
void InsertParticlesCUDA(int numParticles)
{
	CUDA_SAFE_CALL(cudaMemset(fbuf.num_particle_grid, 0, fcudaParams.grid_number * sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(fbuf.grid_offset, 0, fcudaParams.grid_number * sizeof(uint)));

	insertParticles << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: insertParticles: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	//// test 
	//printf("num: %d\n", numParticles);
	//CUDA_SAFE_CALL(cudaMemcpy(ftemp._position, fbuf._position, sizeof(float3)*79120, cudaMemcpyDeviceToDevice));
	//test << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (ftemp, numParticles);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: test: %s\n", cudaGetErrorString(error));
	//}
	//cudaDeviceSynchronize();
}

inline bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }

inline int floorPow2(int n) {
#ifdef WIN32
	return 1 << (int)logb((float)n);
#else
	int exp;
	frexp((float)n, &exp);
	return 1 << (exp - 1);
#endif // WIN32

}

void PrefixSumCellsCUDA(int zero_offsets)
{
	// Prefix Sum - determine grid offsets
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = fcudaParams.grid_number;
	int numElem2 = int(numElem1 / blockSize) + 1;
	int numElem3 = int(numElem2 / blockSize) + 1;
	int threads = SCAN_BLOCKSIZE;
	int zon = 1;

	uint* array1 = fbuf.num_particle_grid;		// input
	uint* scan1 = fbuf.grid_offset;				// output
	uint* array2 = fbuf.aux_array1;
	uint* scan2 = fbuf.aux_scan1;
	uint* array3 = fbuf.aux_array2;
	uint* scan3 = fbuf.aux_scan2;

	if (numElem1 > SCAN_BLOCKSIZE * (unsigned long long)(SCAN_BLOCKSIZE) * SCAN_BLOCKSIZE) {
		printf("ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE\n");
	}

	// sum array1. output -> scan1, array2
	prefixSum << <numElem2, threads >> > (array1, scan1, array2, numElem1, zero_offsets);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: prefixSum1: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// sum array2. output -> scan2, array3
	prefixSum << <numElem3, threads >> > (array2, scan2, array3, numElem2, zon);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: prefixSum2: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	if (numElem3 > 1) {
		//uint narray[1] = { 0 };
		uint* narray = nullptr;
		// sum array3. output -> scan3
		prefixSum << <1, threads >> > (array3, scan3, narray, numElem3, zon);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: prefixSum3: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		// merge scan3 into scan2. output -> scan2
		prefixFixup << <numElem3, threads >> > (scan2, scan3, numElem2);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: prefixUp1: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();
	}

	// merge scan2 into scan1. output -> scan1
	prefixFixup << <numElem2, threads >> > (scan1, scan2, numElem1);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: prefixUp2: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void CountingSortFullCUDA(int numParticles)
{
	// Transfer particle data to temp buffers
	// (gpu-to-gpu) copy, no sync needed
    printf("cccccc%d\n", numParticles);
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._position, fbuf._position, sizeof(float3) * numParticles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ftemp._external_force, fbuf._external_force, sizeof(float3) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._mix_velocity, fbuf._mix_velocity, sizeof(float3) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._drift_velocity, fbuf._drift_velocity, sizeof(float3) * numParticles * MAX_PHASE_NUMBER, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._force, fbuf._force, sizeof(float3) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._alpha, fbuf._alpha, sizeof(float) * numParticles * MAX_PHASE_NUMBER, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._alpha_advanced, fbuf._alpha_advanced, sizeof(float) * numParticles * MAX_PHASE_NUMBER, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._delta_alpha, fbuf._delta_alpha, sizeof(float) * numParticles * MAX_PHASE_NUMBER, cudaMemcpyDeviceToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(ftemp._delta_mass_k, fbuf._delta_mass_k, sizeof(float) * numParticles * MAX_PHASE_NUMBER, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._alpha_sum, fbuf._alpha_sum, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._lambda, fbuf._lambda, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._mix_pressure, fbuf._mix_pressure, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._mix_density, fbuf._mix_density, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._inv_dens, fbuf._inv_dens, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._mix_rest_density, fbuf._mix_rest_density, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._viscosity, fbuf._viscosity, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._rest_mass, fbuf._rest_mass, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._particle_radius, fbuf._particle_radius, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._bound_phi, fbuf._bound_phi, sizeof(float) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._type, fbuf._type, sizeof(int) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._active, fbuf._active, sizeof(bool) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp.particle_grid_cellindex, fbuf.particle_grid_cellindex, sizeof(uint) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp.particle_grid_index, fbuf.particle_grid_index, sizeof(uint) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._tensor_K, fbuf._tensor_K, sizeof(tensor) * numParticles, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ftemp._tensor_F, fbuf._tensor_F, sizeof(tensor) * numParticles, cudaMemcpyDeviceToDevice));

	countingSortFull << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, ftemp, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: countingSortFull: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeShapeCUDA(float time_step, int numParticles)
{
	// compute boundary phi
	computePhi << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computePhi: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute density & pressure
	computeDensity << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeDensity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute shape tensor
	computeK << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeK: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute deformation gradient
	computeF << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeF: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void TestFunc(float time_step, int numParticles)
{
	// apply alpha
	setAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: setAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute drift velocity ( spend a lot of time here)
	computeDriftVel << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeDensity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeAlphaCUDA(float time_step, int numParticles)
{
	cudaError_t error;

	// apply alpha
	setAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: setAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute drift velocity ( spend a lot of time here)
	computeDriftVel << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeDensity: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// compute alpha transport iteration
	/*int max_iter = 2;
	int l = 1;
	while (l <= max_iter)
	{
		// spend a lot of time
		computeDelAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: computeDelAlpha: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();
		
			computeLambda << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles);
			error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf(stderr, "CUDA ERROR: computeLambda: %s\n", cudaGetErrorString(error));
			}
			cudaDeviceSynchronize();
		l += 1;
	}*/
	computeDelAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeDelAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();


	// compute volume correction
	computeCorrection << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeCorrection: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	// normalize alpha using mass fraction
	normalizeAlpha << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: normalizeAlpha: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void ComputeForceCUDA(float time_step, int numParticles)
{
	// compute boundary phi
	computeForce << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeForce: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}

void AdvanceParticleCUDA(float time_step, int numParticles, int frame)
{
	// compute boundary phi
	advanceParticles << <fcudaParams.particle_blocks, fcudaParams.particle_threads >> > (fbuf, numParticles, time_step, frame);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: advanceParticles: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
}
void CopyToComponent(float* pos, float* color,int numParticles) {
    cudaMemcpy(pos, fbuf._render_pos, numParticles * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(color, fbuf._color, numParticles * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
}


// ==================================== //


// * * * * * Kernel Function * * * * * //

// sph base function //

__device__ void collisionHandlingCUDA(float3* position, float3* velocity)
{
	const float3 vec_boundary_min = simData.boundary_min;
	const float3 vec_boundary_max = simData.boundary_max;
	const float particle_radius = simData.particle_radius;
	const float damp = simData.damp;

	if (position->x > vec_boundary_max.x - particle_radius)
	{
		position->x = vec_boundary_max.x - particle_radius;
		velocity->x *= -damp;
	}
	if (position->y > vec_boundary_max.y - particle_radius)
	{
		position->y = vec_boundary_max.y - particle_radius;
		velocity->y *= -damp;
	}
	if (position->z > vec_boundary_max.z - particle_radius)
	{
		position->z = vec_boundary_max.z - particle_radius;
		velocity->z *= -damp;
	}
	if (position->x < vec_boundary_min.x + particle_radius)
	{
		position->x = vec_boundary_min.x + particle_radius;
		velocity->x *= -damp;
	}
	if (position->y < vec_boundary_min.y + particle_radius)
	{
		position->y = vec_boundary_min.y + particle_radius;
		velocity->y *= -damp;
	}
	if (position->z < vec_boundary_min.z + particle_radius)
	{
		position->z = vec_boundary_min.z + particle_radius;
		velocity->z *= -damp;
	}
}

__device__ void limitHandlingCUDA(int i, float3* acceleration, float3* velocity, float3* pressure_force, float3* visc_force, float3* phase_force)
{
	int flag = 1;
	if (velocity->x > 10.0f)
	{
		velocity->x = 10.0f;
		flag = 0;
	}
	if (velocity->y > 10.0f)
	{
		velocity->y = 10.0f;
		flag = 0;
	}
	if (velocity->z > 10.0f)
	{
		velocity->z = 10.0f;
		flag = 0;
	}
	if (velocity->x < -10.0f)
	{
		velocity->x = -10.0f;
		flag = 0;
	}
	if (velocity->y < -10.0f)
	{
		velocity->y = -10.0f;
		flag = 0;
	}
	if (velocity->z < -10.0f)
	{
		velocity->z = -10.0f;
		flag = 0;
	}
	if (acceleration->x > 100.0f)
	{
		acceleration->x = 100.0f;
		flag = 0;
	}
	if (acceleration->y > 100.0f)
	{
		acceleration->y = 100.0f;
		flag = 0;
	}
	if (acceleration->z > 100.0f)
	{
		acceleration->z = 100.0f;
		flag = 0;
	}
	if (acceleration->x < -100.0f)
	{
		acceleration->x = -100.0f;
		flag = 0;
	}
	if (acceleration->y < -100.0f)
	{
		acceleration->y = -100.0f;
		flag = 0;
	}
	if (acceleration->z < -100.0f)
	{
		acceleration->z = -100.0f;
		flag = 0;
	}
	if (i % 10000 == 0)
	{
		if (flag == 0)
		{
			if (pressure_force || visc_force || phase_force)
				printf("index %d out of limit, velocity: %f, %f, %f, pressureforce: %f, %f, %f, viscforce:%f, %f, %f, phaseforce:%f, %f, %f\n", i, velocity->x, velocity->y, velocity->z, pressure_force->x, pressure_force->y, pressure_force->z, visc_force->x, visc_force->y, visc_force->z, phase_force->x, phase_force->y, phase_force->z);
			else
				printf("index %d out of limit, velocity:%f, %f, %f\n", i, velocity->x, velocity->y, velocity->z);
		}
	}
}

__device__ uint GetCell(float3 position, float3 grid_min, int3 grid_resolution, float grid_radius, int3& Cell)
{
	float px = position.x - grid_min.x;
	float py = position.y - grid_min.y;
	float pz = position.z - grid_min.z;

	Cell.x = (int)ceil(px / grid_radius);
	Cell.y = (int)ceil(py / grid_radius);
	Cell.z = (int)ceil(pz / grid_radius);

	return (uint)(Cell.x + grid_resolution.x * (Cell.y + grid_resolution.y * Cell.z));
}

__device__ float kernelM4CUDA(float dist, float sr)
{
	float s = dist / sr;
	float result;
	float factor = 2.546479089470325472f / (sr * sr * sr);
	if (dist < 0.0f || dist >= sr)
		return 0.0f;
	else
	{
		if (s < 0.5f)
		{
			result = 1.0f - 6.0 * s * s + 6.0f * s * s * s;
		}
		else
		{
			float tmp = 1.0f - s;
			result = 2.0 * tmp * tmp * tmp;
		}
	}
	return factor * result;
}

__device__ float kernelM4LutCUDA(float dist, float sr)
{
	int index = dist / sr * LUT_SIZE_CUDA;

	if (index >= LUT_SIZE_CUDA) return 0.0f;
	else return kernelM4CUDA(index * sr / LUT_SIZE_CUDA, sr);
}

__global__ void AllocateGrid(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	const float3 grid_min = simData.boundary_min - simData.grid_boundary_offset;
	const int3 grid_resolution = simData.grid_resolution;
	const float grid_radius = simData.grid_radius;

	int3 Cell;
	uint CellIndex = GetCell(buf._position[i], grid_min, grid_resolution, grid_radius, Cell);

	if (Cell.x >= 0 && Cell.x < grid_resolution.x && Cell.y >= 0 && Cell.y < grid_resolution.y && Cell.z >= 0 && Cell.z < grid_resolution.z)
	{
		buf.particle_grid_cellindex[i] = CellIndex;
		/*
		buf.next_particle_index[i] = buf.grid_particle_table[CellIndex];
		buf.grid_particle_table[CellIndex] = i;*/
		buf.next_particle_index[i] = atomicExch(&buf.grid_particle_table[CellIndex], i);
		atomicAdd(&buf.num_particle_grid[CellIndex], 1);
	}
	else
	{
		buf.particle_grid_cellindex[i] = GRID_UNDEF;
	}
}

// ==================================== //


// * * * * * Muti-phase Function * * * * * //

// compute rest density and pressure
__device__ float contributeMFDensity(int i, bufList buf, float3 pos, int& num_neighbor, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			result += cubicterm * buf._rest_mass[j];
			num_neighbor += 1;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeMFDensity(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	const float gamma = 7.0f;
	if (buf._type[i] == BOUND)
	{
		buf._mix_density[i] = 2000.0f;
		float relative_density = pow(2000.0f / buf._mix_rest_density[i], gamma);
		buf._mix_pressure[i] = max(0.0f, ((relative_density - 1.0f) * simData.gas_constant * buf._mix_rest_density[i]) / gamma);
		return;
	}
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum = 0.0f;
	float3 _position_i = buf._position[i];
	int num_neighbor = 0;
	const float _mass_i = buf._rest_mass[i];
	const float self_density = _mass_i * simData.kernel_self;
	for (int cell = 0; cell < 27; cell++)
	{
		sum += contributeMFDensity(i, buf, _position_i, num_neighbor, iCellIndex + simData.grid_search_offset[cell]);
	}
	sum += self_density;
	buf._mix_density[i] = sum;
	buf.particle_neighbor_number[i] = num_neighbor;
	//buf._mix_pressure[i] = max(0.0f, (sum - buf._mix_rest_density[i]) * simData.gas_constant);
	// wcsph
	if (simData._explicit)
	{
		float relative_density = pow(sum / buf._mix_rest_density[i], gamma);
		buf._mix_pressure[i] = max(0.0f, ((relative_density - 1.0f) * simData.gas_constant * buf._mix_rest_density[i]) / gamma);
	}
	buf._noted[i] = false;
	//if (i % 1000 == 0)
	//{
	//	printf("index: %d, type: %d, density: %f, rest_density: %f, pos_y: %f\n", i, buf._type[i], buf._mix_density[i], buf._mix_rest_density[i], buf._position[i].y);
	//}
}

// apply _alpha(t + dt) = _alpha_advanced(t)
__global__ void ApplyAlpha(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const int muloffset_i = i * MAX_PHASE_NUMBER;

	float _alpha_sum = 0.0f;
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		buf._alpha[muloffset_i + fcount] = buf._alpha_advanced[muloffset_i + fcount];
		_alpha_sum += buf._alpha_advanced[muloffset_i + fcount];
	}

	buf._alpha_sum[i] = _alpha_sum;
	if (_alpha_sum < 0.001f)
	{
		//printf("particle index:%d, alpha:%f, %f, alphasum:%f\n", i, buf._alpha[muloffset_i], buf._alpha[muloffset_i + 1], _alpha_sum);
	}

	if (isnan(_alpha_sum))
	{
		//printf("ApplyAlpha:index %d alphasum is nan\n", i);
	}
}

// compute drift velocity and normalize alpha
__device__ void DriftVelHandling(int i, float3& drift_vel)
{
	bool flag = 0;
	if (drift_vel.x > 2.0f)
	{
		drift_vel.x = 2.0f;
		flag = 1;
	}
	if (drift_vel.y > 2.0f)
	{
		drift_vel.y = 2.0f;
		flag = 1;
	}
	if (drift_vel.z > 2.0f)
	{
		drift_vel.z = 2.0f;
		flag = 1;
	}
	if (drift_vel.x < -2.0f)
	{
		drift_vel.x = -2.0f;
		flag = 1;
	}
	if (drift_vel.y < -2.0f)
	{
		drift_vel.y = -2.0f;
		flag = 1;
	}
	if (drift_vel.z < -2.0f)
	{
		drift_vel.z = -2.0f;
		flag = 1;
	}
	if (flag == 1);
	//printf("index: %d, drift_vel out of limit\n", i);
}
__global__ void ComputeDriftVelocity(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	if (buf._alpha_sum[i] < 0.01)
	{
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = make_float3(0.0f, 0.0f, 0.0f);
		}
		return;
	}

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;

	const float _pressure_i = buf._mix_pressure[i];
	const float3 _position_i = buf._position[i];
	const float3 _acceleration_i = -buf._force[i];

	// temporary container
	float3 forceterm[MAX_PHASE_NUMBER], pressureterm[MAX_PHASE_NUMBER], alphaterm[MAX_PHASE_NUMBER];
	float3 drift_velocity[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER], mass_fraction[MAX_PHASE_NUMBER];
	float _mix_density = 0.0f;
	float densitysum = 0.0f;

	// build alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		// normalization
		if (buf._alpha_sum[i] > 0.0001)
			_alpha_i[fcount] = buf._alpha[i * MAX_PHASE_NUMBER + fcount] / buf._alpha_sum[i];
		else
			_alpha_i[fcount] = 0.0f;
		_mix_density += _alpha_i[fcount] * simData.phase_density[fcount];
	}
	if (_mix_density == 0.0f)
	{
		//printf("ComputeDriftVelocity: index %d rest mix-density equals to zero\n", i);
		return;
	}

	// build mass_fraction and densitysum
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		mass_fraction[fcount] = _alpha_i[fcount] * simData.phase_density[fcount] / _mix_density;
		densitysum += mass_fraction[fcount] * simData.phase_density[fcount];
		forceterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
		pressureterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
		alphaterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
	}

	// kernel parameters
	float q;
	float dist, dist_square;
	float cubicterm, pterm;
	float3 pgrad[MAX_PHASE_NUMBER], alphagrad[MAX_PHASE_NUMBER], relative_alpha_grad[MAX_PHASE_NUMBER];
	float3 pgradsum, alphagradsum;

	// compute kernel grad value (pressureterm and alphaterm)
	for (int cell = 0; cell < 27; cell++)
	{
		const int SearchIndex = iCellIndex + simData.grid_search_offset[cell];

		if (SearchIndex <0 || SearchIndex >simData.grid_number - 1)
		{
			continue;
		}

		uint j = buf.grid_particle_table[SearchIndex];

		while (j != GRID_UNDEF)
		{
			if (j == i || buf._type[j] == BOUND)
			{
				j = buf.next_particle_index[j];
				continue;
			}

			float3 pos_ij = _position_i - buf._position[j];
			const float dx = pos_ij.x;
			const float dy = pos_ij.y;
			const float dz = pos_ij.z;
			dist_square = dx * dx + dy * dy + dz * dz;

			if (dist_square < smooth_radius_square)
			{
				dist = sqrt(dist_square);
				if (dist < 0.00001f)
				{
					dist = 0.00001f;
				}
				q = dist / smooth_radius;
				if (q <= 0.5f)
					cubicterm = simData.GradCubicSplineKern * (3.0f * q * q - 2.0f * q);
				else
					cubicterm = -simData.GradCubicSplineKern * (1.0f - q) * (1.0f - q);
				cubicterm *= buf._rest_mass[j] / (dist * buf._mix_density[j]);

				if (buf._alpha_sum[j] < 0.000001)
				{
					//printf("alphasum:%f\n", buf._alpha_sum[j]);
					j = buf.next_particle_index[j];
					continue;
				}

				pgradsum = make_float3(0.0f, 0.0f, 0.0f);
				alphagradsum = make_float3(0.0f, 0.0f, 0.0f);
				for (int fcount = 0; fcount < simData.phase_number; fcount++)
				{
					const float _alpha_j = buf._alpha[j * MAX_PHASE_NUMBER + fcount] / buf._alpha_sum[j];
					// pressure term
					if (simData.miscible)
						pterm = cubicterm * (-_alpha_i[fcount] * _pressure_i + _alpha_j * buf._mix_pressure[j]);
					else
						pterm = cubicterm * (-_pressure_i + buf._mix_pressure[j]);
					pgrad[fcount] = pterm * pos_ij;
					pgradsum += pgrad[fcount] * mass_fraction[fcount];
					// alpha term
					alphagrad[fcount] = (-_alpha_i[fcount] + _alpha_j) * cubicterm * pos_ij;
					if (_alpha_i[fcount] > 0.0001)
						relative_alpha_grad[fcount] = alphagrad[fcount] / _alpha_i[fcount];
					else
						relative_alpha_grad[fcount] = make_float3(0.0f, 0.0f, 0.0f);
					alphagradsum += mass_fraction[fcount] * relative_alpha_grad[fcount];
				}
				for (int fcount = 0; fcount < simData.phase_number; fcount++)
				{
					// index "j" is added here
					pressureterm[fcount] -= simData.tau * (pgrad[fcount] - pgradsum);
					alphaterm[fcount] -= simData.sigma * (relative_alpha_grad[fcount] - alphagradsum);
				}
			}
			j = buf.next_particle_index[j];
		}
	}

	// apply drift velocity
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		forceterm[fcount] = simData.tau * (simData.phase_density[fcount] - densitysum) * _acceleration_i;
		drift_velocity[fcount] = forceterm[fcount] + pressureterm[fcount] + alphaterm[fcount];
		/*
		if (drift_velocity[fcount].x > 100.0f || drift_velocity[fcount].x < -100.0f)
		{
			buf._noted[i] = true;
			printf("index: %d, phase: %d, alpha_k: %f, alpha_sum:%f, drift_velocity:%f, %f, %f\n",i, fcount, _alpha_i[fcount], buf._alpha_sum[i], drift_velocity[fcount].x, drift_velocity[fcount].y, drift_velocity[fcount].z);
			printf("index: %d, forceterm:%f, %f, %f, a:%f, %f, %f, alphaterm:%f, %f, %f\n",i, forceterm[fcount].x, forceterm[fcount].y, forceterm[fcount].z, _acceleration_i.x, _acceleration_i.y, _acceleration_i.z);
		}*/
		//DriftVelHandling(i, drift_velocity[fcount]);
		buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = drift_velocity[fcount];
		buf._alpha[i * MAX_PHASE_NUMBER + fcount] = _alpha_i[fcount];
	}

	if (isnan(_alpha_i[0]) || isnan(_alpha_i[1]))
	{
		//printf("ComputeDriftVelocity: index %d, alpha:%f, %f\n==========endComputeDriftVelocity\n", i, _alpha_i[0], _alpha_i[1]);
	}/*
	if (isnan(buf._drift_velocity[i * MAX_PHASE_NUMBER].y) || isnan(buf._drift_velocity[i * MAX_PHASE_NUMBER + 1].y) || isnan(buf._drift_velocity[i * MAX_PHASE_NUMBER].z) || isnan(buf._drift_velocity[i * MAX_PHASE_NUMBER + 1].z))
	{
		printf("ComputeDriftVelocity: index %d, drift-velocity:%f, %f, %f, %f, %f, %f\n", i, buf._drift_velocity[i * MAX_PHASE_NUMBER].x, buf._drift_velocity[i * MAX_PHASE_NUMBER].y, buf._drift_velocity[i * MAX_PHASE_NUMBER].z,
			buf._drift_velocity[i * MAX_PHASE_NUMBER + 1].x, buf._drift_velocity[i * MAX_PHASE_NUMBER + 1].y, buf._drift_velocity[i * MAX_PHASE_NUMBER + 1].z);
	}*/
}

// advance alpha
__device__ float* contributeAlphaChange(int i, bufList buf, float3 pos, float3 vel, float3* _drift_velocity_i, float* _alpha_i, int grid_index)
{
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return _alpha_change_i;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float _alpha_j_k;
	int muloffset_j;
	float term1[MAX_PHASE_NUMBER], term2[MAX_PHASE_NUMBER];
	float3 pos_ij, vel_ij, drift_vel_ij;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.GradCubicSplineKern * (3.0f * q * q - 2.0f * q);
			else
				cubicterm = -simData.GradCubicSplineKern * (1.0f - q) * (1.0f - q);
			cubicterm *= buf._rest_mass[j] / (dist * buf._mix_density[j]);
			vel_ij = vel - buf._mix_velocity[j];
			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				_alpha_j_k = buf._alpha[muloffset_j + fcount];
				_alpha_change_i[fcount] -= 0.5f * cubicterm * (_alpha_j_k + _alpha_i[fcount]) * (vel_ij.x * dx + vel_ij.y * dy + vel_ij.z * dz);
				term1[fcount] = 0.5f * cubicterm * (_alpha_j_k + _alpha_i[fcount]) * (vel_ij.x * dx + vel_ij.y * dy + vel_ij.z * dz);
				/*
				drift_vel_ij = make_float3((_alpha_j_k * buf._drift_velocity[muloffset_j + fcount].x + _alpha_i[fcount] * _drift_velocity_i[fcount].x),
					(_alpha_j_k * buf._drift_velocity[muloffset_j + fcount].y + _alpha_i[fcount] * _drift_velocity_i[fcount].y),
					(_alpha_j_k * buf._drift_velocity[muloffset_j + fcount].z + _alpha_i[fcount] * _drift_velocity_i[fcount].z));*/
				drift_vel_ij = _alpha_j_k * buf._drift_velocity[muloffset_j + fcount] + _alpha_i[fcount] * _drift_velocity_i[fcount];
				_alpha_change_i[fcount] += cubicterm * (drift_vel_ij.x * dx + drift_vel_ij.y * dy + drift_vel_ij.z * dz);
				term2[fcount] = cubicterm * (drift_vel_ij.x * dx + drift_vel_ij.y * dy + drift_vel_ij.z * dz);
			}
			if (buf._noted[i])
			{
				//printf("Search index:%d, term1: %f, %f, term2: %f, %f, vel_ij: %f, %f, %f, alpha_j: %f, %f\n", j, term1[0], term1[1], term2[0], term2[1], vel_ij.x, vel_ij.y, vel_ij.z, buf._alpha[muloffset_j], buf._alpha[muloffset_j+1]);
			}
		}
		j = buf.next_particle_index[j];
	}
	return _alpha_change_i;
}
__global__ void ComputeAlphaAdvance(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	if (buf._alpha_sum[i] < 0.01f)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	const float3 _position_i = buf._position[i];
	const float3 _velocity_i = buf._mix_velocity[i];

	// temporary container
	float3 _drift_velocity_i[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER], _alpha_change_i[MAX_PHASE_NUMBER];

	// build alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_i[fcount] = buf._alpha[muloffset_i + fcount];
		_alpha_change_i[fcount] = 0.0f;
		_drift_velocity_i[fcount] = buf._drift_velocity[muloffset_i + fcount];
	}

	// compute kernel grad value (divergence term)
	float* contribute_ptr;
	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = contributeAlphaChange(i, buf, _position_i, _velocity_i, _drift_velocity_i, _alpha_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			_alpha_change_i[fcount] -= contribute_ptr[fcount];
		}
	}
	if (isnan(_alpha_change_i[0]) || isnan(_alpha_change_i[1]))
	{
		printf("ComputeAlphaAdvance: alphachange is nan:%d\n", i);
		_alpha_change_i[0] = 0.0f;
		_alpha_change_i[1] = 0.0f;
	}
	// alpha advance
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_change_i[fcount] *= time_step;

		if (_alpha_change_i[fcount] < -0.99f)
		{
			_alpha_change_i[fcount] = -0.99f;
		}
		buf._delta_alpha[muloffset_i + fcount] = _alpha_change_i[fcount];
		buf._alpha_advanced[muloffset_i + fcount] = _alpha_change_i[fcount] + _alpha_i[fcount];
		if (buf._alpha_advanced[muloffset_i + fcount] < 0.0f)
		{
			//printf("index:%d, alpha is nagetive:%f, %f\n", i, buf._alpha_advanced[muloffset_i], buf._alpha_advanced[muloffset_i + 1]);
			buf._alpha_advanced[muloffset_i + fcount] = 0.0f;
		}
	}
	if (buf._noted[i])
	{
		//printf("index: %d is noted, alpha_change: %f, %f, alpha: %f, %f\n", i, _alpha_change_i[0], _alpha_change_i[1], (_alpha_i[0]+_alpha_change_i[0]), (_alpha_i[1]+_alpha_change_i[1]));
	}
	if (i % 500 == 0)
	{
		//printf("index: %d, alpha_change: %f, %f, alpha: %f, %f, phase: %d\n", i, _alpha_change_i[0], _alpha_change_i[1], _alpha_i[0], _alpha_i[1], buf._phase[i]);
	}
}

// compute correction of alpha and pressure
__global__ void ComputeCorrection(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	// temporary container
	float _alphasum = 0.0f;
	float _mix_density = 0.0f;
	float _mix_viscosity = 0.0f;

	float _mass_i = buf._rest_mass[i];
	float _density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };

	// alpha correction
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		//if (buf._alpha_advanced[muloffset_i + fcount] < 0.01f)
			//buf._alpha_advanced[muloffset_i + fcount] = 0.0f;

		_alphasum += buf._alpha_advanced[muloffset_i + fcount];
	}

	if (isnan(_alphasum))
	{
		//if (i % 1000 == 0)
			//printf("ComputeCorrection: index %d alphasum isnan\n", i);
	}

	if (_alphasum == 0.0f)
	{
		//printf("ComputeCorrection: index %d alphasum equals to zero\n", i);
		//buf._noted[i] = true;
	}

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_change_i[fcount] = buf._delta_alpha[muloffset_i + fcount];
		buf._alpha_advanced[muloffset_i + fcount] = buf._alpha_advanced[muloffset_i + fcount] / _alphasum;
		buf._delta_alpha[muloffset_i + fcount] = buf._alpha_advanced[muloffset_i + fcount] - buf._alpha[muloffset_i + fcount];
		_mix_density += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount];
		_mix_viscosity += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_visc[fcount];
	}

	// pressure correction
	float _delta_pressure = 0.0f;
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		float relative_density = pow(buf._mix_density[i] / _mix_density, 2);
		//_delta_pressure -= simData.gas_constant * simData.phase_density[fcount] * buf._delta_alpha[muloffset_i + fcount];
		_delta_pressure -= 0.5f * simData.gas_constant * simData.phase_density[fcount] * (1.0f * relative_density + 1.0f) * buf._delta_alpha[muloffset_i + fcount];
	}
	if (simData._explicit)
		buf._mix_pressure[i] += _delta_pressure;

	// mu && rho_m correction
	buf._mix_rest_density[i] = _mix_density;
	buf._viscosity[i] = _mix_viscosity;
	buf._eff_V[i] = buf._rest_mass[i] / buf._mix_rest_density[i];
	buf._particle_radius[i] = simData.particle_radius;
}

// compute force
__global__ void ComputeTDM(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND)
		return;

	if (buf._alpha_sum[i] < 0.01f)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float3 _position_i = buf._position[i];
	const float3 _velocity_i = buf._mix_velocity[i];
	const float _density_i = buf._mix_rest_density[i];
	const float _pressure_i = buf._mix_pressure[i];
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	const int phase = buf._phase[i];
	const float mass = simData.phase_mass[phase];
	const float _inv_mass_i = 1.0f / buf._rest_mass[i];
	const float pVol = 1.0f / _density_i;

	if (_density_i == 0.0f)
	{
		//printf("ComputeTDM: index %d density equals to zero\n", i);
		return;
	}

	// temporary container
	float3 _force = make_float3(0.0f, 0.0f, 0.0f);
	float3 _pressure_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _visc_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _phase_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _drift_velocity_i[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER];

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_i[fcount] = buf._alpha_advanced[muloffset_i + fcount];
		_drift_velocity_i[fcount] = buf._drift_velocity[muloffset_i + fcount];
	}

	// kernel parameters
	float q;
	float dist, dist_square;
	float cubicterm;
	float _alpha_j_k;
	int muloffset_j;
	float3 phasesum;
	float3 pressuregrad = make_float3(0.0f, 0.0f, 0.0f);
	float3 TM = make_float3(0.0f, 0.0f, 0.0f);
	float3 pos_ij;

	float3 pressure_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 visc_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 phase_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 surface_tension = make_float3(0.0f, 0.0f, 0.0f);

	// compute kernel grad value (TDM)
	for (int cell = 0; cell < 27; cell++)
	{
		const int SearchIndex = iCellIndex + simData.grid_search_offset[cell];

		if (SearchIndex <0 || SearchIndex >simData.grid_number - 1)
		{
			continue;
		}

		uint j = buf.grid_particle_table[SearchIndex];

		while (j != GRID_UNDEF)
		{
			if (j == i)
			{
				j = buf.next_particle_index[j];
				continue;
			}

			muloffset_j = j * MAX_PHASE_NUMBER;
			pos_ij = _position_i - buf._position[j];
			const float dx = pos_ij.x;
			const float dy = pos_ij.y;
			const float dz = pos_ij.z;
			dist_square = dx * dx + dy * dy + dz * dz;

			if (dist_square < smooth_radius_square)
			{
				const float nVol = mass / buf._mix_density[j];
				dist = sqrt(dist_square);
				if (dist < 0.00001f)
				{
					dist = 0.00001f;
				}
				const float kernelValue = kernelM4LutCUDA(dist, smooth_radius);
				q = dist / smooth_radius;
				if (q <= 0.5f)
					cubicterm = simData.GradCubicSplineKern * (3.0f * q * q - 2.0f * q);
				else
					cubicterm = -simData.GradCubicSplineKern * (1.0f - q) * (1.0f - q);
				if (buf._type[j] == FLUID)
				{
					cubicterm *= buf._rest_mass[j] / (dist * buf._mix_density[j]);
					surface_tension = -buf._rest_mass[j] * pos_ij * cubicterm;
					phasesum = make_float3(0.0f, 0.0f, 0.0f);
					if (simData._explicit)
						pressure_force = -0.5f * cubicterm * (_pressure_i + buf._mix_pressure[j]) * pos_ij * pVol;
					visc_force = -cubicterm * (buf._viscosity[i] + buf._viscosity[j]) * (buf._mix_velocity[j] - _velocity_i) * pVol;

					for (int fcount = 0; fcount < simData.phase_number; fcount++)
					{
						_alpha_j_k = buf._alpha_advanced[muloffset_j + fcount];
						phasesum += simData.phase_density[fcount] * pVol * (_alpha_j_k * buf._drift_velocity[muloffset_j + fcount] * dot(buf._drift_velocity[muloffset_j + fcount], pos_ij)
							+ _alpha_i[fcount] * _drift_velocity_i[fcount] * dot(_drift_velocity_i[fcount], pos_ij));
					}
					phase_force = -cubicterm * phasesum;
				}
				else
				{
					phase_force = make_float3(0.0f, 0.0f, 0.0f);
					if (buf._active[j] == true)
					{
						float3 vel_ij = buf._mix_velocity[j] - _velocity_i;
						cubicterm *= buf._bound_phi[j] * buf._mix_rest_density[i] / (dist * buf._mix_density[i]);
						pressure_force = -cubicterm * pVol * _pressure_i * pos_ij;
						visc_force = -cubicterm * simData.viscosity * vel_ij * pVol;
					}
					else
					{
						pressure_force = make_float3(0.0f, 0.0f, 0.0f);
						visc_force = make_float3(0.0f, 0.0f, 0.0f);
					}
				}
				_force += (pressure_force + visc_force + phase_force + simData.surface_tension_factor * _inv_mass_i * surface_tension);
				_pressure_part += pressure_force;
				_visc_part += visc_force;
				_phase_part += phase_force;
			}
			j = buf.next_particle_index[j];
		}
	}
	buf._force[i] = _force;
	buf._pressure_force[i] = _pressure_part;
	buf._visc_force[i] = _visc_part;
	buf._phase_force[i] = _phase_part;
	if (buf._noted[i])
	{
		//printf("index: %d is noted, pressure_force: %f, %f, %f\nTDM: %f, %f, %f\n", i, _pressure_part.x, _pressure_part.y, _pressure_part.z, _phase_part.x, _phase_part.y, _phase_part.z);
	}
}

// update particles with stiff-bound
__global__ void UpdateMFParticles(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._lock[i] == 1)
		return;
	float3 position = buf._position[i];
	float3 velocity = buf._mix_velocity[i];
	float3 acceleration = buf._force[i] + simData.gravity-0.3f * buf._surface_tension[i] + 0.00006f * buf._vorticity_force[i];
	//if (i % 1000 == 0)
	//	printf("index: %d, vorticity: %f, %f, %f\n", i, buf._vorticity_force[i].x, buf._vorticity_force[i].y, buf._vorticity_force[i].z);
	if (buf._alpha_advanced[MAX_PHASE_NUMBER * i + 1] > 0.5f)
		acceleration += -30.0f*(buf._alpha_advanced[MAX_PHASE_NUMBER * i + 1] - 0.5f) * simData.gravity;
	float3 pressure_force = buf._pressure_force[i];
	float3 visc_force = buf._visc_force[i];
	float3 phase_force = buf._phase_force[i];
	float3 norm;
	float3 color = make_float3(0.0f, 0.0f, 0.0f);
	float diff, adj;
	float stiffness = 10000.0f;
	
	velocity += time_step * acceleration;
	position += time_step * velocity;
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		color += buf._alpha_advanced[i * MAX_PHASE_NUMBER + fcount] * simData.phase_color[fcount];
	}

	// bound
	const float vel_mag = sqrt(dot(velocity, velocity));
	if (vel_mag > 15.0f)
	{
		velocity.x /= vel_mag;
		velocity.y /= vel_mag;
		velocity.z /= vel_mag;
	}

	//if (position.y > simData.boundary_max.y)
	//{
	//	buf._render[i] = false;
	//	buf._active[i] = false;
	//}

	buf._position[i] = position;
	buf._mix_velocity[i] = velocity;
	buf._velocity[i] = velocity;
	buf._color[i] = color;
	//if (buf._noted[i])
	//{
	//	printf("index: %d is noted, vel: %f, %f, %f, pos: %f, %f, %f\n", i, velocity.x, velocity.y, velocity.z, 40.0f * position.x, 40.0f * position.y, 40.0f * position.z);
	//}
	//printf("%d, %f, %f, %f\n", i, position.x, position.y, position.z);

	//if (i % 10000 == 0)
	//	printf("i: %d, mass: %f, density: %f\n", i, buf._rest_mass[i], buf._mix_density[i]);
}


// peridynamics //

__device__ void printTensor(int i, tensor A)
{
	printf("index: %d, tensor:\n| %.10f, %.10f, %.10f |\n| %.10f, %.10f, %.10f |\n| %.10f, %.10f, %.10f |\n", i, A.tensor[0], A.tensor[1], A.tensor[2], A.tensor[3], A.tensor[4], A.tensor[5], A.tensor[6], A.tensor[7], A.tensor[8]);
}
__device__ void initTensor(tensor& A)
{
	A.tensor[0] = A.tensor[4] = A.tensor[8] = 1.0f;
	A.tensor[1] = A.tensor[3] = 0.0f;
	A.tensor[2] = A.tensor[6] = 0.0f;
	A.tensor[5] = A.tensor[7] = 0.0f;
}
__device__ float tensorDet(tensor A)
{
	float result = 0.0f;
	result += A.tensor[0] * (A.tensor[4] * A.tensor[8] - A.tensor[5] * A.tensor[7]);
	result -= A.tensor[1] * (A.tensor[3] * A.tensor[8] - A.tensor[5] * A.tensor[6]);
	result += A.tensor[2] * (A.tensor[3] * A.tensor[7] - A.tensor[4] * A.tensor[6]);
	return result;
}
__device__ void TensorMutiply(tensor A, tensor B, tensor& C)
{
	int index;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			index = 3 * i + j;
			C.tensor[index] = A.tensor[3 * i] * B.tensor[j] + A.tensor[3 * i + 1] * B.tensor[3 + j] + A.tensor[3 * i + 2] * B.tensor[6 + j];
		}
	}
}
__device__ tensor TensorMutiply(tensor A, tensor B)
{
	tensor C;
	int index;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			index = 3 * i + j;
			C.tensor[index] = A.tensor[3 * i] * B.tensor[j] + A.tensor[3 * i + 1] * B.tensor[3 + j] + A.tensor[3 * i + 2] * B.tensor[6 + j];
		}
	}

	return C;
}
__device__ void TensorInverse(tensor& K)
{
	tensor L = { {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f} };
	tensor U = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	tensor L_inv = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	tensor U_inv = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	int index;
	int i, j;
	// caculate tensor L, U
	// step 1:
	i = 0;
	for (j = 0; j < 3; j++)
	{
		index = 3 * i + j;
		U.tensor[index] = K.tensor[index];
	}
	// step 2:
	j = 0;
	for (i = 1; i < 3; i++)
	{
		index = 3 * i + j;
		L.tensor[index] = K.tensor[index] / U.tensor[0];
	}
	// step 3:
	for (i = 1; i < 3; i++)
	{
		for (j = i; j < 3; j++)
		{
			index = 3 * i + j;
			float sum1 = 0.0f;
			float sum2 = 0.0f;
			for (int t = 0; t < i; t++)
			{
				sum1 += L.tensor[3 * i + t] * U.tensor[3 * t + j];
				sum2 += L.tensor[3 * j + t] * U.tensor[3 * t + i];
			}
			U.tensor[index] = K.tensor[index] - sum1;
			L.tensor[3 * j + i] = (K.tensor[3 * j + i] - sum2) / U.tensor[3 * i + i];
		}
	}
	// caculate L_inv, U_inv
	// step 1: caculate L_inv
	for (j = 0; j < 3; j++)
	{
		for (i = j; i < 3; i++)
		{
			index = 3 * i + j;
			if (i == j)
				L_inv.tensor[index] = 1.0f / L.tensor[index];
			else
			{
				if (i < j)
				{
					L_inv.tensor[index] = 0.0f;
				}
				else
				{
					float sum = 0.0f;
					for (int k = j; k < i; k++)
					{
						sum += L.tensor[3 * i + k] * L_inv.tensor[3 * k + j];
					}
					L_inv.tensor[index] = -L_inv.tensor[3 * j + j] * sum;
				}
			}
		}
	}
	//step 2: caculate U_inv
	for (j = 0; j < 3; j++)
	{
		for (i = j; i >= 0; i--)
		{
			index = 3 * i + j;
			if (i == j)
			{
				U_inv.tensor[index] = 1.0f / U.tensor[index];
			}
			else
			{
				if (i > j)
				{
					U_inv.tensor[index] = 0.0f;
				}
				else
				{
					float sum = 0.0f;
					for (int k = i + 1; k <= j; k++)
					{
						sum += U.tensor[3 * i + k] * U_inv.tensor[3 * k + j];
					}
					U_inv.tensor[index] = -(1.0f / U.tensor[3 * i + i]) * sum;
				}
			}
		}
	}
	// step 3: A_inv = U_inv * L_inv
	TensorMutiply(U_inv, L_inv, K);
}
__device__ float contributeTensorK(int i, bufList buf, float3 pos, int grid_index, tensor& K, int& num_neighbor)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	float kterm;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			kterm = (buf._rest_mass[j] / buf._mix_density[j]) * cubicterm;
			float test_value = (buf._rest_mass[i] / buf._mix_density[i]) * simData.CubicSplineKern * simData.particle_radius * simData.particle_radius;
			// diag term:
			K.tensor[0] += kterm * pos_ij.x * pos_ij.x;
			K.tensor[4] += kterm * pos_ij.y * pos_ij.y;
			K.tensor[8] += kterm * pos_ij.z * pos_ij.z;
			// others:
			K.tensor[1] += kterm * pos_ij.x * pos_ij.y;
			K.tensor[2] += kterm * pos_ij.x * pos_ij.z;
			K.tensor[5] += kterm * pos_ij.y * pos_ij.z;
			result += kterm;
			num_neighbor += 1;
			if (i == simData.test_index)
				printf("Search_index:%d, dist = %f r, K: %f, %f, %f, weight:%f, K_self: %f\n", j, 4 * dist / smooth_radius, kterm * dx * dx, kterm * dy * dy, kterm * dz * dz, kterm, test_value);
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeTensorK(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];
	tensor K = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
	{
		K.tensor[0] = 8000.0f;
		K.tensor[4] = 8000.0f;
		K.tensor[8] = 8000.0f;
		buf._tensor_K[i] = K;
		return;
	}
	float self_contribute = 0.0002f;
	float kernelsum = 0.0f;
	int neighbor_num = 0;
	const float smooth_radius = simData.smooth_radius;
	const float particle_radius = buf._particle_radius[i];
	for (int cell = 0; cell < 27; cell++)
	{
		kernelsum += contributeTensorK(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell], K, neighbor_num);
	}
	kernelsum += simData.CubicSplineKern * (buf._rest_mass[i] / buf._mix_density[i]);
	//value = simData.CubicSplineKern * pow((smooth_radius * 0.25f), 5) * (4.0f * MY_PI / 3.0f) * 0.1581f;
	//self_contribute = (buf._rest_mass[i] / buf._mix_density[i]) * simData.CubicSplineKern * simData.particle_radius * simData.particle_radius;
	self_contribute = (buf._rest_mass[i] / buf._mix_density[i]) * simData.CubicSplineKern * particle_radius * particle_radius;
	// Symmetric Tensor K:
	K.tensor[3] = K.tensor[1];
	K.tensor[6] = K.tensor[2];
	K.tensor[7] = K.tensor[5];
	/*
	K.tensor[3] = 0.0f;
	K.tensor[6] = 0.0f;
	K.tensor[7] = 0.0f;
	K.tensor[1] = 0.0f;
	K.tensor[2] = 0.0f;
	K.tensor[5] = 0.0f;*/
	// self contribute
	K.tensor[0] += self_contribute;
	K.tensor[4] += self_contribute;
	K.tensor[8] += self_contribute;
	for (int index = 0; index < 9; index++)
	{
		K.tensor[index] /= kernelsum;
	}
	TensorInverse(K);
	buf._tensor_K[i] = K;
	buf.particle_neighbor_number[i] = neighbor_num;
}

__device__ float contributeTensorF(int i, bufList buf, float3 pos, float3 vel, tensor K, float dt, int grid_index, tensor& F)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	float kterm; // kernel value
	//float3 inv_K_mul_xi; // inv(K) * \xi
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float3 u;  // u = rel_vel * dt
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			kterm = (buf._rest_mass[j] / buf._mix_density[j]) * cubicterm;
			u = -(buf._mix_velocity[j] - vel) * dt;
			K_xi_x = K.tensor[0] * dx + K.tensor[3] * dy + K.tensor[6] * dz;
			K_xi_y = K.tensor[1] * dx + K.tensor[4] * dy + K.tensor[7] * dz;
			K_xi_z = K.tensor[2] * dx + K.tensor[5] * dy + K.tensor[8] * dz;
			// diag term:
			F.tensor[0] += kterm * (pos_ij.x + u.x) * K_xi_x;
			F.tensor[4] += kterm * (pos_ij.y + u.y) * K_xi_y;
			F.tensor[8] += kterm * (pos_ij.z + u.z) * K_xi_z;
			// others:
			F.tensor[1] += kterm * (pos_ij.x + u.x) * K_xi_y;
			F.tensor[2] += kterm * (pos_ij.x + u.x) * K_xi_z;
			F.tensor[3] += kterm * (pos_ij.y + u.y) * K_xi_x;
			F.tensor[5] += kterm * (pos_ij.y + u.y) * K_xi_z;
			F.tensor[6] += kterm * (pos_ij.z + u.z) * K_xi_x;
			F.tensor[7] += kterm * (pos_ij.z + u.z) * K_xi_y;
			result += kterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeTensorF(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];
	float3 _velocity_i = buf._mix_velocity[i];
	tensor F = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
	{
		F.tensor[0] = 1.0f;
		F.tensor[4] = 1.0f;
		F.tensor[8] = 1.0f;
		buf._tensor_F[i] = F;
		return;
	}
	tensor K = buf._tensor_K[i];
	float detF;   // detF = |F|
	float self_contribute[9];
	float kernelsum = 0.0f;
	const float smooth_radius = simData.smooth_radius;
	const float particle_radius = buf._particle_radius[i];
	for (int cell = 0; cell < 27; cell++)
	{
		kernelsum += contributeTensorF(i, buf, _position_i, _velocity_i, K, time_step, iCellIndex + simData.grid_search_offset[cell], F);
	}
	kernelsum += simData.CubicSplineKern * (buf._rest_mass[i] / buf._mix_density[i]);
	float constant = simData.CubicSplineKern * (buf._rest_mass[i] / buf._mix_density[i]) * particle_radius;
	// caculate the missing tensor-self
	self_contribute[0] = (particle_radius) * (K.tensor[0] + K.tensor[3] + K.tensor[6]);
	self_contribute[1] = 0.0f;
	self_contribute[2] = 0.0f;
	self_contribute[3] = 0.0f;
	self_contribute[4] = (particle_radius) * (K.tensor[1] + K.tensor[4] + K.tensor[7]);
	self_contribute[5] = 0.0f;
	self_contribute[6] = 0.0f;
	self_contribute[7] = 0.0f;
	self_contribute[8] = (particle_radius) * (K.tensor[2] + K.tensor[5] + K.tensor[8]);
	for (int index = 0; index < 9; index++)
	{
		self_contribute[index] *= constant;
		F.tensor[index] += self_contribute[index];
		F.tensor[index] /= kernelsum;
	}
	//if (F.tensor[0] == 0.0f)
		//F.tensor[0] = 1.0f;
	detF = tensorDet(F);
	if (i == simData.test_index)
	{
		printf("tensor F:\n");
		printf("det F: %f, value: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", detF, self_contribute[0], self_contribute[1], self_contribute[2], self_contribute[3], self_contribute[4], self_contribute[5], self_contribute[6], self_contribute[7], self_contribute[8]);
		printTensor(i, F);
	}
	if (buf.particle_neighbor_number[i] < 5)
	{
		//initTensor(F);
		//printf("index: %d, num_neighbor: %d\n", i, buf.particle_neighbor_number[i]);
	}
	if (detF < 0.1f)
		initTensor(F);
	else
		TensorInverse(F);
	buf._tensor_F[i] = F;
	if (i == simData.test_index)
	{
		//printf("index:%d, tensor_K:%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, sum:%f, mass:%f, density:%f\n",i, K.tensor[0], K.tensor[1], K.tensor[2], K.tensor[3], K.tensor[4], K.tensor[5], K.tensor[6], K.tensor[7], K.tensor[8], kernelsum, buf._rest_mass[i], buf._mix_density[i]);
		//printTensor(i, F);
	}
}

// ======================================= //



// * * * * * PD Multi-phase Fluid * * * * * //

// muti-phase in peridynamics //
__device__ float3 DriftVelocityBound(float3& drift_velociy_k, float bound_value)
{
	bool _symbol;
	if (abs(drift_velociy_k.x) > bound_value)
	{
		_symbol = (abs(drift_velociy_k.x) > 0) ? true : false;
		if (_symbol)
			drift_velociy_k.x = bound_value;
		else
			drift_velociy_k.x = -bound_value;
	}
	if (abs(drift_velociy_k.y) > bound_value)
	{
		_symbol = (abs(drift_velociy_k.y) > 0) ? true : false;
		if (_symbol)
			drift_velociy_k.y = bound_value;
		else
			drift_velociy_k.y = -bound_value;
	}
	if (abs(drift_velociy_k.z) > bound_value)
	{
		_symbol = (abs(drift_velociy_k.z) > 0) ? true : false;
		if (_symbol)
			drift_velociy_k.z = bound_value;
		else
			drift_velociy_k.z = -bound_value;
	}
	return drift_velociy_k;
}
__global__ void ComputeDriftVelocityPeridynamics(bufList buf, int numParticles, float bound_vel, float factor)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	if (buf._alpha_sum[i] < 0.01)
	{
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = make_float3(0.0f, 0.0f, 0.0f);
		}
		return;
	}

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;

	const float _pressure_i = buf._mix_pressure[i];
	const float3 _position_i = buf._position[i];
	const float3 _acceleration_i = -buf._force[i];
	tensor K = buf._tensor_K[i];

	// temporary container
	float3 forceterm[MAX_PHASE_NUMBER], pressureterm[MAX_PHASE_NUMBER], alphaterm[MAX_PHASE_NUMBER];
	float3 drift_velocity[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER], mass_fraction[MAX_PHASE_NUMBER];
	float _mix_density = 0.0f;
	float sum_mass_fraction_mul_dens = 0.0f;
	// build alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		// normalization
		if (buf._alpha_sum[i] > 0.0001)
			_alpha_i[fcount] = buf._alpha[i * MAX_PHASE_NUMBER + fcount];// / buf._alpha_sum[i];
		else
			_alpha_i[fcount] = 0.0f;
		_mix_density += _alpha_i[fcount] * simData.phase_density[fcount];
	}
	if (_mix_density == 0.0f && i %1000==0)
	{
		//printf("ComputeDriftVelocity: index %d rest mix-density equals to zero, type: %d\nalpha:%f, %f, %f, %f, %f\n", i, buf._type[i],
		//	buf._alpha[i * MAX_PHASE_NUMBER + 0], buf._alpha[i * MAX_PHASE_NUMBER + 1], buf._alpha[i * MAX_PHASE_NUMBER + 2], buf._alpha[i * MAX_PHASE_NUMBER + 3], buf._alpha[i * MAX_PHASE_NUMBER + 4]);
		return;
	}

	// build mass_fraction and densitysum
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		mass_fraction[fcount] = _alpha_i[fcount] * simData.phase_density[fcount] / _mix_density;
		sum_mass_fraction_mul_dens += mass_fraction[fcount] * simData.phase_density[fcount];
		forceterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
		pressureterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
		alphaterm[fcount] = make_float3(0.0f, 0.0f, 0.0f);
	}

	// kernel parameters
	float q;
	float dist, dist_square;
	float cubicterm, pterm;
	float3 inv_K_mul_xi; // inv(K) * \xi
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	float3 pgrad[MAX_PHASE_NUMBER], alphagrad[MAX_PHASE_NUMBER], relative_alpha_grad[MAX_PHASE_NUMBER];
	float3 pgradsum, alphagradsum;
	float3 pos_ij;

	// compute kernel grad value (pressureterm and alphaterm)
	for (int cell = 0; cell < 27; cell++)
	{
		const int SearchIndex = iCellIndex + simData.grid_search_offset[cell];

		if (SearchIndex <0 || SearchIndex >simData.grid_number - 1)
		{
			continue;
		}

		uint j = buf.grid_particle_table[SearchIndex];

		while (j != GRID_UNDEF)
		{
			if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID || buf.bubble[j] == true)
			{
				j = buf.next_particle_index[j];
				continue;
			}

			pos_ij = _position_i - buf._position[j];
			const float dx = pos_ij.x;
			const float dy = pos_ij.y;
			const float dz = pos_ij.z;
			dist_square = dx * dx + dy * dy + dz * dz;
			if (dist_square < smooth_radius_square)
			{
				dist = sqrt(dist_square);
				if (dist < 0.00001f)
				{
					dist = 0.00001f;
				}
				q = dist / smooth_radius;
				if (q <= 0.5f)
					cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
				else
					cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
				_volume_j = buf._rest_mass[j] / buf._mix_density[j];
				//cubicterm *= -(_volume_i * _volume_j) / (_volume_i + _volume_j);
				cubicterm *= -_volume_j;
				K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
				K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
				K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
				inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);

				if (buf._alpha_sum[j] < 0.000001)
				{
					j = buf.next_particle_index[j];
					continue;
				}
				pgradsum = make_float3(0.0f, 0.0f, 0.0f);
				alphagradsum = make_float3(0.0f, 0.0f, 0.0f);
				for (int fcount = 0; fcount < simData.phase_number; fcount++)
				{
					const float _alpha_j = buf._alpha[j * MAX_PHASE_NUMBER + fcount] / buf._alpha_sum[j];
					// pressure term
					if (simData.miscible)
						pterm = cubicterm * (-_alpha_i[fcount] * _pressure_i + _alpha_j * buf._mix_pressure[j]);
					else
						pterm = cubicterm * (-_pressure_i + buf._mix_pressure[j]);/**/



					// for example testing
					/*
					if (fcount == 1)
					{
						pterm = cubicterm * (-_pressure_i + buf._mix_pressure[j]);
					}
					else
					{
						pterm = cubicterm * (-_alpha_i[fcount] * _pressure_i + _alpha_j * buf._mix_pressure[j]);
					}*/

					pgrad[fcount] = pterm * inv_K_mul_xi;
					pgradsum += pgrad[fcount] * mass_fraction[fcount];
					// alpha term
					alphagrad[fcount] = (-_alpha_i[fcount] + _alpha_j) * cubicterm * inv_K_mul_xi;
					if (_alpha_i[fcount] > 0.0001)
						relative_alpha_grad[fcount] = alphagrad[fcount] / _alpha_i[fcount];
					else
						relative_alpha_grad[fcount] = make_float3(0.0f, 0.0f, 0.0f);
					alphagradsum += mass_fraction[fcount] * relative_alpha_grad[fcount];
				}
				for (int fcount = 0; fcount < simData.phase_number; fcount++)
				{
					// index "j" is added here
					pressureterm[fcount] -= simData.tau * (pgrad[fcount] - pgradsum) * factor;
					alphaterm[fcount] -= simData.sigma * (relative_alpha_grad[fcount] - alphagradsum);
				}
			}
			j = buf.next_particle_index[j];
		}
	}

	// apply drift velocity
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		forceterm[fcount] = simData.tau * (simData.phase_density[fcount] - sum_mass_fraction_mul_dens) * _acceleration_i * factor;
		drift_velocity[fcount] = forceterm[fcount] + pressureterm[fcount] + alphaterm[fcount];
		buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = DriftVelocityBound(drift_velocity[fcount], bound_vel);
		buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = drift_velocity[fcount];
		buf._alpha[i * MAX_PHASE_NUMBER + fcount] = _alpha_i[fcount];
	}

	if (isnan(_alpha_i[0]) || isnan(_alpha_i[1]))
	{
		//printf("ComputeDriftVelocity: index %d, alpha:%f, %f\n==========endComputeDriftVelocity\n", i, _alpha_i[0], _alpha_i[1]);
	}

	// for alpha transport
	buf._lambda[i] = 1.0f;
	//if (i % 200 == 0)
		//printf("i: %d, drift velocity: %f, %f, %f,  %f, %f, %f\n",i, drift_velocity[0].x, drift_velocity[0].y, drift_velocity[0].z, drift_velocity[1].x, drift_velocity[1].y, drift_velocity[1]);
}

// SdV means (delta V) / (V * delta t)
__device__ float contributeSdVPeridynamics(int i, bufList buf, float3 pos, float3 mix_vel, float volume, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist, dist_square;
	float q, cubicterm;
	float s_dv_term;
	float _volume_j;
	float3 inv_K_mul_xi; // inv(K) * \xi
	float3 inv_F_mul_inv_K_mul_xi; // inv(F) * inv(K) * \xi
	float3 inv_F_mul_inv_K_mul_xi_j; // inv(F) * inv(K) * \xi for neighbor j
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float K_xi_x_j, K_xi_y_j, K_xi_z_j; // components of [inv(K) * \xi] for neighbor j
	float F_K_xi_x, F_K_xi_y, F_K_xi_z; // components of [inv(F) * inv(K) * \xi]
	float F_K_xi_x_j, F_K_xi_y_j, F_K_xi_z_j; // components of [inv(F) * inv(K) * \xi] for neighbor j
	//const float smooth_radius = simData.smooth_radius;
	//const float smooth_radius_square = smooth_radius * smooth_radius;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float pairwise_factor;
	float3 pos_ij, vel_ij, drift_vel_ij;
	tensor K = buf._tensor_K[i];
	tensor F = buf._tensor_F[i];
	tensor K_j, F_j;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			//cubicterm *= -(volume * _volume_j) / (volume + _volume_j);
			pairwise_factor = 2.0f * buf._rest_mass[j] / (buf._rest_mass[j] + buf._rest_mass[i]);
			cubicterm *= -_volume_j;
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);
			F_K_xi_x = F.tensor[0] * K_xi_x + F.tensor[1] * K_xi_y + F.tensor[2] * K_xi_z;
			F_K_xi_y = F.tensor[3] * K_xi_x + F.tensor[4] * K_xi_y + F.tensor[5] * K_xi_z;
			F_K_xi_z = F.tensor[6] * K_xi_x + F.tensor[7] * K_xi_y + F.tensor[8] * K_xi_z;
			inv_F_mul_inv_K_mul_xi = make_float3(F_K_xi_x, F_K_xi_y, F_K_xi_z);
			// j tensor
			K_j = buf._tensor_K[j];
			F_j = buf._tensor_F[j];
			K_xi_x_j = K_j.tensor[0] * dx + K_j.tensor[1] * dy + K_j.tensor[2] * dz;
			K_xi_y_j = K_j.tensor[3] * dx + K_j.tensor[4] * dy + K_j.tensor[5] * dz;
			K_xi_z_j = K_j.tensor[6] * dx + K_j.tensor[7] * dy + K_j.tensor[8] * dz;

			F_K_xi_x_j = F_j.tensor[0] * K_xi_x_j + F_j.tensor[1] * K_xi_y_j + F_j.tensor[2] * K_xi_z_j;
			F_K_xi_y_j = F_j.tensor[3] * K_xi_x_j + F_j.tensor[4] * K_xi_y_j + F_j.tensor[5] * K_xi_z_j;
			F_K_xi_z_j = F_j.tensor[6] * K_xi_x_j + F_j.tensor[7] * K_xi_y_j + F_j.tensor[8] * K_xi_z_j;
			inv_F_mul_inv_K_mul_xi_j = make_float3(F_K_xi_x_j, F_K_xi_y_j, F_K_xi_z_j);
			//inv_F_mul_inv_K_mul_xi = 0.5f * (inv_F_mul_inv_K_mul_xi + inv_F_mul_inv_K_mul_xi_j);
			vel_ij = buf._mix_velocity[j] - mix_vel;
			s_dv_term = dot(vel_ij, inv_F_mul_inv_K_mul_xi);
			result += s_dv_term * cubicterm * pairwise_factor;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeSdVPeridynamics(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
	{
		buf._SdV[i] = 0.0f;
		return;
	}
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum = 0.0f;
	float3 _position_i = buf._position[i];
	float3 _mix_vel_i = buf._mix_velocity[i];
	const float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	for (int cell = 0; cell < 27; cell++)
	{
		sum += contributeSdVPeridynamics(i, buf, _position_i, _mix_vel_i, _volume_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	buf._SdV[i] = sum;
	if (i % 1000 == 0)
	{
		//printf("index:%d, SdV:%f\n", i, sum);
	}
}

__device__ float* contributeAlphaChangePeridynamics(int i, bufList buf, float3 pos, float3 vel, float3* _drift_velocity_i, float* _alpha_i, int grid_index)
{
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return _alpha_change_i;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float _alpha_j_k;
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	float term1[MAX_PHASE_NUMBER], term2[MAX_PHASE_NUMBER];
	float gradterm1[MAX_PHASE_NUMBER], gradterm2[MAX_PHASE_NUMBER];
	float apx, apy, apz;
	float afx, afy, afz;
	float apx_j, apy_j, apz_j, afx_j, afy_j, afz_j;
	float3 pos_ij, vel_ij, drift_vel_ij;
	float3 p_K;   // p_K = K * x
	float3 s_K;   // s_K = F * p_K
	float3 s_K_j;
	tensor K = buf._tensor_K[i];
	tensor F = buf._tensor_F[i];
	tensor K_j, F_j;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			cubicterm *= -(_volume_i * _volume_j) / (_volume_i + _volume_j);
			//cubicterm *= -buf._rest_mass[j] / (buf._mix_density[j]);
			apx = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			apy = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			apz = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			p_K = make_float3(apx, apy, apz);
			afx = F.tensor[0] * apx + F.tensor[1] * apy + F.tensor[2] * apz;
			afy = F.tensor[3] * apx + F.tensor[4] * apy + F.tensor[5] * apz;
			afz = F.tensor[6] * apx + F.tensor[7] * apy + F.tensor[8] * apz;
			s_K = make_float3(afx, afy, afz);
			// j tensor
			K_j = buf._tensor_K[j];
			F_j = buf._tensor_F[j];
			apx_j = K_j.tensor[0] * dx + K_j.tensor[1] * dy + K_j.tensor[2] * dz;
			apy_j = K_j.tensor[3] * dx + K_j.tensor[4] * dy + K_j.tensor[5] * dz;
			apz_j = K_j.tensor[6] * dx + K_j.tensor[7] * dy + K_j.tensor[8] * dz;

			afx_j = F_j.tensor[0] * apx_j + F_j.tensor[1] * apy_j + F_j.tensor[2] * apz_j;
			afy_j = F_j.tensor[3] * apx_j + F_j.tensor[4] * apy_j + F_j.tensor[5] * apz_j;
			afz_j = F_j.tensor[6] * apx_j + F_j.tensor[7] * apy_j + F_j.tensor[8] * apz_j;
			s_K_j = make_float3(afx_j, afy_j, afz_j);
			s_K = 0.5f * (s_K + s_K_j);
			vel_ij = buf._mix_velocity[j] - vel;
			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				_alpha_j_k = buf._alpha[muloffset_j + fcount];
				//_alpha_change_i[fcount] += 0.5f * cubicterm * (_alpha_j_k + _alpha_i[fcount]) * dot(vel_ij,p_K);
				_alpha_change_i[fcount] += 0.5f * cubicterm * (_alpha_j_k + _alpha_i[fcount]) * dot(vel_ij, s_K);
				term1[fcount] = 0.5f * cubicterm * (_alpha_j_k - _alpha_i[fcount]) * dot(vel_ij, s_K);
				drift_vel_ij = _alpha_j_k * buf._drift_velocity[muloffset_j + fcount] + _alpha_i[fcount] * _drift_velocity_i[fcount];
				_alpha_change_i[fcount] += cubicterm * dot(drift_vel_ij, s_K);
				term2[fcount] = cubicterm * dot(drift_vel_ij, s_K);
			}
			if (i == simData.test_index)
			{
				//printf("index: %d, Search index:%d, term1: %f, %f, gradterm1: %f, %f, term2: %f, %f, gradterm2: %f, %f\n",i, j, term1[0], term1[1], gradterm1[0], gradterm1[1], term2[0], term2[1], gradterm2[0], gradterm2[1]);
				printf("///index: %d, Search index:%d, term1: %f, %f\n|||vel_ij:%f, %f, %f, cubicterm:%f\n```p_K: %f, %f, %f, dist: %f, %f, %f\n", i, j, term1[0], term1[1], vel_ij.x, vel_ij.z, vel_ij.z, cubicterm, cubicterm * p_K.x, cubicterm * p_K.y, cubicterm * p_K.z);
			}
		}
		j = buf.next_particle_index[j];
	}
	return _alpha_change_i;
}
__global__ void ComputeAlphaAdvancePeridynamics(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	if (buf._alpha_sum[i] < 0.01f)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	const float3 _position_i = buf._position[i];
	const float3 _velocity_i = buf._mix_velocity[i];

	int count;
	bool correction = false;

	// temporary container
	float3 _drift_velocity_i[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER], _alpha_change_i[MAX_PHASE_NUMBER];

	// build alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_i[fcount] = buf._alpha[muloffset_i + fcount];
		_alpha_change_i[fcount] = 0.0f;
		_drift_velocity_i[fcount] = buf._drift_velocity[muloffset_i + fcount];
	}

	// compute kernel grad value (divergence term)
	float* contribute_ptr;
	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = contributeAlphaChangePeridynamics(i, buf, _position_i, _velocity_i, _drift_velocity_i, _alpha_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			_alpha_change_i[fcount] += -contribute_ptr[fcount];
		}
	}
	if (isnan(_alpha_change_i[0]) || isnan(_alpha_change_i[1]))
	{
		printf("ComputeAlphaAdvance: alphachange is nan:%d\n", i);
		_alpha_change_i[0] = 0.0f;
		_alpha_change_i[1] = 0.0f;
	}
	// alpha advance
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_change_i[fcount] *= time_step;

		if (_alpha_change_i[fcount] < -0.99f)
		{
			_alpha_change_i[fcount] = -0.99f;
			count = fcount;
			correction = true;
		}
		buf._delta_alpha[muloffset_i + fcount] = _alpha_change_i[fcount];
		buf._alpha_advanced[muloffset_i + fcount] = _alpha_change_i[fcount] + _alpha_i[fcount];
		if (buf._alpha_advanced[muloffset_i + fcount] < 0.0f)
		{
			//printf("index:%d, alpha is nagetive:%f, %f\n", i, buf._alpha_advanced[muloffset_i], buf._alpha_advanced[muloffset_i + 1]);
			buf._alpha_advanced[muloffset_i + fcount] = 0.0f;
		}
	}
	if (correction)
	{
		if (count == 0)
		{
			if (_alpha_change_i[1] + _alpha_i[1] < 0.0f)
			{
				buf._alpha_advanced[muloffset_i] = 1.0f;
				buf._alpha_advanced[muloffset_i + 1] = 0.0f;
			}
			else
			{
				buf._alpha_advanced[muloffset_i] = 0.0f;
				buf._alpha_advanced[muloffset_i + 1] = 1.0f;
				printf("value:%f\n", _alpha_change_i[1] + _alpha_i[1]);
			}
		}
		else
		{
			if (_alpha_change_i[0] + _alpha_i[0] < 0.0f)
			{
				buf._alpha_advanced[muloffset_i] = 0.0f;
				buf._alpha_advanced[muloffset_i + 1] = 1.0f;
			}
			else
			{
				buf._alpha_advanced[muloffset_i] = 1.0f;
				buf._alpha_advanced[muloffset_i + 1] = 0.0f;
			}
		}
	}
	if (buf._noted[i])
	{
		printf("index: %d is noted, alpha_change: %f, %f, alpha: %f, %f\n", i, _alpha_change_i[0], _alpha_change_i[1], (_alpha_i[0] + _alpha_change_i[0]), (_alpha_i[1] + _alpha_change_i[1]));
	}
	if (i == simData.test_index)
	{
		printf("index: %d, alpha_change: %f, %f, alpha_advanced: %f, %f\n", i, _alpha_change_i[0], _alpha_change_i[1], (_alpha_i[0] + _alpha_change_i[0]), (_alpha_i[1] + _alpha_change_i[1]));
	}

	float alphasum = 0.0f;
	alphasum += buf._alpha_advanced[muloffset_i] + buf._alpha_advanced[muloffset_i + 1];
	if (alphasum < 0.01f)
	{
		printf("index: %d alphasum: %f, alpha: %f, %f, alpha_change:%f, %f\n", i, alphasum, _alpha_i[0], _alpha_i[1], _alpha_change_i[0], _alpha_change_i[1]);
		printf("index: %d pos:%f, %f, %f\n", i, _position_i.x, _position_i.y, _position_i.z);
	}

	if (buf._alpha_advanced[muloffset_i + 1] > 0.5f)
	{
		//printf("index %d betrays us!\n", i);
	}
	if (i % 1000 == 0)
	{
		//printf("index: %d, alpha: %f, %f, alpha_change: %f, %f\n", i, _alpha_i[0], _alpha_i[1], _alpha_change_i[0], _alpha_change_i[1]);
		//printTensor(i, buf._tensor_K[i]);
	}
}

// use lambda to correct negative alpha
__device__ float* contributeAlphaChangeIterPeridynamics(int i, bufList buf, float3 pos, float3 vel, float3* _drift_velocity_i, float* _alpha_i, int grid_index)
{
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _alpha_change_j[MAX_PHASE_NUMBER] = { 0.0f };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return _alpha_change_i;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float _alpha_j_k;
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	int containeroffset_i = simData.container_size * i;
	float term1[MAX_PHASE_NUMBER], term2[MAX_PHASE_NUMBER];
	float gradterm1[MAX_PHASE_NUMBER], gradterm2[MAX_PHASE_NUMBER];
	float lambda;
	float3 inv_K_mul_xi; // inv(K) * \xi
	float3 inv_F_mul_inv_K_mul_xi; // inv(F) * inv(K) * \xi
	float3 inv_F_mul_inv_K_mul_xi_j; // inv(F) * inv(K) * \xi for neighbor j
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float K_xi_x_j, K_xi_y_j, K_xi_z_j; // components of [inv(K) * \xi] for neighbor j
	float F_K_xi_x, F_K_xi_y, F_K_xi_z; // components of [inv(F) * inv(K) * \xi]
	float F_K_xi_x_j, F_K_xi_y_j, F_K_xi_z_j; // components of [inv(F) * inv(K) * \xi] for neighbor j
	float pairwise_factor; // for mass conservation
	float gamma_i = buf._rest_mass[i] / buf._mix_rest_density[i];
	float3 pos_ij, vel_ij, drift_vel_ij;
	tensor K = buf._tensor_K[i];
	tensor F = buf._tensor_F[i];
	tensor K_j, F_j;
	int neighbor_count_i = 0; // for neighbor value storing
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND || buf._type[j] == RIGID || buf._type[j] == AIR || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			float gamma_j = buf._rest_mass[j] / buf._mix_rest_density[j];
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			pairwise_factor = 2.0f * gamma_j / (gamma_i + gamma_j);
			//cubicterm *= -(_volume_i * _volume_j) / (_volume_i + _volume_j);
			cubicterm *= -_volume_j;
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);
			F_K_xi_x = F.tensor[0] * K_xi_x + F.tensor[1] * K_xi_y + F.tensor[2] * K_xi_z;
			F_K_xi_y = F.tensor[3] * K_xi_x + F.tensor[4] * K_xi_y + F.tensor[5] * K_xi_z;
			F_K_xi_z = F.tensor[6] * K_xi_x + F.tensor[7] * K_xi_y + F.tensor[8] * K_xi_z;
			inv_F_mul_inv_K_mul_xi = make_float3(F_K_xi_x, F_K_xi_y, F_K_xi_z);
			// j tensor
			K_j = buf._tensor_K[j];
			F_j = buf._tensor_F[j];
			K_xi_x_j = K_j.tensor[0] * dx + K_j.tensor[1] * dy + K_j.tensor[2] * dz;
			K_xi_y_j = K_j.tensor[3] * dx + K_j.tensor[4] * dy + K_j.tensor[5] * dz;
			K_xi_z_j = K_j.tensor[6] * dx + K_j.tensor[7] * dy + K_j.tensor[8] * dz;

			F_K_xi_x_j = F_j.tensor[0] * K_xi_x_j + F_j.tensor[1] * K_xi_y_j + F_j.tensor[2] * K_xi_z_j;
			F_K_xi_y_j = F_j.tensor[3] * K_xi_x_j + F_j.tensor[4] * K_xi_y_j + F_j.tensor[5] * K_xi_z_j;
			F_K_xi_z_j = F_j.tensor[6] * K_xi_x_j + F_j.tensor[7] * K_xi_y_j + F_j.tensor[8] * K_xi_z_j;
			inv_F_mul_inv_K_mul_xi_j = make_float3(F_K_xi_x_j, F_K_xi_y_j, F_K_xi_z_j);
			inv_F_mul_inv_K_mul_xi = 0.5f * (inv_F_mul_inv_K_mul_xi + inv_F_mul_inv_K_mul_xi_j);
			vel_ij = buf._mix_velocity[j] - vel;
			lambda = min(buf._lambda[i], buf._lambda[j]);
			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				_alpha_j_k = buf._alpha[muloffset_j + fcount];
				_alpha_change_i[fcount] += cubicterm * (_alpha_j_k - _alpha_i[fcount]) * dot(vel_ij, inv_F_mul_inv_K_mul_xi) * lambda;
				//_alpha_change_i[fcount] += 1.0f * cubicterm * (_alpha_j_k - _alpha_i[fcount]) * dot(vel_ij, s_K)*s;
				term1[fcount] = 1.0f * cubicterm * (_alpha_j_k - _alpha_i[fcount]) * dot(vel_ij, inv_F_mul_inv_K_mul_xi) * lambda;
				drift_vel_ij = _alpha_j_k * (buf._drift_velocity[muloffset_j + fcount]) + _alpha_i[fcount] * (_drift_velocity_i[fcount]);
				_alpha_change_i[fcount] += cubicterm * dot(drift_vel_ij, inv_F_mul_inv_K_mul_xi) * lambda;
				term2[fcount] = cubicterm * dot(drift_vel_ij, inv_F_mul_inv_K_mul_xi);
				//if (isnan(_alpha_change_i[fcount]))
				//	printf("j: %d, i: %d, vel_ij: %f, %f, %f, drift_vel_ij: %f, %f, %f, FK: %f, %f, %f\n", j, i,
				//		vel_ij.x, vel_ij.y, vel_ij.z, drift_vel_ij.x, drift_vel_ij.y, drift_vel_ij.z,
				//		inv_F_mul_inv_K_mul_xi.x, inv_F_mul_inv_K_mul_xi.y, inv_F_mul_inv_K_mul_xi.z);
				//if (lambda * (abs(term1[fcount] + term2[fcount])) > 10.0f)
					//printf("i: %d, j:%d, delta_alpha: %f, %f, drift_vel_ij: %f, %f, %f, v_ij_dot_F_K_xi: %f, alpha_j: %f, alpha_i: %f, cubicterm: %f\n", i, j, term1[fcount], term2[fcount], drift_vel_ij.x, drift_vel_ij.y, drift_vel_ij.z, dot(vel_ij, inv_F_mul_inv_K_mul_xi), _alpha_j_k, _alpha_i[fcount], cubicterm);
			}
			//if (i % 2000 == 0)
				//printf("i: %d, j: %d, pairwise_factor: %f\n", i, j, pairwise_factor);
			/*
			if (buf.particle_neighbor_number[i] < simData.container_size)
			{
				neighbor_count_i = buf.particle_neighbor_number[i];
				buf._container[containeroffset_i+ neighbor_count_i].index = j;
				for (int fcount = 0; fcount < simData.phase_number; fcount++)
					buf._container[containeroffset_i + neighbor_count_i].value[fcount] = (term1[fcount] + term2[fcount]) * lambda;
			}
			else
				printf("TEMP CONTAINER: index: %d, too many neighbors\n", i);
			buf.particle_neighbor_number[i] += 1;*/
		}
		j = buf.next_particle_index[j];
	}
	return _alpha_change_i;
}
__global__ void ComputeAlphaTransportIter(bufList buf, int numParticles, float time_step, float factor)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)// || buf._mix[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID || buf._type[i] == AIR)
		return;

	//if (buf._alpha_sum[i] < 0.01f)
		//return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const int muloffset_i = i * MAX_PHASE_NUMBER;
	const int containeroffset_i = i * simData.container_size;

	const float3 _position_i = buf._position[i];
	const float3 _velocity_i = buf._mix_velocity[i];

	bool OutputNeighborImfo = false;


	// temporary container
	float3 _drift_velocity_i[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER], _alpha_change_i[MAX_PHASE_NUMBER];

	// build alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_i[fcount] = buf._alpha[muloffset_i + fcount];
		_alpha_change_i[fcount] = 0.0f;
		_drift_velocity_i[fcount] = buf._drift_velocity[muloffset_i + fcount];
	}

	// compute kernel grad value (divergence term)
	float* contribute_ptr;
	//buf.particle_neighbor_number[i] = 0;
	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = contributeAlphaChangeIterPeridynamics(i, buf, _position_i, _velocity_i, _drift_velocity_i, _alpha_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			_alpha_change_i[fcount] += -contribute_ptr[fcount];
		}
	}
	if (isnan(_alpha_change_i[0]) || isnan(_alpha_change_i[1]))
	{
		//if(i%1000==0)
		//	printf("ComputeAlphaAdvance: alphachange is nan:%d\n", i);
		_alpha_change_i[0] = 0.0f;
		_alpha_change_i[1] = 0.0f;
		_alpha_change_i[2] = 0.0f;
		_alpha_change_i[3] = 0.0f;
		_alpha_change_i[4] = 0.0f;
	}
	// alpha advance
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_change_i[fcount] *= time_step * factor;
		if (abs(_alpha_change_i[fcount]) > 1.0f)
		{
			//printf("Alpha_advance: index: %d, delta_alpha: %f, phase: %d, lambda: %f\n", i, _alpha_change_i[fcount], fcount, buf._lambda[i]);
			//OutputNeighborImfo = true;
		}
		buf._delta_alpha[muloffset_i + fcount] = _alpha_change_i[fcount];
		buf._alpha_advanced[muloffset_i + fcount] = _alpha_change_i[fcount] + _alpha_i[fcount];
		//if (_alpha_change_i[fcount] + _alpha_i[fcount] < -0.001f || (buf._lambda[i]>0.1f && buf._lambda[i]<0.9f))
			//printf("alpha < 0, index: %d, vel: %f, %f, %f, phase: %d, delta_alpha: %f, alpha: %f, lambda: %f\n", i, _velocity_i.x, _velocity_i.y, _velocity_i.z, fcount, _alpha_change_i[fcount], _alpha_change_i[fcount] + _alpha_i[fcount], buf._lambda[i]);
	}
	//if (i % 1000 == 0)
		//printf("i: %d, alpha: %f, %f\n",i, buf._alpha_advanced[muloffset_i], buf._alpha_advanced[muloffset_i + 1]);
	if (OutputNeighborImfo)
	{
		neighbor_value_container container;
		for (int neighbor_index = 0; neighbor_index < buf.particle_neighbor_number[i]; neighbor_index++)
		{
			if (neighbor_index >= simData.container_size)
				break;
			container = buf._container[containeroffset_i + neighbor_index];
			printf("index: %d, neighbor index: %d, delta_alpha: %f, %f, neighbor_num: %d\n", i, container.index, container.value[0], container.value[1], buf.particle_neighbor_number[i]);
		}
	}
}

__global__ void UpdateLambda(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)// || buf._mix[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID || buf._type[i] == AIR)
		return;

	const int muloffset_i = i * MAX_PHASE_NUMBER;
	float _alpha_i[MAX_PHASE_NUMBER], _alpha_change_i[MAX_PHASE_NUMBER], _alpha_advanced_i[MAX_PHASE_NUMBER];
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		_alpha_i[fcount] = buf._alpha[muloffset_i + fcount];
		_alpha_change_i[fcount] = buf._delta_alpha[muloffset_i + fcount];
		_alpha_advanced_i[fcount] = buf._alpha_advanced[muloffset_i + fcount];
	}
	/**/
	float temp_coef = 0.0f, coef = 10000.0f;
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		if (_alpha_advanced_i[fcount] < -0.001f)
			break;
		if (fcount == MAX_PHASE_NUMBER - 1)
			return;
	}
	bool _negative[MAX_PHASE_NUMBER];
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		if (_alpha_advanced_i[fcount] < -0.001f)
		{
			_negative[fcount] = true;
			temp_coef = _alpha_i[fcount] / (-_alpha_change_i[fcount]);
			coef = min(coef, temp_coef);
		}
		else
			_negative[fcount] = false;
	}
	if (coef < 0.0f)
		coef = 0.0f;
	buf._lambda[i] *= coef;
	/*
	if (_alpha_advanced_i[0] >= 0.0f && _alpha_advanced_i[1] >= 0.0f)
	{
		return;
	}
	else
	{
		float coef;
		if (_alpha_advanced_i[0] < 0.0f && _alpha_advanced_i[1] >= 0.0f)
		{
			coef = _alpha_i[0] / (-_alpha_change_i[0]);
		}
		if (_alpha_advanced_i[0] > -0.0f && _alpha_advanced_i[1] < 0.0f)
		{
			coef = _alpha_i[1] / (-_alpha_change_i[1]);
		}
		if (_alpha_advanced_i[0] < 0.0f && _alpha_advanced_i[1] < 0.0f)
		{
			coef = min(_alpha_i[0] / (-_alpha_change_i[0]), _alpha_i[1] / (-_alpha_change_i[1]));
		}
		buf._lambda[i] *= coef;
	}*/
	//if(i%500==0)
		//printf("i: %d, lambda: %f\n",i, buf._lambda[i]);
}

__global__ void AlphaCorrection(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const int muloffset_i = i * MAX_PHASE_NUMBER;
	float _alpha_i[MAX_PHASE_NUMBER], _alpha_change_i[MAX_PHASE_NUMBER], _rel_change[MAX_PHASE_NUMBER];
	float epsilon = 0.01f;
	float fmass = simData.phase_mass[0], smass = simData.phase_mass[1];

	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		_alpha_i[fcount] = buf._alpha[muloffset_i + fcount];
		_alpha_change_i[fcount] = buf._delta_alpha[muloffset_i + fcount];
		_rel_change[fcount] = _alpha_change_i[fcount] / (_alpha_i[fcount] + epsilon);
	}
	float multi_alpha = abs(_alpha_change_i[0] * _alpha_change_i[1]);

	//float value = sqrt(multi_alpha) / (1.0f + _alpha_change_i[0] + _alpha_change_i[1]);
	float value = abs((_alpha_change_i[0] * fmass + _alpha_change_i[1] * smass) / (fmass + smass));
	if (_rel_change[0] > _rel_change[1])
	{
		buf._alpha_advanced[muloffset_i] = _alpha_i[0] + value;
		buf._alpha_advanced[muloffset_i + 1] = _alpha_i[1] - value;
	}
	else
	{
		buf._alpha_advanced[muloffset_i] = _alpha_i[0] - value;
		buf._alpha_advanced[muloffset_i + 1] = _alpha_i[1] + value;
	}
	//if (i % 1000 == 0)
		//printf("index: %d, delta_alpha:%f, %f, alpha:%f, %f, rel_change:%f, %f, value:%f\n", i, _alpha_change_i[0], _alpha_change_i[1], _alpha_i[0], _alpha_i[1], _alpha_change_i[0]/_alpha_i[0], _alpha_change_i[1] / _alpha_i[1], value);
}

// compute force vector
__global__ void ComputeTDMPeridynamics(bufList buf, int numParticles, float time_step, float surface_factor)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	if (buf._alpha_sum[i] < 0.01f)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const float3 _position_i = buf._position[i];
	const float3 _velocity_i = buf._mix_velocity[i];
	const float _mix_rest_density_i = buf._mix_rest_density[i];
	const float _mix_pressure_i = buf._mix_pressure[i];
	const float _inv_mass_i = 1.0f / buf._rest_mass[i];
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const int muloffset_i = i * MAX_PHASE_NUMBER;
	const float pVol = 1.0f / _mix_rest_density_i; // inv(mix_rest_density)
	const float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	tensor K = buf._tensor_K[i];

	const int phase = buf._phase[i];
	const float mass = simData.phase_mass[phase];

	if (_mix_rest_density_i < 0.01f)
	{
		//printf("ComputeTDM: index %d density equals to zero\n", i);
		return;
	}

	// temporary container
	float3 _force = make_float3(0.0f, 0.0f, 0.0f);
	float3 _pressure_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _visc_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _phase_part = make_float3(0.0f, 0.0f, 0.0f);
	float3 _drift_velocity_i[MAX_PHASE_NUMBER];
	float _alpha_i[MAX_PHASE_NUMBER];

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_i[fcount] = buf._alpha_advanced[muloffset_i + fcount];
		_drift_velocity_i[fcount] = buf._drift_velocity[muloffset_i + fcount];
	}

	// kernel parameters
	float q;
	float dist, dist_square;
	float cubicterm;
	float _alpha_j_k;
	float3 inv_K_mul_xi; // inv(K) * \xi
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float xi_dot_inv_K_mul_xi; // \xi * inv(K) * \xi
	float _volume_j;
	int muloffset_j;
	float3 phasesum = make_float3(0.0f, 0.0f, 0.0f);
	float3 pressuregrad = make_float3(0.0f, 0.0f, 0.0f);
	float3 TM = make_float3(0.0f, 0.0f, 0.0f);
	float3 pos_ij, vel_ij;

	float3 pressure_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 pressure_force_grad = make_float3(0.0f, 0.0f, 0.0f);
	float3 visc_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 phase_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 surface_tension = make_float3(0.0f, 0.0f, 0.0f);

	// compute kernel grad value (TDM)
	for (int cell = 0; cell < 27; cell++)
	{
		const int SearchIndex = iCellIndex + simData.grid_search_offset[cell];

		if (SearchIndex <0 || SearchIndex >simData.grid_number - 1)
		{
			continue;
		}

		uint j = buf.grid_particle_table[SearchIndex];

		while (j != GRID_UNDEF)
		{
			if (j == i || buf.bubble[j] == true)
			{
				j = buf.next_particle_index[j];
				continue;
			}

			muloffset_j = j * MAX_PHASE_NUMBER;
			pos_ij = _position_i - buf._position[j];
			const float dx = pos_ij.x;
			const float dy = pos_ij.y;
			const float dz = pos_ij.z;
			dist_square = dx * dx + dy * dy + dz * dz;

			if (dist_square < smooth_radius_square)
			{
				const float nVol = mass / buf._mix_density[j];
				dist = sqrt(dist_square);
				if (dist < 0.00001f)
				{
					dist = 0.00001f;
				}
				q = dist / smooth_radius;
				if (q <= 0.5f)
				{
					cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
				}
				else
				{
					cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
				}
				K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
				K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
				K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
				xi_dot_inv_K_mul_xi = dx * K_xi_x + dy * K_xi_y + dz * K_xi_z;
				inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);
				if (buf._type[j] == FLUID || buf._type[j] == AIR)
				{
					_volume_j = buf._rest_mass[j] / buf._mix_density[j];
					//cubicterm *= (_volume_i * _volume_j) / (_volume_i + _volume_j);
					surface_tension = -buf._rest_mass[j] * pos_ij * cubicterm;
					cubicterm *= _volume_j;
					vel_ij = buf._mix_velocity[j] - _velocity_i;
					if (simData._explicit)
						pressure_force = 0.5f * cubicterm * (_mix_pressure_i + buf._mix_pressure[j]) * inv_K_mul_xi * pVol;
					visc_force = 2.0f * cubicterm * (buf._viscosity[i] + buf._viscosity[j]) * vel_ij * pVol * xi_dot_inv_K_mul_xi / (dist * dist);
					//if (max(buf._viscosity[i], buf._viscosity[j]) / min(buf._viscosity[i], buf._viscosity[j]) > 2.0f)
					//	visc_force *= 0.3f;
				}
				else
				{
					if (buf._active[j] == true)
					{
						vel_ij = buf._mix_velocity[j] - _velocity_i;
						cubicterm *= buf._bound_phi[j] * buf._mix_rest_density[i] / (buf._mix_density[i]);
						pressure_force = cubicterm * pVol * _mix_pressure_i * inv_K_mul_xi;
						visc_force = 2.0f * cubicterm * simData.viscosity * vel_ij * pVol * xi_dot_inv_K_mul_xi / (dist * dist);
					}
					else
					{
						pressure_force = make_float3(0.0f, 0.0f, 0.0f);
						visc_force = make_float3(0.0f, 0.0f, 0.0f);
					}
				}
				_force += (pressure_force + visc_force + surface_factor * _inv_mass_i * surface_tension);
				_pressure_part += pressure_force;
				_visc_part += visc_force;
			}
			j = buf.next_particle_index[j];
		}
	}
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		phasesum += simData.phase_density[fcount] * pVol * (buf._delta_alpha[muloffset_i + fcount] / time_step) * (_drift_velocity_i[fcount] + _velocity_i);
	}
	//phasesum -= _velocity_i * buf._SdV[i];
	phase_force = phasesum;
	//_force += phase_force;
	buf._force[i] = _force;
	buf._pressure_force[i] = _pressure_part;
	buf._visc_force[i] = _visc_part;
	buf._phase_force[i] = _phase_part;
	//if (buf._noted[i])
	//{
	//	printf("index: %d is noted, pressure_force: %f, %f, %f\nTDM: %f, %f, %f\n", i, _pressure_part.x, _pressure_part.y, _pressure_part.z, _phase_part.x, _phase_part.y, _phase_part.z);
	//}
}

__global__ void ComputeMFSourceTermPeridynamics(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];
	float3 _pre_velocity_i = buf._pre_velocity[i];
	const float _volume_i = buf._volume[i];
	float div = 0.0f;
	for (int cell = 0; cell < 27; cell++)
	{
		//div += contributeDivergencePeridynamics(i, buf, _position_i, _pre_velocity_i, _volume_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	buf._source_term[i] = (1.0f - buf._rest_volume[i] / _volume_i) + div * time_step;
}

__global__ void ComputeMFDiagElementPeridynamics(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];
	float3 _pre_velocity_i = buf._pre_velocity[i];
	const float _volume_i = buf._volume[i];
	const float _mass_i = buf._rest_mass[i];
	float diag_term = 0.0f;
	float sum = 0.0f;
	for (int cell = 0; cell < 27; cell++)
	{
		//sum += contributeDiagSumPeridynamics(i, buf, _position_i, _mass_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	diag_term = -sum * _volume_i / _mass_i * time_step * time_step;
	if (abs(diag_term) < 0.0000001f)
	{
		diag_term = -0.0000001f;
	}
	buf._diag_term[i] = diag_term;
}

__device__ float3 contributeMFAccelerationPeridynamics(int i, bufList buf, float3 pos, float mix_pressure, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return make_float3(0.0f, 0.0f, 0.0f);
	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	float dist, dist_square;
	float q, cubicterm;
	float3 pos_ij;
	float3 p_K;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	tensor K = buf._tensor_K[i];
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			p_K = make_float3(pos_ij.x / K.tensor[0], pos_ij.y / K.tensor[4], pos_ij.z / K.tensor[8]);
			result += -buf._volume[j] * cubicterm * (mix_pressure + buf._mix_pressure[j]) * p_K;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeMFPressureAccelPeridynamics(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
	{
		buf._acceleration_p[i] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];;
	float _mix_pressure_i = buf._mix_pressure[i];
	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	for (int cell = 0; cell < 27; cell++)
	{
		//sum += contributeMFAcceleration(i, buf, _position_i, _mix_pressure_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	buf._acceleration_p[i] = -(buf._volume[i] / buf._rest_mass[i]) * sum;
}

__global__ void MFPressureUpdatePeridynamics(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 _position_i = buf._position[i];
	float3 _acceleration_p_i = buf._acceleration_p[i];
	float _mix_pressure_i = buf._mix_pressure[i];
	float Ap_sum = 0.0f;
	for (int cell = 0; cell < 27; cell++)
	{
		//Ap_sum += contributeAccelPSumPeridynamics(i, buf, _position_i, _acceleration_p_i, time_step, iCellIndex + simData.grid_search_offset[cell]);
	}
	if (buf._diag_term[i] < -0.0000001f)
	{
		buf._mix_pressure[i] = max((_mix_pressure_i + simData.omega * (buf._source_term[i] - Ap_sum) / (buf._diag_term[i])), 0.0f);
	}
	buf._residual[i] = abs(Ap_sum - buf._source_term[i]);
}

__global__ void ComputeCorrectionVolume(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	// temporary container
	float _alphasum = 0.0f;
	float _mix_rest_density = 0.0f;
	float _mix_viscosity = 0.0f;
	float _rest_mass = 0.0f;

	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _temp_alpha_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;
	const float gamma = 7.0f;

	// alpha correction
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		//if (buf._alpha_advanced[muloffset_i + fcount] < 0.01f)
			//buf._alpha_advanced[muloffset_i + fcount] = 0.0f;

		_alphasum += buf._alpha_advanced[muloffset_i + fcount];
	}

	if (isnan(_alphasum))
	{
		//if (i % 1000 == 0)
		//	printf("ComputeCorrection: index %d alphasum isnan\n", i);
	}

	if (_alphasum < 0.01f)
	{
		//if (i % 1000 == 0)
		//	printf("ComputeCorrection: index %d alphasum equals to zero, type: %d\n", i, buf._type[i]);
		buf._noted[i] = true;
	}
	/*
	// banlanced alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		// alpha banlance
		_alpha_change_i[fcount] = buf._delta_alpha[muloffset_i + fcount];
		buf._alpha_advanced[muloffset_i + fcount] = buf._alpha_advanced[muloffset_i + fcount] / _alphasum;
		buf._delta_alpha[muloffset_i + fcount] = buf._alpha_advanced[muloffset_i + fcount] - buf._alpha[muloffset_i + fcount];
		_mix_density += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount];
		//_delta_density += _alpha_change_i[fcount] * simData.phase_density[fcount];
		_mix_viscosity += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_visc[fcount];
		//_delta_mass += buf._delta_alpha[muloffset_i + fcount] * simData.phase_mass[fcount];
		//_delta_mass += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_mass[fcount];
		_delta_mass += _volume * buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount];
	}
	//_mix_density = _delta_density + _mix_rest_density_i;*/

	float sum_delta_mass_k = 0.0f;
	float sum_mass_k = 0.0f;
	// raw alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_temp_alpha_i[fcount] = buf._alpha_advanced[muloffset_i + fcount] / _alphasum; // for pressure computing
		_alpha_change_i[fcount] = _temp_alpha_i[fcount] - buf._alpha[muloffset_i + fcount];

		_mix_rest_density += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount]; // _mix_rest_density = Sum_k( alpha_i_k * rho_k)
		_mix_viscosity += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_visc[fcount];
		_rest_mass += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount] * _volume; // _rest_mass = Sum_k( alpha_i_k * rho_k * V)

		//buf._alpha_advanced[muloffset_i + fcount] = _temp_alpha_i[fcount];
		//buf._delta_alpha[muloffset_i + fcount] = _alpha_change_i[fcount];
		//if (abs(buf._delta_alpha[muloffset_i + fcount]) > 1.0f)
			//printf("index: %d, delta_alpha: %f\n", i, buf._delta_alpha[muloffset_i + fcount]);
		buf._delta_mass_k[muloffset_i + fcount] = buf._delta_alpha[muloffset_i + fcount] * simData.phase_density[fcount] * _volume;
		sum_delta_mass_k += buf._delta_mass_k[muloffset_i + fcount];
		buf._rest_mass_k[muloffset_i + fcount] += buf._delta_mass_k[muloffset_i + fcount];
	}

	/*
	// pressure correction
	float _delta_pressure = 0.0f;
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		float relative_density = pow(_mix_density_i / _mix_rest_density, gamma);
		//_delta_pressure -= simData.gas_constant * simData.phase_density[fcount] * buf._delta_alpha[muloffset_i + fcount];
		_delta_pressure -= simData.gas_constant * simData.phase_density[fcount] * ((gamma-1.0f) * relative_density + 1.0f) * _alpha_change_i[fcount] / gamma;
	}
	if (simData._explicit)
		buf._mix_pressure[i] += _delta_pressure;*/

	// mu && rho_m correction
	buf._mix_rest_density[i] = _mix_rest_density;
	buf._viscosity[i] = _mix_viscosity;
	//buf._rest_mass[i] = (_rest_mass + 0.0f*_mass_i * time_step * buf._SdV[i]);
	buf._rest_mass[i] = _mass_i + sum_delta_mass_k;
	buf._delta_mass[i] = _rest_mass - _mass_i;
	//if (sum_delta_mass_k > 1.0f)
	//	printf("index: %d, delta_mass: %f, sum_delta_alpha_k: %f, %f\n", i, sum_delta_mass_k, buf._delta_alpha[muloffset_i + 0], buf._delta_alpha[muloffset_i + 1]);
}

__global__ void AlphaBanlanceUsingMassFraction(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;

	// temporary container
	float Sum_mass_fraction_divide_rho_k = 0.0f;
	float Sum_mass_fraction = 0.0f;
	float Sum_alpha = 0.0f;

	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _mass_fraction_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _alpha_banlanced[MAX_PHASE_NUMBER] = { 0.0f };
	float _alpha[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;

	float particle_diameter;


	// re-compute mass fraction
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		//_mass_fraction_i[fcount] = buf._rest_mass_k[muloffset_i + fcount] / _mass_i;
		_mass_fraction_i[fcount] = buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount] * _volume / _mass_i;  // m_k = alpha_k * rho_k * volume
		Sum_mass_fraction_divide_rho_k += _mass_fraction_i[fcount] / simData.phase_density[fcount];
		Sum_mass_fraction += _mass_fraction_i[fcount];
		buf._mass_fraction[muloffset_i + fcount] = _mass_fraction_i[fcount];
	}
	//if (i % 1000==0)
		//printf("index: %d, Sum_c: %.15f\n", i, Sum_mass_fraction);

	// re-compute alpha
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha[fcount] = buf._alpha_advanced[muloffset_i + fcount];
		_alpha_banlanced[fcount] = (_mass_fraction_i[fcount] / simData.phase_density[fcount]) / Sum_mass_fraction_divide_rho_k;
		buf._alpha_advanced[muloffset_i + fcount] = _alpha_banlanced[fcount];
		//buf._delta_alpha[muloffset_i + fcount] = _alpha_banlanced[fcount] - _alpha[fcount];
		Sum_alpha += _alpha_banlanced[fcount];
		//if (_alpha_banlanced[fcount] > 1.1f || _alpha_banlanced[fcount] < -0.1f)
		//	printf("index: %d, banlanced_alpha: %f\n", i, _alpha_banlanced[fcount]);
	}
	//if (i % 1000 == 0)
		//printf("index: %d, Sum_alpha: %.15f\n", i, Sum_alpha);
	particle_diameter = pow(_volume, 1.0f / 3.0f);

	// V correction
	buf._eff_V[i] = buf._rest_mass[i] / buf._mix_density[i];
	buf._particle_radius[i] = 0.5f * particle_diameter;
	//buf._smooth_radius[i] = 2.0f * particle_diameter;
}

// particle bound //

__device__ float contributeBoundKernel(int i, bufList buf, float3 pos, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	//float smooth_radius = buf._smooth_radius[i];
	//float smooth_radius_square;// = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			result += cubicterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputePhiParticleBound(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == FLUID || buf._active[i] == false || buf.bubble[i] == true)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float result = 0.0f;
	float3 _position_i = buf._position[i];
	for (int cell = 0; cell < 27; cell++)
	{
		result += contributeBoundKernel(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	// 1.4f means banlanced value for its lack of neighbors
	buf._bound_phi[i] = (1.4f / result);
}

__device__ float contributeMFDensityParticleBound(int i, bufList buf, float3 pos, float& sum_b, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			if (buf._type[j] == FLUID || buf._type[j] == AIR)
			{
				result += cubicterm * buf._rest_mass[i];
				buf._test[j] = true;
			}
			else
				sum_b += cubicterm * buf._mix_rest_density[i] * buf._bound_phi[j];
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeMFDensityParticleBound(bufList buf, int numParticles, bool transferAlpha)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	const float gamma = 2.0f;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID ||buf.bubble[i] == true)
		return;
	if (buf._active[i] == false)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum_b = 0.0f;
	float sum_f = 0.0f;
	float3 _position_i = buf._position[i];
	const float _mass_i = buf._rest_mass[i];
	//const float smooth_radius = buf._smooth_radius[i];
	//const float self_density = _mass_i * simData.kernel_self;
	const float self_density = _mass_i * simData.CubicSplineKern;
	for (int cell = 0; cell < 27; cell++)
	{
		sum_f += contributeMFDensityParticleBound(i, buf, _position_i, sum_b, iCellIndex + simData.grid_search_offset[cell]);
	}
	if (buf._type[i] == FLUID || buf._type[i] == AIR)
		sum_f += self_density;
	buf._mix_density[i] = (sum_b + sum_f);
	// wcsph
	if (simData._explicit && (buf._type[i] == FLUID || buf._type[i] == AIR))
	{
		float relative_density = pow(buf._mix_density[i] / buf._mix_rest_density[i], gamma);
		buf._mix_pressure[i] = max(0.0f, ((relative_density - 1.0f) * simData.gas_constant * buf._mix_rest_density[i])/gamma);
	}
	buf._noted[i] = false;
	//if (buf._position[i].x * simData.simscale > 26.0f && transferAlpha == true)
	//	buf._mix[i] = true;
}

__device__ float contributeCoef(int i, bufList buf, float3 pos, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			cubicterm *= buf._rest_mass[j] / buf._mix_density[j];
			result += cubicterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ContributePressure(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum = 0.0f;
	float3 _position_i = buf._position[i];
	float _density_i = buf._mix_density[i];
	const float _rest_density = buf._mix_rest_density[i];
	const float _mass_i = buf._rest_mass[i];
	const float self_density = _mass_i * simData.kernel_self;
	for (int cell = 0; cell < 27; cell++)
	{
		sum += contributeCoef(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	sum += simData.kernel_self * (_mass_i / _density_i);
	_density_i /= sum;
	//buf._mix_pressure[i] = max(0.0f, (_density_i - buf._mix_rest_density[i]) * simData.gas_constant);
	// wcsph
	if (simData._explicit && buf._type[i] == FLUID)
	{
		float relative_density = pow(_density_i / _rest_density, 2);
		buf._mix_pressure[i] = max(0.0f, ((relative_density - 1.0f) * simData.gas_constant * _rest_density));
	}
	buf._mix_density[i] = _density_i;
	/**/
	if (i % 1000 == 0)
	{
		printf("index: %d, type: %d, density: %f, rest_density: %f, sum: %f\n", i, buf._type[i], _density_i, _rest_density, sum);
	}
}

__device__ float contributeBoundPressure(int i, bufList buf, float3 pos, float& kernelsum, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == BOUND)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			cubicterm *= buf._rest_mass[j] / buf._mix_density[j];
			result += cubicterm * buf._mix_pressure[j];
			kernelsum += cubicterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeBoundPressure(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == FLUID)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum = 0.0f;
	float kernelsum = 0.0f;
	float3 _position_i = buf._position[i];;
	for (int cell = 0; cell < 27; cell++)
	{
		sum += contributeBoundPressure(i, buf, _position_i, kernelsum, iCellIndex + simData.grid_search_offset[cell]);
	}
	kernelsum += simData.kernel_self * buf._rest_mass[i] / buf._mix_density[i];
	buf._mix_pressure[i] = sum / kernelsum;
	if (i % 100 == 0)
	{
		//printf("index: %d, pressure: %f\n", i, sum / kernelsum);
	}
}

__global__ void AdvanceParticleBound(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND)
		return;
	float3 position = buf._position[i];
	float3 velocity = buf._mix_velocity[i];
	float3 acceleration = buf._force[i] + simData.gravity;
	float3 pressure_force = buf._pressure_force[i];
	float3 visc_force = buf._visc_force[i];
	float3 phase_force = buf._phase_force[i];

	if (isnan(velocity.y))
	{
		velocity = make_float3(0.0f, 0.0f, 0.0f);
		printf("index:%d, velocity is nan\n", i);
	}

	if (isnan(acceleration.y))
	{
		acceleration = make_float3(0.0f, 0.0f, 0.0f);
		if (i % 1000 == 0)
			printf("index:%d, acceleration is nan\n", i);
	}

	limitHandlingCUDA(i, &acceleration, &velocity, &pressure_force, &visc_force, &phase_force);
	velocity += acceleration * time_step;
	position += velocity * time_step;

	buf._position[i] = position;
	buf._mix_velocity[i] = velocity;
}

// generate mf particles
__global__ void GenerateParticles(bufList buf, int numParticles, int begin, int N, float time_step, int Generate_pos, float3 start_point, float GenFrameRate)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	float coef_x = start_point.x;
	float coef_y = start_point.y;
	float coef_z = start_point.z;/*
	float preset_alpha[MAX_PHASE_NUMBER] = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
	switch (Generate_pos)
	{
	case 0:
		preset_alpha[2] = 0.0f;
		preset_alpha[0] = 1.0f;
	case 1:
		preset_alpha[2] = 0.0f;
		preset_alpha[1] = 1.0f;
		break;
	case 2:
		break;
	case 3:
		preset_alpha[2] = 0.0f;
		preset_alpha[3] = 1.0f;
		break;
	case 4:
		preset_alpha[2] = 0.0f;
		preset_alpha[4] = 1.0f;
		break;
	case 5:
		preset_alpha[2] = 0.0f;
		preset_alpha[2] = 0.5f;
		preset_alpha[4] = 0.5f;
	default:
		break;
	}
	const float dr = 2.0f * simData.particle_radius;
	int ix = 0.0f, iy = 0.0f, iz = 0.0f;
	if ((i < begin + N * N) && i >= begin)
	{
		ix = (i - begin) % N;
		iz = (i - begin) / N;
		const float x = (coef_x * simData.boundary_max.x + (1.0f - coef_x) * simData.boundary_min.x);
		const float y = (coef_y * simData.boundary_max.y + (1.0f - coef_y) * simData.boundary_min.y);
		const float z = (coef_z * simData.boundary_max.z + (1.0f - coef_z) * simData.boundary_min.z);
		float3 pos = make_float3(x + dr * float(ix), y + dr * float(iy), z + dr * float(iz));
		float vel = 2.0f * simData.particle_radius / (time_step * GenFrameRate);
		float rest_mass = 0.0f, rest_density = 0.0f;
		buf._active[i] = true;
		buf._position[i] = pos;
		buf._mix_velocity[i] = make_float3(0.0f, vel, 0.0f);
		buf._acceleration[i] = simData.gravity;
		for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
		{
			rest_mass += preset_alpha[fcount] * simData.phase_mass[fcount];
			rest_density += preset_alpha[fcount] * simData.phase_density[fcount];
			buf._alpha_advanced[i * MAX_PHASE_NUMBER + fcount] = preset_alpha[fcount];
		}
		buf._rest_mass[i] = rest_mass;
		buf._mix_rest_density[i] = rest_density;
		buf._particle_radius[i] = simData.particle_radius;
		buf._phase[i] = 0;
		buf._type[i] = FLUID;
		buf._render[i] = true;
	}*/
}

__global__ void ChemicalReaction(bufList buf, int numParticles, float ReactionSpeed)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;
	const int react_phase1 = 3;
	const int react_phase2 = 4;
	const int generate_phase = 1;

	// temporary container
	float _mix_rest_density = 0.0f;
	float _mix_viscosity = 0.0f;
	float _rest_mass = 0.0f;
	float Sum_mass_fraction_divide_rho_k = 0.0f;
	float Sum_mass_fraction = 0.0f;

	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _alpha_change_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _temp_alpha_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _mass_fraction_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _alpha_banlanced[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;

	/*const float factor = 28.9f;
	const float reaction_alpha = -buf._alpha_advanced[muloffset_i + 1] * buf._alpha_advanced[muloffset_i + 2] * factor;

	buf._alpha_advanced[muloffset_i + 1] += reaction_alpha;
	buf._alpha_advanced[muloffset_i + 2] += reaction_alpha;
	buf._alpha_advanced[muloffset_i + 4] += -2.0f * reaction_alpha;*/

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		//_mass_fraction_i[fcount] = buf._rest_mass_k[muloffset_i + fcount] / _mass_i;
		_mass_fraction_i[fcount] = buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount] * _volume / _mass_i;  // m_k = alpha_k * rho_k * volume
	}

	float coef_gamma = 0.000001f * ReactionSpeed;
	float coef_lamda = 100.0f;

	float reaction_mass_fraction = -coef_gamma * buf._alpha_advanced[muloffset_i + react_phase1] * buf._alpha_advanced[muloffset_i + react_phase2] * simData.phase_density[react_phase1] * simData.phase_density[react_phase2] /
		(coef_lamda + buf._alpha_advanced[muloffset_i + generate_phase] * simData.phase_density[generate_phase]);

	_mass_fraction_i[react_phase1] += reaction_mass_fraction;
	_mass_fraction_i[react_phase2] += reaction_mass_fraction;
	_mass_fraction_i[generate_phase] += -2.0f * reaction_mass_fraction;

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		Sum_mass_fraction_divide_rho_k += _mass_fraction_i[fcount] / simData.phase_density[fcount];
		Sum_mass_fraction += _mass_fraction_i[fcount];
	}

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_alpha_banlanced[fcount] = (_mass_fraction_i[fcount] / simData.phase_density[fcount]) / Sum_mass_fraction_divide_rho_k;
		buf._alpha_advanced[muloffset_i + fcount] = _alpha_banlanced[fcount];
	}

	if (buf._alpha_advanced[muloffset_i + react_phase1] < 0.0f)
		buf._alpha_advanced[muloffset_i + react_phase1] = 0.0f;
	if (buf._alpha_advanced[muloffset_i + react_phase2] < 0.0f)
		buf._alpha_advanced[muloffset_i + react_phase2] = 0.0f;
	if (buf._alpha_advanced[muloffset_i + generate_phase] > 1.0f)
	{
		buf._alpha_advanced[muloffset_i + generate_phase] = 1.0f;
	}

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_mix_rest_density += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_density[fcount]; // _mix_rest_density = Sum_k( alpha_i_k * rho_k)
		_mix_viscosity += buf._alpha_advanced[muloffset_i + fcount] * simData.phase_visc[fcount];
	}

	// mu && rho_m correction
	buf._mix_rest_density[i] = _mix_rest_density;
	buf._viscosity[i] = _mix_viscosity;
}

__global__ void ConcentrationDecay(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	const float factor = 0.99f;
	buf._concentration[i] *= factor;
	if (i % 1000 == 0)
		printf("concentration :%f\n", buf._concentration[i]);
}

// rigid body //

__global__ void UpdateRigidBody(bufList buf, int numParticles, float time_step, float omega)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i]!=RIGID || buf._rotate[i] != true)
		return;
	float3 position = buf._position[i];
	float3 velocity = buf._mix_velocity[i];
	float3 center = buf._center[i];
	float3 rel_pos = position - center;
	const float radius = sqrt(rel_pos.x * rel_pos.x + rel_pos.z * rel_pos.z);
	const float d_theta = omega * time_step;
	const float cos_d_theta = cos(d_theta);
	const float sin_d_theta = sin(d_theta);
	const float cos_theta = rel_pos.x / radius;
	const float sin_theta = rel_pos.z / radius;
	const float sin_theta_plus_d_theta = sin_theta * cos_d_theta + cos_theta * sin_d_theta;
	const float cos_theta_plus_d_theta = cos_theta * cos_d_theta - sin_theta * sin_d_theta;

	velocity = make_float3(-omega * radius * sin_theta_plus_d_theta, 0.0f, omega * radius * cos_theta_plus_d_theta);
	position = center + make_float3(radius * cos_theta_plus_d_theta, rel_pos.y, radius * sin_theta_plus_d_theta);

	buf._position[i] = position;
	buf._mix_velocity[i] = velocity;
}

__global__ void UpdateRigidBodyDrift(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == FLUID)
		return;
	if (buf._explosion[i] != 1)
		return;
	float3 position = buf._position[i];
	float3 velocity = buf._mix_velocity[i];
	float3 center = buf._center[i];


	velocity = make_float3(0.0f, -1.0f, 0.0f);
	position += velocity * time_step;
	center += velocity * time_step;

	buf._position[i] = position;
	buf._mix_velocity[i] = velocity;
	buf._center[i] = center;
}

__global__ void RigidBodyTransition(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == FLUID)
		return;

	//buf._type[i] = FLUID;
	if (buf._explosion[i] == 1)
	{
		buf._active[i] = false;
		buf._render[i] = false;
	}
}

__global__ void UpdateUpBound(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == FLUID)
		return;
	if (buf._explosion[i] == 1)
		return;
	float3 position = buf._position[i];
	float3 velocity = buf._mix_velocity[i];
	float3 center = buf._center[i];


	velocity = make_float3(0.0f, -1.0f, 0.0f);
	position += velocity * time_step;
	center += velocity * time_step;

	buf._position[i] = position;
	buf._mix_velocity[i] = velocity;
	buf._center[i] = center;
}


// scalar field //

// compute average position to smooth surface
__device__ float3 contributeAveragePos(int i, bufList buf, float3 pos, int grid_index, float& kernel_sum)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return make_float3(0.0f, 0.0f, 0.0f);
	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	float dist;
	float cubicterm, q;
	float kernel_factor;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == RIGID || buf._type[j] == BOUND)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			kernel_factor = cubicterm * buf._rest_mass[j] / buf._mix_density[j];
			result += kernel_factor * buf._position[j];
			kernel_sum += kernel_factor;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void GetAverageKernelPos(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._active[i] == false)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float3 ave_pos = make_float3(0.0f, 0.0f, 0.0f);
	float3 weighted_pos = make_float3(0.0f, 0.0f, 0.0f);
	const float3 _position_i = buf._position[i];
	const float3 self_distribution = (1.0f - simData.smoothing_factor) * _position_i;
	float kernel_sum = 0.0f;
	for (int cell = 0; cell < 27; cell++)
	{
		ave_pos += contributeAveragePos(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell], kernel_sum);
	}
	weighted_pos = ave_pos / kernel_sum;
	ave_pos = simData.smoothing_factor * weighted_pos;
	ave_pos += self_distribution;
	buf._ave_position[i] = ave_pos;
	buf._weighted_position[i] = weighted_pos;
	buf.particle_neighbor_number[i] = 0;
}


// compute position covariance matrix C
inline __device__ tensor TensorTranspose(tensor A)
{
	tensor A_T = A;
	A_T.tensor[1] = A.tensor[3];
	A_T.tensor[2] = A.tensor[6];
	A_T.tensor[5] = A.tensor[7];

	A_T.tensor[3] = A.tensor[1];
	A_T.tensor[6] = A.tensor[2];
	A_T.tensor[7] = A.tensor[5];

	return A_T;
}
inline __device__ tensor TensorOutproduct(float3 vec1, float3 vec2)
{
	tensor Outproduct_v12;
	Outproduct_v12.tensor[0] = vec1.x * vec2.x;
	Outproduct_v12.tensor[1] = vec1.x * vec2.y;
	Outproduct_v12.tensor[2] = vec1.x * vec2.z;
	Outproduct_v12.tensor[3] = vec1.y * vec2.x;
	Outproduct_v12.tensor[4] = vec1.y * vec2.y;
	Outproduct_v12.tensor[5] = vec1.y * vec2.z;
	Outproduct_v12.tensor[6] = vec1.z * vec2.x;
	Outproduct_v12.tensor[7] = vec1.z * vec2.y;
	Outproduct_v12.tensor[8] = vec1.z * vec2.z;

	return Outproduct_v12;
}
__device__ float contributePosCovariance(int i, bufList buf, float3 pos, int grid_index, tensor& C)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	tensor C_j;
	float kernel_factor;
	float3 ave_pos_j;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float ks = 1.0f;// / smooth_radius;
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == RIGID || buf._type[j] == BOUND)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			kernel_factor = cubicterm * buf._rest_mass[j] / buf._mix_density[j];
			result += kernel_factor;
			ave_pos_j = ks * (buf._position[j] - buf._weighted_position[i]);
			C_j = TensorOutproduct(ave_pos_j, ave_pos_j);
			for (int idx = 0; idx < 9; idx++)
			{
				C.tensor[idx] += kernel_factor * C_j.tensor[idx];
			}
			if (dist < simData.anisotropic_radius)
				buf.particle_neighbor_number[i]++;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputePosCovariance(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._active[i] == false)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float kernelsum = 0.0f;
	float3 _position_i = buf._position[i];
	const float sq_h = simData.smooth_radius * simData.smooth_radius;
	tensor C = { { sq_h, 0.0f, 0.0f, 0.0f, sq_h, 0.0f, 0.0f, 0.0f, sq_h} };
	//tensor C = { {0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f} };
	for (int cell = 0; cell < 27; cell++)
	{
		kernelsum += contributePosCovariance(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell], C);
	}
	for (int index = 0; index < 9; index++)
	{
		C.tensor[index] /= kernelsum;
	}
	buf._C[i] = C;
}

// compute shape matrix G
// get eigenvector of a 3x3 matrix
__device__ float3 GaussianReduce3x3(tensor A)
{
	float a1 = abs(A.tensor[0]);
	float a2 = abs(A.tensor[3]);
	float a3 = abs(A.tensor[6]);
	int row_idx;
	if (a1 > a2)
	{
		row_idx = 0;
		if (a1 < a3) row_idx = 2;
	}
	else
	{
		row_idx = 1;
		if (a2 < a3) row_idx = 2;
	}
	float temp1, temp2, temp3;
	if (row_idx > 0)
	{
		temp1 = A.tensor[0];
		temp2 = A.tensor[1];
		temp3 = A.tensor[2];
		A.tensor[0] = A.tensor[row_idx * 3];
		A.tensor[1] = A.tensor[row_idx * 3 + 1];
		A.tensor[2] = A.tensor[row_idx * 3 + 2];
		A.tensor[row_idx * 3] = temp1;
		A.tensor[row_idx * 3 + 1] = temp2;
		A.tensor[row_idx * 3 + 2] = temp3;
	}
	float factor1 = A.tensor[3] / A.tensor[0];
	float factor2 = A.tensor[6] / A.tensor[0];
	float factor3;
	A.tensor[3] -= factor1 * A.tensor[0];
	A.tensor[4] -= factor1 * A.tensor[1];
	A.tensor[5] -= factor1 * A.tensor[2];
	A.tensor[6] -= factor2 * A.tensor[0];
	A.tensor[7] -= factor2 * A.tensor[1];
	A.tensor[8] -= factor2 * A.tensor[2];
	if (abs(A.tensor[4]) < abs(A.tensor[7]))
	{
		temp1 = A.tensor[4];
		temp2 = A.tensor[5];
		A.tensor[4] = A.tensor[7];
		A.tensor[5] = A.tensor[8];
		A.tensor[7] = temp1;
		A.tensor[8] = temp2;
	}
	factor3 = A.tensor[7] / A.tensor[4];
	A.tensor[7] -= factor3 * A.tensor[4];
	A.tensor[8] -= factor3 * A.tensor[5];
	float3 eigen_vec;
	eigen_vec.z = 1.0f;
	eigen_vec.y = -A.tensor[5] / A.tensor[4];
	eigen_vec.x = (-A.tensor[2] - A.tensor[1] * eigen_vec.y) / A.tensor[0];
	const float norm = 1.0f / sqrt(eigen_vec.x * eigen_vec.x + eigen_vec.y * eigen_vec.y + eigen_vec.z * eigen_vec.z);
	eigen_vec.z *= norm;
	eigen_vec.y *= norm;
	eigen_vec.x *= norm;


	return eigen_vec;
}
__device__ float3 SymmetricSVD(tensor A, int i)
{
	const float m = (A.tensor[0] + A.tensor[4] + A.tensor[8]) / 3.0f;
	const tensor I = { m, 0.0f, 0.0f, 0.0f, m, 0.0f, 0.0f, 0.0f, m };
	tensor B = A;
	float p = 0.0f;
	for (int idx = 0; idx < 9; idx++)
	{
		B.tensor[idx] -= I.tensor[idx];
		p += B.tensor[idx] * B.tensor[idx];
	}
	const float q = tensorDet(B) / 2.0f;
	const float norm = q / abs(q);
	p = p / 6.0f;
	const float sqr_p = sqrt(p);
	float sin_phi, cos_phi, tan_phi;

	if (abs(q) < 1e-5)
	{
		sin_phi = 1.0f;
		cos_phi = 0.0f;
		tan_phi = 99999.0f;
	}
	else
	{
		tan_phi = sqrt(abs(p * p * p - q * q)) / (3.0f * q);
		const float phi_factor = 1.0f / sqrt(1.0f + tan_phi * tan_phi);
		sin_phi = norm * tan_phi * phi_factor;
		cos_phi = norm * phi_factor;
	}

	float eigen_value[3];
	eigen_value[0] = m + 2.0f * sqr_p * cos_phi;
	eigen_value[1] = m - sqr_p * (cos_phi - 1.732051f * sin_phi);
	eigen_value[2] = m - sqr_p * (cos_phi + 1.732051f * sin_phi);

	bool flag = true;
	while (flag)
	{
		flag = false;
		for (int i = 0; i < 2; i++)
		{
			if (eigen_value[i] < eigen_value[i + 1])
			{
				flag = true;
				float t = eigen_value[i];
				eigen_value[i] = eigen_value[i + 1];
				eigen_value[i + 1] = t;
			}
		}
	}

	return make_float3(eigen_value[0], eigen_value[1], eigen_value[2]);
}

__device__ bool Jacobi(float* matrix, int dim, float* eigenvectors, float* eigenvalues, float precision, int max)
{
	for (int i = 0; i < dim; i++) {
		eigenvectors[i * dim + i] = 1.0f;
		for (int j = 0; j < dim; j++) {
			if (i != j)
				eigenvectors[i * dim + j] = 0.0f;
		}
	}

	int nCount = 0;		//current iteration
	while (1) {
		//find the largest element on the off-diagonal line of the matrix
		float dbMax = matrix[1];
		int nRow = 0;
		int nCol = 1;
		for (int i = 0; i < dim; i++) {			//row
			for (int j = 0; j < dim; j++) {		//column
				float d = fabs(matrix[i * dim + j]);
				if ((i != j) && (d > dbMax)) {
					dbMax = d;
					nRow = i;
					nCol = j;
				}
			}
		}

		if (dbMax < precision)     //precision check 
			break;
		if (nCount > max)       //iterations check
			break;
		nCount++;

		float dbApp = matrix[nRow * dim + nRow];
		float dbApq = matrix[nRow * dim + nCol];
		float dbAqq = matrix[nCol * dim + nCol];
		//compute rotate angle
		float dbAngle = 0.5f * atan2(-2.0f * dbApq, dbAqq - dbApp);
		float dbSinTheta = sin(dbAngle);
		float dbCosTheta = cos(dbAngle);
		float dbSin2Theta = sin(2 * dbAngle);
		float dbCos2Theta = cos(2 * dbAngle);
		matrix[nRow * dim + nRow] = dbApp * dbCosTheta * dbCosTheta +
			dbAqq * dbSinTheta * dbSinTheta + 2.0f * dbApq * dbCosTheta * dbSinTheta;
		matrix[nCol * dim + nCol] = dbApp * dbSinTheta * dbSinTheta +
			dbAqq * dbCosTheta * dbCosTheta - 2.0f * dbApq * dbCosTheta * dbSinTheta;
		matrix[nRow * dim + nCol] = 0.5f * (dbAqq - dbApp) * dbSin2Theta + dbApq * dbCos2Theta;
		matrix[nCol * dim + nRow] = matrix[nRow * dim + nCol];

		for (int i = 0; i < dim; i++) {
			if ((i != nCol) && (i != nRow)) {
				int u = i * dim + nRow;	//p  
				int w = i * dim + nCol;	//q
				dbMax = matrix[u];
				matrix[u] = matrix[w] * dbSinTheta + dbMax * dbCosTheta;
				matrix[w] = matrix[w] * dbCosTheta - dbMax * dbSinTheta;
			}
		}

		for (int j = 0; j < dim; j++) {
			if ((j != nCol) && (j != nRow)) {
				int u = nRow * dim + j;	//p
				int w = nCol * dim + j;	//q
				dbMax = matrix[u];
				matrix[u] = matrix[w] * dbSinTheta + dbMax * dbCosTheta;
				matrix[w] = matrix[w] * dbCosTheta - dbMax * dbSinTheta;
			}
		}

		//compute eigenvector
		for (int i = 0; i < dim; i++) {
			int u = i * dim + nRow;		//p   
			int w = i * dim + nCol;		//q
			dbMax = eigenvectors[u];
			eigenvectors[u] = eigenvectors[w] * dbSinTheta + dbMax * dbCosTheta;
			eigenvectors[w] = eigenvectors[w] * dbCosTheta - dbMax * dbSinTheta;
		}
	}
	for (int i = 0; i < dim; i++) {
		eigenvalues[i] = matrix[i * dim + i];
	}
	/*
	//sort eigenvalues
	std::map<double, int> mapEigen;
	for (int i = 0; i < dim; i++) {
		eigenvalues[i] = matrix[i * dim + i];
		mapEigen.insert(make_pair(eigenvalues[i], i));
	}

	double* pdbTmpVec = new double[dim * dim];
	std::map<double, int>::reverse_iterator iter = mapEigen.rbegin();
	for (int j = 0; iter != mapEigen.rend(), j < dim; ++iter, ++j) {
		for (int i = 0; i < dim; i++) {
			pdbTmpVec[i * dim + j] = eigenvectors[i * dim + iter->second];
		}
		eigenvalues[j] = iter->first;
	}

	for (int i = 0; i < dim; i++) {
		double dSumVec = 0;
		for (int j = 0; j < dim; j++)
			dSumVec += pdbTmpVec[j * dim + i];
		if (dSumVec < 0) {
			for (int j = 0; j < dim; j++)
				pdbTmpVec[j * dim + i] *= -1;
		}
	}
	memcpy(eigenvectors, pdbTmpVec, sizeof(double) * dim * dim);
	delete[]pdbTmpVec;*/
	return true;
}
inline __device__ void swap_column3x3(float* eigenvectors, int c_1, int c_2)
{
	const float3 column_temp = make_float3(eigenvectors[c_1], eigenvectors[c_1 + 3], eigenvectors[c_1 + 6]);
	eigenvectors[c_1] = eigenvectors[c_2];
	eigenvectors[c_1 + 3] = eigenvectors[c_2 + 3];
	eigenvectors[c_1 + 6] = eigenvectors[c_2 + 6];
	eigenvectors[c_2] = column_temp.x;
	eigenvectors[c_2 + 3] = column_temp.y;
	eigenvectors[c_2 + 6] = column_temp.z;
}
__device__ int3 GetDominant(float* eigenvectors)
{
	int3 dominant = make_int3(0, 1, 2);
	float3 eigenvector_1 = make_float3(eigenvectors[0], eigenvectors[3], eigenvectors[6]);
	float3 eigenvector_2 = make_float3(eigenvectors[1], eigenvectors[4], eigenvectors[7]);
	float3 eigenvector_3 = make_float3(eigenvectors[2], eigenvectors[5], eigenvectors[8]);
	const float epsilon = 1e-6;

	if (eigenvector_1.y - eigenvector_1.x > epsilon)
	{
		if (eigenvector_1.y - eigenvector_1.z > epsilon)
		{
			dominant.x = 1;
		}
		else
			dominant.x = 2;
	}

	if (eigenvector_2.x - eigenvector_2.y > epsilon)
	{
		if (eigenvector_2.x - eigenvector_2.z > epsilon)
		{
			dominant.y = 0;
		}
		else
			dominant.y = 2;
	}

	if (eigenvector_3.x - eigenvector_3.z > epsilon)
	{
		if (eigenvector_3.x - eigenvector_3.y > epsilon)
		{
			dominant.z = 0;
		}
		else
			dominant.z = 1;
	}
	return dominant;
}

inline __device__ float3 TensorMulVec(tensor A, float3 V)
{
	float3 result;
	result.x = A.tensor[0] * V.x + A.tensor[1] * V.y + A.tensor[2] * V.z;
	result.y = A.tensor[3] * V.x + A.tensor[4] * V.y + A.tensor[5] * V.z;
	result.z = A.tensor[6] * V.x + A.tensor[7] * V.y + A.tensor[8] * V.z;

	return result;
}
inline __device__ tensor TensorInit(float3 column1, float3 column2, float3 column3)
{
	tensor init_tensor;
	init_tensor.tensor[0] = column1.x;
	init_tensor.tensor[3] = column1.y;
	init_tensor.tensor[6] = column1.z;
	init_tensor.tensor[1] = column2.x;
	init_tensor.tensor[4] = column2.y;
	init_tensor.tensor[7] = column2.z;
	init_tensor.tensor[2] = column3.x;
	init_tensor.tensor[5] = column3.y;
	init_tensor.tensor[8] = column3.z;

	return init_tensor;
}
inline __device__ float3 VecOutproduct(float3 vec1, float3 vec2)
{
	return make_float3(vec1.y * vec2.z - vec1.z * vec2.y, vec1.z * vec2.x - vec2.z * vec1.x, vec1.x * vec2.y - vec2.x * vec1.y);
}
__global__ void ComputeShapeMatG(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._active[i] == false)
		return;

	const tensor C = buf._C[i];
	const float epsilon = 1e-7;
	const int max_iter = 20;
	float mat_C[9] = { C.tensor[0],C.tensor[1], C.tensor[2],C.tensor[3],C.tensor[4], C.tensor[5],C.tensor[6],C.tensor[7], C.tensor[8] };
	float eigenvector[9] = { 0.0f };
	float eigenvalue[3] = { 0.0f };
	Jacobi(mat_C, 3, eigenvector, eigenvalue, epsilon, max_iter);

	bool flag = true;
	while (flag)
	{
		flag = false;
		for (int i = 0; i < 2; i++)
		{
			if (eigenvalue[i] < eigenvalue[i + 1])
			{
				flag = true;
				float t = eigenvalue[i];
				eigenvalue[i] = eigenvalue[i + 1];
				eigenvalue[i + 1] = t;
				swap_column3x3(eigenvector, i, i + 1);
			}
		}
	}

	const float3 eigen_value = make_float3(eigenvalue[0], eigenvalue[1], eigenvalue[2]);
	tensor R = { {eigenvector[0],eigenvector[1], eigenvector[2],eigenvector[3], eigenvector[4],eigenvector[5], eigenvector[6],eigenvector[7], eigenvector[8]} };
	tensor R_T = TensorTranspose(R);

	tensor Sigma = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	if (buf.particle_neighbor_number[i] > simData.search_num_threshold)
	{
		Sigma.tensor[0] = 1.0f / (simData.diag_k_s * eigen_value.x);
		Sigma.tensor[4] = 1.0f / (simData.diag_k_s * max(eigen_value.x, eigen_value.y / simData.diag_k_r));
		Sigma.tensor[8] = 1.0f / (simData.diag_k_s * max(eigen_value.x, eigen_value.z / simData.diag_k_r));
	}
	else
	{
		Sigma.tensor[0] = 1.0f / (simData.diag_k_n);
		Sigma.tensor[4] = 1.0f / (simData.diag_k_n);
		Sigma.tensor[8] = 1.0f / (simData.diag_k_n);
	}
	tensor G = TensorMutiply(R, Sigma);
	G = TensorMutiply(G, R_T);

	tensor G_0 = { {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f} };
	buf._G[i] = G_0;
}

// Compute G test
__global__ void ComputeShapeMatGSVD(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._active[i] == false)
		return;

	tensor G = { {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f} };
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float h_inv = 1.0f / smooth_radius;


	if (buf.particle_neighbor_number[i] < simData.search_num_threshold)
	{
		G.tensor[0] = G.tensor[4] = G.tensor[8] = h_inv * simData.diag_k_n;
	}
	else
	{
		tensor C = buf._C[i];
		float cov[9] = { C.tensor[0],C.tensor[1], C.tensor[2],C.tensor[3],C.tensor[4], C.tensor[5],C.tensor[6],C.tensor[7], C.tensor[8] };

		// singular value decomposition
		float u[9] = { 0.0f };
		float3 v;
		float w[9] = { 0.0f };
		svd(cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7], cov[8],
			u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
			v.x, v.y, v.z,
			w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]);

		v.x = fabsf(v.x);
		v.y = fabsf(v.y);
		v.z = fabsf(v.z);

		float max_singular_value = max(v.x, max(v.y, v.z)) / simData.diag_k_r;
		v.x = max(v.x, max_singular_value);
		v.y = max(v.y, max_singular_value);
		v.z = max(v.z, max_singular_value);


		float3 invV;
		invV.x = 1.0f / v.x;
		invV.y = 1.0f / v.y;
		invV.z = 1.0f / v.z;

		G.tensor[0] = u[0] * invV.x; G.tensor[1] = u[1] * invV.x; G.tensor[2] = u[2] * invV.x;
		G.tensor[3] = u[3] * invV.y; G.tensor[4] = u[4] * invV.y; G.tensor[5] = u[5] * invV.y;
		G.tensor[6] = u[6] * invV.z; G.tensor[7] = u[7] * invV.z; G.tensor[8] = u[8] * invV.z;
		
		tensor W = { {w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]} };
		G = TensorMutiply(W, G);

		float scale = powf(v.x * v.y * v.z, 1.0 / 3.0);
		float cof = h_inv * scale;

		for (int idx = 0; idx < 9; idx++)
		{
			G.tensor[idx] *= cof;
		}
	}
	
	buf._G[i] = G;
}

// compute density scalar field for surface tracking
__device__ float contributeScalarField(int i, bufList buf, float3 pos, int grid_index)
{
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const tensor G = buf._G[i];
	const float factor_G = tensorDet(G);
	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] == RIGID || buf._type[j] == BOUND)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		float3 pos_ij = pos - buf._position[j];
		float3 anisotropic_pos_ij = TensorMulVec(G, pos_ij);
		const float dx = anisotropic_pos_ij.x;
		const float dy = anisotropic_pos_ij.y;
		const float dz = anisotropic_pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			result += factor_G * cubicterm * buf._rest_mass[j] / buf._mix_density[j];
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void CalDensityScalarFieldParticle(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;
	if (buf._active[i] == false)
		return;
	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;
	float sum = 0.0f;
	float3 _position_i = buf._ave_position[i];
	const float _mass_i = buf._rest_mass[i];
	const float _density_i = buf._mix_density[i];
	const float factor_G = tensorDet(buf._G[i]);
	const float self_distribution = (_mass_i / _density_i) * simData.kernel_self * factor_G;
	for (int cell = 0; cell < 27; cell++)
	{
		sum += contributeScalarField(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	sum += self_distribution;
	//if (buf.particle_neighbor_number[i] < 5)
		//sum *= 6.0f/((float)buf.particle_neighbor_number[i] + 1.0f);
	buf._surface_scalar_field[i] = sum;
}

// mcube grid
__global__ void AllocateMcGrid(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;
	const float3 grid_min = simData.boundary_min - simData.grid_boundary_offset;
	const int3 grid_resolution = simData.mc_grid_resolution;
	const float grid_radius = simData.mc_grid_radius;

	int3 Cell;
	uint CellIndex = GetCell(buf._position[i], grid_min, grid_resolution, grid_radius, Cell);

	if (Cell.x >= 0 && Cell.x < grid_resolution.x && Cell.y >= 0 && Cell.y < grid_resolution.y && Cell.z >= 0 && Cell.z < grid_resolution.z)
	{
		buf.particle_mc_grid_cellindex[i] = CellIndex;
		/*
		buf.next_particle_index[i] = buf.grid_particle_table[CellIndex];
		buf.grid_particle_table[CellIndex] = i;*/
		buf.mc_next_particle_index[i] = atomicExch(&buf.mc_grid_particle_table[CellIndex], i);
		atomicAdd(&buf.num_particle_mc_grid[CellIndex], 1);
	}
	else
	{
		buf.particle_mc_grid_cellindex[i] = GRID_UNDEF;
	}
}

__device__ int3 CellIndexToCell(uint CellIndex)
{
	int3 Cell;
	const int3 grid_res = simData.mc_grid_ver_resolution;

	Cell.x = CellIndex % grid_res.x;
	const uint CellIndex_jk = CellIndex / grid_res.x;
	Cell.y = CellIndex_jk % grid_res.y;
	Cell.z = CellIndex_jk / grid_res.y;

	return Cell;
}
__device__ float ScalarFieldPtoG(bufList buf, float3 grid_ver_pos, int grid_index, float3& color, float kernel_factor, float& p_num)
{
	if (grid_index<0 || grid_index>simData.mc_grid_number - 1)
		return 0.0f;
	float result = 0.0f;
	float dist;
	float cubicterm, q;
	const float itp_radius = simData.itp_radius;
	const float itp_radius_square = itp_radius * itp_radius;
	const float scale_factor = 0.00001f;
	uint j = buf.mc_grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (buf._type[j] == RIGID || buf._type[j] == BOUND)
		{
			j = buf.mc_next_particle_index[j];
			continue;
		}
		float3 pos_ij = grid_ver_pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < itp_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / itp_radius;
			if (q <= 0.5f)
				cubicterm = kernel_factor * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			else
				cubicterm = kernel_factor * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			result += buf._surface_scalar_field[j] * cubicterm * scale_factor;// *buf._rest_mass[j] / buf._mix_density[j];
			color += buf._color[j] * cubicterm * scale_factor;// *buf._rest_mass[j] / buf._mix_density[j];
			p_num += 1.0f;
		}
		j = buf.mc_next_particle_index[j];
	}
	return result;
}
__global__ void ParticleScalarvalueToGrid(bufList buf, int numMcGridver)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numMcGridver)
		return;
	int3 mc_grid_cell = CellIndexToCell(i);
	int cell_idx = mc_grid_cell.x + simData.mc_grid_resolution.x * (mc_grid_cell.y + simData.mc_grid_resolution.y * mc_grid_cell.z);
	float scalar_field_sum = 0.0f;
	const float3 grid_min = simData.boundary_min - simData.grid_boundary_offset;
	const float3 grid_ver_pos = grid_min + make_float3(mc_grid_cell.x, mc_grid_cell.y, mc_grid_cell.z) * simData.mc_grid_radius;

	float3 color = make_float3(0.0f, 0.0f, 0.0f);

	float p_num = 0.0f;
	const float kernel_factor = 8.0f / (3.141592 * pow(simData.itp_radius, 3));
	for (int cell = 0; cell < 729; cell++)
	{
		scalar_field_sum += ScalarFieldPtoG(buf, grid_ver_pos, cell_idx + simData.mc_grid_search_offset[cell], color, kernel_factor, p_num);
	}
	buf.scalar_field_value_grid[i] = scalar_field_sum;
	buf.color_field_grid[i] = color;
	//if (scalar_field_sum > 0.0f && i % 5000 == 0)
	//	printf("index: %d, scalar_value: %f, color: %f, %f, %f, p_num: %f\n", i, scalar_field_sum, color.x, color.y, color.z, p_num);
	/*
	if (i % 5000 == 0)
	{
		printf("index: %d, grid_cell: %d, %d, %d, gz: %f, g_min: %f, %f, %f, pos: %f, %f, %f, scalar_value: %f\n",i, mc_grid_cell.x, mc_grid_cell.y, mc_grid_cell.z, simData.mc_grid_radius, 40.0f * grid_min.x, 40.0f * grid_min.y, 40.0f * grid_min.z,
			40.0f * grid_ver_pos.x, 40.0f * grid_ver_pos.y, 40.0f * grid_ver_pos.z, scalar_field_sum);
	}
	if (i % 1000 == 0 && i < simData.mc_grid_number)
		printf("index: %d, grid_p_num: %d, grid_table: %d\n", i, buf.num_particle_mc_grid[i], buf.mc_grid_particle_table[i]);*/
}

// reset particle
__global__ void ResetParticles(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;

	buf._mix_velocity[i] = make_float3(0.0f, 0.0f, 0.0f);
	buf._acceleration[i] = make_float3(0.0f, 0.0f, 0.0f);
	buf._force[i] = make_float3(0.0f, 0.0f, 0.0f);
	buf._lambda[i] = 0.0f;
	buf._mix_density[i] = buf._mix_rest_density[i];
	tensor C = { 0.0 };
	buf._tensor_F[i] = C;
	buf._tensor_K[i] = C;
	for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
	{
		buf._delta_alpha[i * MAX_PHASE_NUMBER + fcount] = 0.0f;
		buf._alpha[i * MAX_PHASE_NUMBER + fcount] = buf._alpha_advanced[i * MAX_PHASE_NUMBER + fcount];
		buf._drift_velocity[i * MAX_PHASE_NUMBER + fcount] = make_float3(0.0f, 0.0f, 0.0f);
	}
	//if (i % 1000 == 0 && buf._type[i] == FLUID)
	//	printf("i: %d, alpha: %f, %f, %f, %f, %f\nalpha_advanced: %f, %f, %f, %f, %f\ndensity: %f, rest_density: %f, mass: %f, pressure: %f\n",i,
	//		buf._alpha_advanced[i * MAX_PHASE_NUMBER + 0], buf._alpha_advanced[i * MAX_PHASE_NUMBER + 1],
	//		buf._alpha_advanced[i * MAX_PHASE_NUMBER + 2], buf._alpha_advanced[i * MAX_PHASE_NUMBER + 3], buf._alpha_advanced[i * MAX_PHASE_NUMBER + 4],
	//		buf._alpha[i * MAX_PHASE_NUMBER + 0], buf._alpha[i * MAX_PHASE_NUMBER + 1],
	//		buf._alpha[i * MAX_PHASE_NUMBER + 2], buf._alpha[i * MAX_PHASE_NUMBER + 3], buf._alpha[i * MAX_PHASE_NUMBER + 4], buf._mix_density[i], buf._mix_rest_density[i],
	//		buf._rest_mass[i], buf._mix_pressure[i]);
}


// secondary bubbles //

// compute interpolate alpha
__device__ float* computeAlpha(int i, bufList buf, float3 pos, int grid_index)
{
	float result[MAX_PHASE_NUMBER] = { 0.0f };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			cubicterm *= _volume_j;
			
			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				result[fcount] += _volume_j * buf._alpha_advanced[j * MAX_PHASE_NUMBER + fcount] * cubicterm;
			}
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeBubbleAlpha(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	float3 _position_i = buf._position[i];
	const float _mass_i = buf._rest_mass[i];
	float inter_alpha[MAX_PHASE_NUMBER] = { 0.0f };
	float* contribute_ptr;
	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = computeAlpha(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			inter_alpha[fcount] += contribute_ptr[fcount];
		}
	}

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		buf.inter_alpha[i * MAX_PHASE_NUMBER + fcount] = inter_alpha[fcount];
	}
}

// insert bubbles
__device__ void insertParticlePos(int i, bufList buf, float volume, int bubble_type)
{
	if (buf.bubbleNum[0] > MAX_BUBBLE_NUM || i%100!=0)
		return;
	int rel_idx = atomicAdd(&buf.bubbleNum[0], 1);
	int idx = simData.num_particle - rel_idx;
	const float3 pos = buf._position[i];

	buf.bubbleList[rel_idx] = idx;
	buf.bubblePosList[rel_idx] = pos;
	buf.idxToListIdx[idx] = rel_idx;

	if (bubble_type == MIDDLEBUBBLE)
		buf.attached_id[idx] = i;

	// bubble setting
	buf.bubble[idx] = true;
	buf.bubble_volume[idx] = volume;
	buf.bubble_type[idx] = bubble_type;
	// global particle setting
	buf._active[idx] = true;
	buf._position[idx] = pos;
	buf._type[idx] = BUBBLE;
	buf._render[idx] = true;
	buf._particle_radius[idx] = 0.5f * pow(volume, 1.0f / 3.0f);
}
__global__ void InsertBubbleParticle(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	const float gas_alpha = buf._alpha_advanced[i * MAX_PHASE_NUMBER+3];
	const float gas_mass_fraction = gas_alpha * simData.phase_density[3] / buf._mix_density[i];
	const float inter_alpha = buf.inter_alpha[i * MAX_PHASE_NUMBER+3];
	const float volume_i = buf._rest_mass[i] / buf._mix_density[i];
	const float epsilon = 0.5f;
	float V_new = 0.0f;
	if (gas_alpha < simData.theta1 && gas_alpha > 0.01f)
	{
		if (gas_alpha - inter_alpha > epsilon * gas_alpha)
		{
			V_new = volume_i * (gas_alpha - inter_alpha);
			insertParticlePos(i, buf, V_new, SMALLBUBBLE);
		}
	}
	//if (gas_alpha >= simData.theta1 && gas_mass_fraction < simData.theta2)
	//{
	//	V_new = gas_alpha * volume_i;
	//	insertParticlePos(i, buf, V_new, MIDDLEBUBBLE);
	//}
}

// advance bubbles
__device__ float3 contributeVel(int i, bufList buf, float3 pos, int grid_index)
{
	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			cubicterm *= -_volume_j;

			// fcount = 0 means that phase 0 is air
			result += cubicterm * (buf._mix_velocity[j] + buf._drift_velocity[muloffset_j+3]);
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void UpdatePosBubble(bufList buf, int numParticles, float time_step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == false)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	if (buf.bubble_type[i] == MIDDLEBUBBLE)
	{
		const int attached_id = buf.attached_id[i];
		buf._position[i] = buf._position[attached_id];
		buf._mix_velocity[i] = buf._mix_velocity[attached_id];
		return;
	}

	float3 bubble_vel = make_float3(0.0f, 0.0f, 0.0f);
	if (buf.bubble_type[i] == SMALLBUBBLE)
	{
		for (int cell = 0; cell < 27; cell++)
		{
			bubble_vel += contributeVel(i, buf, buf._position[i], iCellIndex + simData.grid_search_offset[cell]);
		}
		buf._mix_velocity[i] = bubble_vel;
	}
	bubble_vel = make_float3(0.0f, 1.0f, 0.0f);
	float3 position_advanced = buf._position[i] + bubble_vel * time_step;
	buf._position[i] = position_advanced;
	buf.bubblePosList[buf.idxToListIdx[i]] = position_advanced;
	//printf("UpdatePosBubble: bubble idx: %d, type: %d\n", i, buf._type[i]);
}

__device__ void deleteParticlePos(int i, bufList buf)
{
	buf._render[i] = false;
	buf.bubble[i] = false;
	buf._active[i] = false;
	int num = atomicAdd(&buf.bubbleNum[0], -1);
	int rel_idx = buf.idxToListIdx[i];
	buf.bubbleList[rel_idx] = UNDEF_INT;
}
__global__ void DeleteBubbleParticle(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == false)
		return;

	if (buf._position[i].y * simData.simscale > -10.0f)
	{
		deleteParticlePos(i, buf);
	}
}


// air-liquid surface tension //
__device__ float3* computeNablaC(int i, bufList buf, float3 pos, float* c, int grid_index)
{
	float3 result[MAX_PHASE_NUMBER] = { make_float3(0.0f,0.0f,0.0f) };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float density_i = buf._mix_density[i];
	float K_xi_x, K_xi_y, K_xi_z;
	float3 inv_K_mul_xi;
	tensor K = buf._tensor_K[i];
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);

			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				float square_density_j = buf._mix_density[j] * buf._mix_density[j];
				result[fcount] += density_i * buf._rest_mass[j] * (c[fcount]/(density_i*density_i)+buf._mass_fraction[muloffset_j+fcount]/ square_density_j) * inv_K_mul_xi * cubicterm;
			}
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeSfParticlePhase(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;


	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;
	float3 _position_i = buf._position[i];
	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _mass_fraction_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_mass_fraction_i[fcount] = buf._mass_fraction[muloffset_i + fcount];
	}

	float3 nabla_c[MAX_PHASE_NUMBER] = { make_float3(0.0f,0.0f,0.0f) };
	float3* contribute_ptr;

	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = computeNablaC(i, buf, _position_i, _mass_fraction_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			nabla_c[fcount] += contribute_ptr[fcount];
		}
	}

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		buf._nabla_c[muloffset_i + fcount] = nabla_c[fcount];
	}
}


__device__ float* computeSF(int i, bufList buf, float3 pos, float* c, int grid_index)
{
	float result[MAX_PHASE_NUMBER] = { 0.0f };
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float density_i = buf._mix_density[i];
	const float sq_eta = 0.1f * 0.1f * smooth_radius_square;
	float K_xi_x, K_xi_y, K_xi_z;
	float3 inv_K_mul_xi;
	tensor K = buf._tensor_K[i];
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	int muloffset_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		muloffset_j = j * MAX_PHASE_NUMBER;
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			cubicterm *= _volume_j;
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);

			for (int fcount = 0; fcount < simData.phase_number; fcount++)
			{
				float inv_nabla_j = 1.0f / length(buf._nabla_c[muloffset_j + fcount]);
				float inv_nabla_i = 1.0f / length(buf._nabla_c[i * MAX_PHASE_NUMBER + fcount]);
				if (length(buf._nabla_c[i * MAX_PHASE_NUMBER + fcount]) < 1e-6)
					inv_nabla_i = 0.0f;
				if (length(buf._nabla_c[muloffset_j + fcount]) < 1e-6)
					inv_nabla_j = 0.0f;
				result[fcount] += cubicterm * dot(pos_ij, inv_K_mul_xi) * (inv_nabla_i + inv_nabla_j) * (buf._mass_fraction[i * MAX_PHASE_NUMBER + fcount] - buf._mass_fraction[muloffset_j + fcount]) / (dist_square + sq_eta);
				//result[fcount] += density_i * buf._rest_mass[j] * (c[fcount] / (density_i * density_i) + buf._mass_fraction[muloffset_j + fcount] / square_density_j) * inv_K_mul_xi * cubicterm;
			}
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeSurfaceTensionParticle(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// param
	const float epsilon = 0.01f;
	const float coef = -6.0f * sqrt(2.0f) * epsilon;
	const float sigma = 0.01f;

	// definition
	const int muloffset_i = i * MAX_PHASE_NUMBER;
	float3 _position_i = buf._position[i];
	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _mass_fraction_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;

	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		_mass_fraction_i[fcount] = buf._mass_fraction[muloffset_i + fcount];
	}

	float divergence_rel_c[MAX_PHASE_NUMBER] = { 0.0f };
	float* contribute_ptr;

	for (int cell = 0; cell < 27; cell++)
	{
		contribute_ptr = computeSF(i, buf, _position_i, _mass_fraction_i, iCellIndex + simData.grid_search_offset[cell]);
		for (int fcount = 0; fcount < simData.phase_number; fcount++)
		{
			divergence_rel_c[fcount] += contribute_ptr[fcount];
		}
	}

	float3 sf_c = make_float3(0.0f, 0.0f, 0.0f);
	float kai = 5.0f * buf._mass_fraction[muloffset_i + 1] * buf._mass_fraction[muloffset_i + 4];
	if (buf._mass_fraction[muloffset_i + 1] > 1.0f || buf._mass_fraction[muloffset_i + 1] < 0.0f)
		kai = 0.0f;
	for (int fcount = 0; fcount < simData.phase_number; fcount++)
	{
		sf_c += coef * divergence_rel_c[fcount] * length(buf._nabla_c[muloffset_i + fcount]) * buf._nabla_c[muloffset_i + fcount];
	}

	buf._surface_tension[i] = 0.5f * sigma * sf_c * kai;
}


// vorticity //
__device__ float3 computeVorticity(int i, bufList buf, float3 pos, int grid_index)
{
	float3 result = make_float3(0.0f,0.0f,0.0f);
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float density_i = buf._mix_density[i];
	float K_xi_x, K_xi_y, K_xi_z;
	float3 inv_K_mul_xi;
	tensor K = buf._tensor_K[i];
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			_volume_j = buf._rest_mass[j] / buf._mix_density[j];
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);

			result += -cross((buf._mix_velocity[i] - buf._mix_velocity[j]), inv_K_mul_xi) * cubicterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeVorticityParticle(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;


	// definition
	float3 _position_i = buf._position[i];
	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _mass_fraction_i[MAX_PHASE_NUMBER] = { 0.0f };
	float _volume = _mass_i / _mix_rest_density_i;
	
	float3 vorticity = make_float3(0.0f, 0.0f, 0.0f);

	for (int cell = 0; cell < 27; cell++)
	{
		vorticity += computeVorticity(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
	}
	buf._vorticity[i] = vorticity;
}


__device__ float3 computeEta(int i, bufList buf, float3 pos, int grid_index)
{
	float3 result = make_float3(0.0f,0.0f,0.0f);
	if (grid_index<0 || grid_index>simData.grid_number - 1)
		return result;
	float dist;
	float cubicterm, q;
	const float smooth_radius = simData.smooth_radius;
	const float smooth_radius_square = smooth_radius * smooth_radius;
	const float density_i = buf._mix_density[i];
	float K_xi_x, K_xi_y, K_xi_z;
	float3 inv_K_mul_xi;
	tensor K = buf._tensor_K[i];
	float _volume_i = buf._rest_mass[i] / buf._mix_density[i];
	float _volume_j;
	float3 pos_ij;

	uint j = buf.grid_particle_table[grid_index];
	while (j != GRID_UNDEF)
	{
		if (j == i || buf._type[j] != FLUID || buf.bubble[j] == true)
		{
			j = buf.next_particle_index[j];
			continue;
		}
		pos_ij = pos - buf._position[j];
		const float dx = pos_ij.x;
		const float dy = pos_ij.y;
		const float dz = pos_ij.z;
		const float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square < smooth_radius_square)
		{
			dist = sqrt(dist_square);
			if (dist < 0.00001f)
			{
				dist = 0.00001f;
			}
			q = dist / smooth_radius;
			if (q <= 0.5f)
			{
				cubicterm = simData.CubicSplineKern * (1.0f + 6.0f * q * q * q - 6.0f * q * q);
			}
			else
			{
				cubicterm = simData.CubicSplineKern * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
			}
			K_xi_x = K.tensor[0] * dx + K.tensor[1] * dy + K.tensor[2] * dz;
			K_xi_y = K.tensor[3] * dx + K.tensor[4] * dy + K.tensor[5] * dz;
			K_xi_z = K.tensor[6] * dx + K.tensor[7] * dy + K.tensor[8] * dz;
			inv_K_mul_xi = make_float3(K_xi_x, K_xi_y, K_xi_z);

			float square_density_j = buf._mix_density[j] * buf._mix_density[j];
			result += density_i * buf._rest_mass[j] * (length(buf._vorticity[i]) / (density_i * density_i) + length(buf._vorticity[j]) / square_density_j) * inv_K_mul_xi * cubicterm;
		}
		j = buf.next_particle_index[j];
	}
	return result;
}
__global__ void ComputeForceVorticity(bufList buf, int numParticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numParticles)
		return;
	if (buf._active[i] == false || buf.bubble[i] == true)
		return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID)
		return;

	const uint iCellIndex = buf.particle_grid_cellindex[i];
	if (iCellIndex == GRID_UNDEF)
		return;

	// param
	const float epsilon = 0.01f;

	// definition
	float3 _position_i = buf._position[i];
	float _mass_i = buf._rest_mass[i];
	float _mix_density_i = buf._mix_density[i];
	float _mix_rest_density_i = buf._mix_rest_density[i];
	float _volume = _mass_i / _mix_rest_density_i;


	float3 eta = make_float3(0.0f, 0.0f, 0.0f);

	for (int cell = 0; cell < 27; cell++)
	{
		eta += computeEta(i, buf, _position_i, iCellIndex + simData.grid_search_offset[cell]);
	}

	float3 normal = eta / length(eta);

	buf._vorticity_force[i] = epsilon * cross(normal, buf._vorticity[i]);
}


// change:
__global__ void insertParticles(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;

	//float3 pos = buf._position[i] * 40.0f;
    //if (i % 5000 == 0)
    //    printf("%d, pos: %f, %f, %f\n", i, pos.x, pos.y, pos.z);
	register float3 gridMin = simData.boundary_min - simData.grid_boundary_offset;
	register float inv_dx = 1.0f / simData.grid_radius;
	register float3 gridDelta = make_float3(inv_dx, inv_dx, inv_dx);
	register int3 gridRes = simData.grid_resolution;
	register int3 gridScan = simData.grid_scan_max;
	//register float poff = simData.smooth_radius;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (buf._position[i] - gridMin) * gridDelta;
	gc = make_int3(int(gcf.x), int(gcf.y), int(gcf.z));
	gs = (gc.z * gridRes.y + gc.y) * gridRes.x + gc.x;
	if (gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z) {
		buf.particle_grid_cellindex[i] = gs;											// Grid cell insert.
		buf.particle_grid_index[i] = atomicAdd(&buf.num_particle_grid[gs], 1);			// Grid counts.

		//gcf = (-make_float3(poff, poff, poff) + buf._position[i] - gridMin) * gridDelta;
		//gc = make_int3(int(gcf.x), int(gcf.y), int(gcf.z));
		//gs = (gc.y * gridRes.z + gc.z) * gridRes.x + gc.x;
	}
	else {
		buf.particle_grid_cellindex[i] = GRID_UNDEF;
	}
	//if (i == 2863)
	//	printf("cell: %d (%d, %d, %d), pos: %f, %f, %f\n", gs, gc.x, gc.y, gc.z,
	//		simData.simscale * buf._position[i].x, simData.simscale * buf._position[i].y, simData.simscale * buf._position[i].z);
}

__global__ void prefixFixup(uint* input, uint* aux, int len)
{
	uint t = threadIdx.x;
	uint start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	if (start < len)					input[start] += aux[blockIdx.x];
	if (start + SCAN_BLOCKSIZE < len)	input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

__global__ void prefixSum(uint* input, uint* output, uint* aux, int len, int zeroff)
{
	__shared__ uint scan_array[SCAN_BLOCKSIZE << 1];
	uint t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	uint t2 = t1 + SCAN_BLOCKSIZE;

	// Pre-load into shared memory
	scan_array[threadIdx.x] = (t1 < len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2 < len) ? input[t2] : 0.0f;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * SCAN_BLOCKSIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * SCAN_BLOCKSIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}
	__syncthreads();

	// Output values & aux
	if (t1 + zeroff < len) output[t1 + zeroff] = scan_array[threadIdx.x];
	if (t2 + zeroff < len) output[t2 + zeroff] = (threadIdx.x == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];
	if (threadIdx.x == 0) {
		if (zeroff) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];
	}
}

__device__ int3 cellidTocell(int cell)
{
	int i = cell % simData.grid_resolution.x;
	int offset = cell / simData.grid_resolution.x;
	int j = offset % simData.grid_resolution.y;
	int k = offset / simData.grid_resolution.y;
	return make_int3(i, j, k);
}

__global__ void countingSortFull(bufList buf, bufList temp, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles)
		return;
	// Copy particle from original, unsorted buffer
	// into sorted memory location on device
	uint icell = temp.particle_grid_cellindex[i];
	
	if (icell != GRID_UNDEF) {
		// Determine the sort_ndx, location of the particle after sort
		uint indx = temp.particle_grid_index[i];
		int sort_ndx = buf.grid_offset[icell] + indx;		// global_ndx = grid_cell_offset + particle_offset

		//if (i % 1 == 0)
		//	printf("i: %d, cell: %d, grid_offset: %d, indx: %d\n", i, icell, buf.grid_offset[icell],indx);
		//transfer data to sort location

		buf.sorted_grid_map[sort_ndx] = sort_ndx;
		buf._position[sort_ndx] = temp._position[i];
        buf._external_force[sort_ndx]               = temp._external_force[i];
		buf._mix_velocity[sort_ndx] = temp._mix_velocity[i];
		buf._force[sort_ndx] = temp._force[i];
		buf._alpha_sum[sort_ndx] = temp._alpha_sum[i];
		buf._lambda[sort_ndx] = temp._lambda[i];
		buf._mix_pressure[sort_ndx] = temp._mix_pressure[i];
		buf._mix_density[sort_ndx] = temp._mix_density[i];
		buf._inv_dens[sort_ndx] = temp._inv_dens[i];
		buf._mix_rest_density[sort_ndx] = temp._mix_rest_density[i];
		buf._viscosity[sort_ndx] = temp._viscosity[i];
		buf._rest_mass[sort_ndx] = temp._rest_mass[i];
		buf._particle_radius[sort_ndx] = temp._particle_radius[i];
		buf._bound_phi[sort_ndx] = temp._bound_phi[i];
		buf._type[sort_ndx] = temp._type[i];
		buf._active[sort_ndx] = temp._active[i];
		buf._tensor_K[sort_ndx] = temp._tensor_K[i];
		buf._tensor_F[sort_ndx] = temp._tensor_F[i];
		buf.particle_grid_cellindex[sort_ndx] = icell;
		buf.particle_grid_index[sort_ndx] = indx;

		uint muloffset_i = i * MAX_PHASE_NUMBER;
		uint muloffset_sort = sort_ndx * MAX_PHASE_NUMBER;
		for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
		{
			buf._drift_velocity[muloffset_sort + fcount] = temp._drift_velocity[muloffset_i + fcount];
			buf._alpha[muloffset_sort + fcount] = temp._alpha[muloffset_i + fcount];
			buf._alpha_advanced[muloffset_sort + fcount] = temp._alpha_advanced[muloffset_i + fcount];
			buf._delta_alpha[muloffset_sort + fcount] = temp._delta_alpha[muloffset_i + fcount];
			//buf._delta_mass_k[muloffset_sort + fcount] = temp._delta_mass_k[muloffset_i + fcount];
		}/**/

		int3 Cell = CellIndexToCell(icell);
		//if (sort_ndx == 8000)
		//	printf("i: %d, cell: %d (%d, %d, %d), pos: %f, %f, %f\n", i, temp.particle_grid_cellindex[i], Cell.x, Cell.y, Cell.z,simData.simscale * temp._position[i].x,
		//		simData.simscale * temp._position[i].y, simData.simscale * temp._position[i].z);
	}
}

inline __device__ float cubicKernel(float r2, float h2)
{
	float q = sqrt(r2 / h2);
	return (q <= 0.5f ? (1.0f + 6.0f * q * q * q - 6.0f * q * q) : 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q));
}


__device__ float contributePhi(int i, bufList buf, float3 pos, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return 0.0;

	float3 dist;
	float dsq, c, sum = 0.0;
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (buf._type[pndx] == FLUID || pndx == i) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2);
			sum += c;
		}
	}
	return sum;
}
__global__ void computePhi(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == FLUID) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Sum kernel
	float3 pos = buf._position[i];
	register float sum = 0.0f;
	for (int c = 0; c < 27; c++) {
		sum += contributePhi(i, buf, pos, gc + simData.grid_search_offset[c]);
	}
	__syncthreads();

	sum *= simData.CubicSplineKern;
	buf._bound_phi[i] = 1.4f / sum;
	//if (i % 1000 == 0)
	//	printf("idx: %d, type: %d, phi: %f\n", i, buf._type[i], buf._bound_phi[i]);
}

__device__ float contributeDens(int i, bufList buf, float3 pos, float& sum_b, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return 0.0;

	float3 dist;
	float dsq, c, sum = 0.0;
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2);
			if (buf._type[pndx] == FLUID) sum += c * buf._rest_mass[i];
			else sum_b += c * buf._mix_rest_density[i] * buf._bound_phi[pndx];
		}
	}
	return sum;
}
__global__ void computeDensity(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Sum mass
	float3 pos = buf._position[i];
	register float sum_b = 0.0f, sum_f = 0.0f;
	for (int c = 0; c < 27; c++) {
		sum_f += contributeDens(i, buf, pos, sum_b, gc + simData.grid_search_offset[c]);
	}
	__syncthreads();

	if (buf._type[i] == FLUID) sum_f += buf._rest_mass[i];
	buf._mix_density[i] = (sum_f + sum_b) * simData.CubicSplineKern;
	buf._inv_dens[i] = 1.0f / buf._mix_density[i];

	// wcsph
	float relative_density = pow(buf._mix_density[i] / buf._mix_rest_density[i], 2);
	buf._mix_pressure[i] = max(0.0f, ((relative_density - 1.0f) * simData.gas_constant * buf._mix_rest_density[i]));
	//if (i % 1000 == 0)
	//	printf("ComputeDensity: %d, dens: %f, pressure: %f\n", i, buf._mix_density[i], buf._mix_pressure[i]);
}

__device__ float contributeK(int i, bufList buf, float3 pos, tensor& K, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return 0.0;

	float3 dist;
	float dsq, c, k, sum = 0.0;
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i || buf._type[pndx] == BOUND) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2);
			k = c * buf._rest_mass[pndx] * buf._inv_dens[pndx];
			// diag term:
			K.tensor[0] += k * dist.x * dist.x;
			K.tensor[4] += k * dist.y * dist.y;
			K.tensor[8] += k * dist.z * dist.z;
			// others:
			K.tensor[1] += k * dist.x * dist.y;
			K.tensor[2] += k * dist.x * dist.z;
			K.tensor[5] += k * dist.y * dist.z;
			sum += k;
		}
	}
	return sum;
}
__global__ void computeK(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._active[i] == false) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Sum kernel
	float3 pos = buf._position[i];
	register tensor K = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	register float sum = 0.0f;
	if (buf._type[i] == BOUND)
	{
		K.tensor[0] = 8000.0f;
		K.tensor[4] = 8000.0f;
		K.tensor[8] = 8000.0f;
		buf._tensor_K[i] = K;
		return;
	}

	for (int c = 0; c < 27; c++) {
		sum += contributeK(i, buf, pos, K, gc + simData.grid_search_offset[c]);
	}
	__syncthreads();

	float pV = buf._rest_mass[i] * buf._inv_dens[i];
	sum += pV;
	float self_contribute = pV * buf._particle_radius[i] * buf._particle_radius[i];
	// Symmetric Tensor K:
	K.tensor[3] = K.tensor[1];
	K.tensor[6] = K.tensor[2];
	K.tensor[7] = K.tensor[5];
	// Self contribute
	K.tensor[0] += self_contribute;
	K.tensor[4] += self_contribute;
	K.tensor[8] += self_contribute;
	for (int index = 0; index < 9; index++)
	{
		K.tensor[index] /= sum;
	}
	TensorInverse(K);
	buf._tensor_K[i] = K;
}

__device__ float contributeF(int i, bufList buf, float3 pos, float3 vel, float dt, tensor& K, tensor& F, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return 0.0;

	float3 dist;
	float dsq, c, k, sum = 0.0;
	float K_xi_x, K_xi_y, K_xi_z; // components of [inv(K) * \xi]
	float3 u;  // u = rel_vel * dt
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i || buf._type[pndx] == BOUND) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2);
			k = c * buf._rest_mass[pndx] * buf._inv_dens[pndx];
			u = -(buf._mix_velocity[pndx] - vel) * dt;
			K_xi_x = K.tensor[0] * dist.x + K.tensor[3] * dist.y + K.tensor[6] * dist.z;
			K_xi_y = K.tensor[1] * dist.x + K.tensor[4] * dist.y + K.tensor[7] * dist.z;
			K_xi_z = K.tensor[2] * dist.x + K.tensor[5] * dist.y + K.tensor[8] * dist.z;
			// diag term:
			F.tensor[0] += k * (dist.x + u.x) * K_xi_x;
			F.tensor[4] += k * (dist.y + u.y) * K_xi_y;
			F.tensor[8] += k * (dist.z + u.z) * K_xi_z;
			// others:
			F.tensor[1] += k * (dist.x + u.x) * K_xi_y;
			F.tensor[2] += k * (dist.x + u.x) * K_xi_z;
			F.tensor[3] += k * (dist.y + u.y) * K_xi_x;
			F.tensor[5] += k * (dist.y + u.y) * K_xi_z;
			F.tensor[6] += k * (dist.z + u.z) * K_xi_x;
			F.tensor[7] += k * (dist.z + u.z) * K_xi_y;
			sum += k;
		}
	}
	return sum;
}
__global__ void computeF(bufList buf, int numParticles, float time_step)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._active[i] == false) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Sum kernel
	float3 pos = buf._position[i];
	float3 vel = buf._mix_velocity[i];
	register tensor F = { {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
	register tensor K = buf._tensor_K[i];
	register float sum = 0.0f;
	float detF, pradius = buf._particle_radius[i];
	float self_contribute[9] = { 0.0f };
	if (buf._type[i] == BOUND)
	{
		F.tensor[0] = 1.0f;
		F.tensor[4] = 1.0f;
		F.tensor[8] = 1.0f;
		buf._tensor_F[i] = F;
		return;
	}

	for (int c = 0; c < 27; c++) {
		sum += contributeF(i, buf, pos, vel, time_step, K, F, gc + simData.grid_search_offset[c]);
	}
	__syncthreads();

	float pV = buf._rest_mass[i] * buf._inv_dens[i];
	sum += pV;
	// caculate the missing tensor-self
	float selfterm = 1.0f / simData.CubicSplineKern;
	self_contribute[0] = selfterm * pradius * (K.tensor[0] + K.tensor[3] + K.tensor[6]);
	self_contribute[4] = selfterm * pradius * (K.tensor[1] + K.tensor[4] + K.tensor[7]);
	self_contribute[8] = selfterm * pradius * (K.tensor[2] + K.tensor[5] + K.tensor[8]);
	for (int index = 0; index < 9; index++)
	{
		self_contribute[index] *= pV * pradius;
		F.tensor[index] += self_contribute[index];
		F.tensor[index] /= sum;
	}
	detF = tensorDet(F);
	if (detF < 0.1f)
		initTensor(F);
	else
		TensorInverse(F);
	buf._tensor_F[i] = F;
}

__global__ void setAlpha(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	const int muloff_i = i * MAX_PHASE_NUMBER;

	float alpha_sum = 0.0f;
	for (int f = 0; f < simData.phase_number; f++)
	{
		buf._alpha[muloff_i + f] = buf._alpha_advanced[muloff_i + f];
		alpha_sum += buf._alpha_advanced[muloff_i + f];
	}
	buf._alpha_sum[i] = alpha_sum;
	//if (i % 1000 == 0)
	//	printf("i: %d, alphasum: %f\n", i, alpha_sum);
}

__device__ void contributeDriftVel(int i, bufList buf, float3 pos, float* alpha, float* mass_fraction, float3* pterm, float3* aterm, tensor& K, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return;

	float3 dist, K_xi, pgradsum, alphagradsum;
	float dsq, c, p;
	float3 pgrad[MAX_PHASE_NUMBER], alphagrad[MAX_PHASE_NUMBER], relative_alpha_grad[MAX_PHASE_NUMBER];
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i || buf._type[pndx] == BOUND || buf._alpha_sum[pndx] < 0.00001f) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2) * (-buf._rest_mass[pndx] * buf._inv_dens[pndx]);
			K_xi.x = K.tensor[0] * dist.x + K.tensor[1] * dist.y + K.tensor[2] * dist.z;
			K_xi.y = K.tensor[3] * dist.x + K.tensor[4] * dist.y + K.tensor[5] * dist.z;
			K_xi.z = K.tensor[6] * dist.x + K.tensor[7] * dist.y + K.tensor[8] * dist.z;
			pgradsum = make_float3(0.0f, 0.0f, 0.0f);
			alphagradsum = make_float3(0.0f, 0.0f, 0.0f);
			float inv_alpha_sum = 1.0f / buf._alpha_sum[pndx];
			for (int f = 0; f < simData.phase_number; f++)
			{
				float nalpha = buf._alpha[pndx * MAX_PHASE_NUMBER + f] * inv_alpha_sum;
				// pressure term
				if (simData.miscible)
					p = c * (-alpha[f] * buf._mix_pressure[i] + nalpha * buf._mix_pressure[pndx]);
				else
					p = c * (-buf._mix_pressure[i] + buf._mix_pressure[pndx]);
				pgrad[f] = p * K_xi;
				pgradsum += pgrad[f] * mass_fraction[f];
				// alpha term
				alphagrad[f] = (-alpha[f] + nalpha) * c * K_xi;
				if (alpha[f] > 0.0001f) relative_alpha_grad[f] = alphagrad[f] / alpha[f];
				else relative_alpha_grad[f] = make_float3(0.0f, 0.0f, 0.0f);
				alphagradsum += mass_fraction[f] * relative_alpha_grad[f];/**/
			}
			for (int f = 0; f < simData.phase_number; f++)
			{
				pterm[f] -= simData.tau * (pgrad[f] - pgradsum);
				aterm[f] -= simData.sigma * (relative_alpha_grad[f] - alphagradsum);
			}

		}
	}
}
__global__ void computeDriftVel(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	if (buf._alpha_sum[i] < 0.01f)
	{
		for (int f = 0; f < simData.phase_number; f++)
			buf._drift_velocity[i * MAX_PHASE_NUMBER + f] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	float3 pos = buf._position[i];
	tensor K = buf._tensor_K[i];
	float _mix_density = 0.0f, sum_mass_dens = 0.0f;
	//float alpha[MAX_PHASE_NUMBER], mass_fraction[MAX_PHASE_NUMBER];
	//register float3 fterm[MAX_PHASE_NUMBER], pterm[MAX_PHASE_NUMBER], aterm[MAX_PHASE_NUMBER];
	//
	//
	float alpha0 = 0.0f, alpha1 = 0.0f;
	float mfrac0 = 0.0f, mfrac1 = 0.0f;
	float3 zero = make_float3(0.0f, 0.0f, 0.0f);
	float3 f0 = zero, f1 = zero, pt0 = zero, pt1 = zero, at0 = zero, at1 = zero;
	//
	//
	//
	float3 drift_vel[MAX_PHASE_NUMBER];
	const int muloff_i = i * MAX_PHASE_NUMBER;
	// Initialize
	/*
	for (int f = 0; f < simData.phase_number; f++)
	{
		if (buf._alpha_sum[i] > 0.0001f) alpha[f] = buf._alpha[muloff_i + f];
		else alpha[f] = 0.0f;
		_mix_density += alpha[f] * simData.phase_density[f];
	}*/
	if (buf._alpha_sum[i] > 0.0001f) {
		alpha0 = buf._alpha[muloff_i];
		alpha1 = buf._alpha[muloff_i + 1];
	}
	else
	{
		alpha0 = 0.0f;
		alpha1 = 0.0f;
	}
	_mix_density = alpha0 * simData.phase_density[0] + alpha1 * simData.phase_density[1];
	_mix_density = 1.0f / _mix_density;
	/*
	for (int f = 0; f < simData.phase_number; f++)
	{
		mass_fraction[f] = alpha[f] * simData.phase_density[f] * _mix_density;
		sum_mass_dens += mass_fraction[f] * simData.phase_density[f];
		fterm[f] = make_float3(0.0f, 0.0f, 0.0f);
		pterm[f] = make_float3(0.0f, 0.0f, 0.0f);
		aterm[f] = make_float3(0.0f, 0.0f, 0.0f);
	}*/
	mfrac0 = alpha0 * simData.phase_density[0] * _mix_density;
	mfrac1 = alpha1 * simData.phase_density[1] * _mix_density;
	sum_mass_dens = mfrac0 * simData.phase_density[0] + mfrac1 * simData.phase_density[1];

	// Sum terms
	for (int c = 0; c < 27; c++) {
		//contributeDriftVel(i, buf, pos, alpha, mass_fraction, pterm, aterm, K, gc + simData.grid_search_offset[c]);
		int cell = gc + simData.grid_search_offset[c];
		if (buf.num_particle_grid[cell] == 0) continue;

		float3 dist, K_xi, pgradsum, alphagradsum;
		float dsq, cubic, p0, p1;
		//float3 pgrad[MAX_PHASE_NUMBER], alphagrad[MAX_PHASE_NUMBER], relative_alpha_grad[MAX_PHASE_NUMBER];
		float3 pgrad0, pgrad1, alphagrad0, alphagrad1, rel_a_grad0, rel_a_grad1;
		register float r2 = simData.smooth_radius * simData.smooth_radius;

		int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

		for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
			int pndx = buf.sorted_grid_map[cndx];

			if (pndx == i || buf._type[pndx] == BOUND || buf._type[pndx] == RIGID || buf._alpha_sum[pndx] < 0.00001f) continue;

			dist = pos - buf._position[pndx];
			dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
			if (dsq < r2 && dsq>0.0) {
				cubic = cubicKernel(dsq, r2) * (-buf._rest_mass[pndx] * buf._inv_dens[pndx]);
				K_xi.x = K.tensor[0] * dist.x + K.tensor[1] * dist.y + K.tensor[2] * dist.z;
				K_xi.y = K.tensor[3] * dist.x + K.tensor[4] * dist.y + K.tensor[5] * dist.z;
				K_xi.z = K.tensor[6] * dist.x + K.tensor[7] * dist.y + K.tensor[8] * dist.z;
				pgradsum = make_float3(0.0f, 0.0f, 0.0f);
				alphagradsum = make_float3(0.0f, 0.0f, 0.0f);
				float inv_alpha_sum = 1.0f / buf._alpha_sum[pndx];/*
				for (int f = 0; f < simData.phase_number; f++)
				{
					float nalpha = buf._alpha[pndx * MAX_PHASE_NUMBER + f] * inv_alpha_sum;
					// pressure term
					if (simData.miscible)
						p = cubic * (-alpha[f] * buf._mix_pressure[i] + nalpha * buf._mix_pressure[pndx]);
					else
						p = cubic * (-buf._mix_pressure[i] + buf._mix_pressure[pndx]);
					pgrad[f] = p * K_xi;
					pgradsum += pgrad[f] * mass_fraction[f];
					// alpha term
					alphagrad[f] = (-alpha[f] + nalpha) * cubic * K_xi;
					if (alpha[f] > 0.0001f) relative_alpha_grad[f] = alphagrad[f] / alpha[f];
					else relative_alpha_grad[f] = make_float3(0.0f, 0.0f, 0.0f);
					alphagradsum += mass_fraction[f] * relative_alpha_grad[f];
				}
				for (int f = 0; f < simData.phase_number; f++)
				{
					//pterm[f] -= simData.tau *(pgrad[f] - pgradsum);
					//aterm[f] -= simData.sigma *(relative_alpha_grad[f] - alphagradsum);
				}*/

				//
				//
				float nalpha0 = buf._alpha[pndx * MAX_PHASE_NUMBER] * inv_alpha_sum;
				float nalpha1 = buf._alpha[pndx * MAX_PHASE_NUMBER + 1] * inv_alpha_sum;
				if (simData.miscible) {
					p0 = cubic * (-alpha0 * buf._mix_pressure[i] + nalpha0 * buf._mix_pressure[pndx]);
					p1 = cubic * (-alpha1 * buf._mix_pressure[i] + nalpha1 * buf._mix_pressure[pndx]);
				}
				else {
					p0 = cubic * (-buf._mix_pressure[i] + buf._mix_pressure[pndx]);
					p1 = p0;
				}
				pgrad0 = p0 * K_xi; pgrad1 = p1 * K_xi;
				pgradsum = pgrad0 * mfrac0 + pgrad1 * mfrac1;
				alphagrad0 = (-alpha0 + nalpha0) * cubic * K_xi; alphagrad1 = (-alpha1 + nalpha1) * cubic * K_xi;
				rel_a_grad0 = (alpha0 > 0.001f ? alphagrad0 / alpha0 : make_float3(0.0f, 0.0f, 0.0f));
				rel_a_grad1 = (alpha1 > 0.001f ? alphagrad1 / alpha1 : make_float3(0.0f, 0.0f, 0.0f));
				alphagradsum = mfrac0 * rel_a_grad0 + mfrac1 * rel_a_grad1;


				pt0 -= simData.tau * (pgrad0 - pgradsum); pt1 -= simData.tau * (pgrad1 - pgradsum);
				at0 -= simData.sigma * (rel_a_grad0 - alphagradsum); at1 -= simData.sigma * (rel_a_grad1 - alphagradsum);
				//
				//
			}
		}
	}
	__syncthreads();
	/*
	for (int f = 0; f < simData.phase_number; f++)
	{
		fterm[f] = simData.tau * (simData.phase_density[f] - sum_mass_dens) * (-buf._force[i]);
		//fterm = simData.tau * (simData.phase_density[f] - sum_mass_dens) * (-buf._force[i]);
		//buf._drift_velocity[muloff_i + f] = fterm + ((f == 2 ? (pt2 + at2) : make_float3(0.0f, 0.0f, 0.0f)) + (f == 3 ? (pt3 + at3) : make_float3(0.0f, 0.0f, 0.0f))) * simData.CubicSplineKern;// +(pterm[f] + aterm[f]) * simData.CubicSplineKern;
		//drift_vel[f] = fterm + (at2+pt2)*simData.CubicSplineKern;// ((f == 2 ? (pt2 + at2) : make_float3(0.0f, 0.0f, 0.0f)) + (f == 3 ? (pt3 + at3) : make_float3(0.0f, 0.0f, 0.0f)))* simData.CubicSplineKern;
		//buf._drift_velocity[muloff_i + f] = fterm[f] + (pterm[f] + aterm[f]) * simData.CubicSplineKern;
		//drift_vel[f] = buf._drift_velocity[muloff_i + f];
		buf._alpha[muloff_i + f] = alpha[f];
	}*/

	f0 = simData.tau * (simData.phase_density[0] - sum_mass_dens) * (-buf._force[i]);
	f1 = simData.tau * (simData.phase_density[1] - sum_mass_dens) * (-buf._force[i]);
	buf._drift_velocity[muloff_i] = f0 + (pt0 + at0) * simData.CubicSplineKern;
	buf._drift_velocity[muloff_i + 1] = f1 + (pt1 + at1) * simData.CubicSplineKern;
	buf._alpha[muloff_i] = alpha0;
	buf._alpha[muloff_i + 1] = alpha1;


	// For alpha transport
	buf._lambda[i] = 1.0f;
}

__device__ float* contributeDAlpha(int i, bufList buf, float3 pos, float3 vel, float3* drift_vel, float* alpha, int cell)
{
	register float d_alpha[MAX_PHASE_NUMBER] = { 0.0f };

	if (buf.num_particle_grid[cell] == 0) return d_alpha;

	float3 dist, K_xi, F_K_xi, nK_xi, nF_K_xi, u, rel_drift_vel;
	tensor K = buf._tensor_K[i], F = buf._tensor_F[i], nK, nF;
	float dsq, c, lambda, nalpha, pairwise, ngamma, gamma = buf._rest_mass[i] / buf._mix_rest_density[i];
	register float sum1 = 0.0f, sum2 = 0.0f;
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i || buf._type[pndx] == BOUND || buf._type[pndx] == RIGID) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2) * (-buf._rest_mass[pndx] * buf._inv_dens[pndx]);
			//ngamma = buf._rest_mass[pndx] / buf._mix_rest_density[pndx];
			//pairwise = 2.0f * ngamma / (gamma + ngamma);
			K_xi.x = K.tensor[0] * dist.x + K.tensor[1] * dist.y + K.tensor[2] * dist.z;
			K_xi.y = K.tensor[3] * dist.x + K.tensor[4] * dist.y + K.tensor[5] * dist.z;
			K_xi.z = K.tensor[6] * dist.x + K.tensor[7] * dist.y + K.tensor[8] * dist.z;
			F_K_xi.x = F.tensor[0] * K_xi.x + F.tensor[1] * K_xi.y + F.tensor[2] * K_xi.z;
			F_K_xi.y = F.tensor[3] * K_xi.x + F.tensor[4] * K_xi.y + F.tensor[5] * K_xi.z;
			F_K_xi.z = F.tensor[6] * K_xi.x + F.tensor[7] * K_xi.y + F.tensor[8] * K_xi.z;

			//neighbor
			nK = buf._tensor_K[pndx];
			nF = buf._tensor_F[pndx];
			nK_xi.x = nK.tensor[0] * dist.x + nK.tensor[1] * dist.y + nK.tensor[2] * dist.z;
			nK_xi.y = nK.tensor[3] * dist.x + nK.tensor[4] * dist.y + nK.tensor[5] * dist.z;
			nK_xi.z = nK.tensor[6] * dist.x + nK.tensor[7] * dist.y + nK.tensor[8] * dist.z;
			nF_K_xi.x = nF.tensor[0] * K_xi.x + nF.tensor[1] * K_xi.y + nF.tensor[2] * K_xi.z;
			nF_K_xi.y = nF.tensor[3] * K_xi.x + nF.tensor[4] * K_xi.y + nF.tensor[5] * K_xi.z;
			nF_K_xi.z = nF.tensor[6] * K_xi.x + nF.tensor[7] * K_xi.y + nF.tensor[8] * K_xi.z;
			F_K_xi = 0.5f * (F_K_xi + nF_K_xi);
			u = buf._mix_velocity[pndx] - vel;
			lambda = min(buf._lambda[i], buf._lambda[pndx]);
			int noffset = pndx * MAX_PHASE_NUMBER;
			for (int f = 0; f < simData.phase_number; f++)
			{
				nalpha = buf._alpha[noffset + f];
				rel_drift_vel = nalpha * buf._drift_velocity[noffset + f] + alpha[f] * drift_vel[f];
				d_alpha[f] += c * dot((nalpha - alpha[f]) * u + rel_drift_vel, F_K_xi) * lambda;
			}
			//if (i == 96000)
			//	printf("i: %d, nidx: %d, lambda: %f, alpha2: %f, d_alpha: %f, u: %f, %f, %f, F_K_xi: %f, %f, %f, c: %f\n", i, pndx, lambda, d_alpha[2], (buf._alpha[noffset + 2] - alpha[2]),
			//		u.x,u.y,u.z,F_K_xi.x,F_K_xi.y,F_K_xi.z,c);
		}
	}
	//d_alpha[0] = sum1;
	//d_alpha[1] = sum2;
	return d_alpha;
}

__device__ void contributeDAlphaChanged(int i, bufList buf, float3 pos, float3 vel, float3 drift_vel0, float3 drift_vel1, float alpha0, float alpha1, int cell, float& dalpha0, float& dalpha1)
{
	if (buf.num_particle_grid[cell] == 0) return;

	float3 dist, K_xi, F_K_xi, nK_xi, nF_K_xi, u, rel_drift_vel0, rel_drift_vel1;
	tensor K = buf._tensor_K[i], F = buf._tensor_F[i], nK, nF;
	float dsq, c, lambda, nalpha0, nalpha1, pairwise, ngamma, gamma = buf._rest_mass[i] / buf._mix_rest_density[i];
	register float sum0 = 0.0f, sum1 = 0.0f;
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		if (pndx == i || buf._type[pndx] == BOUND) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2) * (-buf._rest_mass[pndx] * buf._inv_dens[pndx]);
			//ngamma = buf._rest_mass[pndx] / buf._mix_rest_density[pndx];
			//pairwise = 2.0f * ngamma / (gamma + ngamma);
			K_xi.x = K.tensor[0] * dist.x + K.tensor[1] * dist.y + K.tensor[2] * dist.z;
			K_xi.y = K.tensor[3] * dist.x + K.tensor[4] * dist.y + K.tensor[5] * dist.z;
			K_xi.z = K.tensor[6] * dist.x + K.tensor[7] * dist.y + K.tensor[8] * dist.z;
			F_K_xi.x = F.tensor[0] * K_xi.x + F.tensor[1] * K_xi.y + F.tensor[2] * K_xi.z;
			F_K_xi.y = F.tensor[3] * K_xi.x + F.tensor[4] * K_xi.y + F.tensor[5] * K_xi.z;
			F_K_xi.z = F.tensor[6] * K_xi.x + F.tensor[7] * K_xi.y + F.tensor[8] * K_xi.z;

			//neighbor
			nK = buf._tensor_K[pndx];
			nF = buf._tensor_F[pndx];
			nK_xi.x = nK.tensor[0] * dist.x + nK.tensor[1] * dist.y + nK.tensor[2] * dist.z;
			nK_xi.y = nK.tensor[3] * dist.x + nK.tensor[4] * dist.y + nK.tensor[5] * dist.z;
			nK_xi.z = nK.tensor[6] * dist.x + nK.tensor[7] * dist.y + nK.tensor[8] * dist.z;
			nF_K_xi.x = nF.tensor[0] * K_xi.x + nF.tensor[1] * K_xi.y + nF.tensor[2] * K_xi.z;
			nF_K_xi.y = nF.tensor[3] * K_xi.x + nF.tensor[4] * K_xi.y + nF.tensor[5] * K_xi.z;
			nF_K_xi.z = nF.tensor[6] * K_xi.x + nF.tensor[7] * K_xi.y + nF.tensor[8] * K_xi.z;
			F_K_xi = 0.5f * (F_K_xi + nF_K_xi);
			u = buf._mix_velocity[pndx] - vel;
			lambda = min(buf._lambda[i], buf._lambda[pndx]);
			int noffset = pndx * MAX_PHASE_NUMBER;/*
			for (int f = 0; f < simData.phase_number; f++)
			{
				nalpha = buf._alpha[noffset + f];
				rel_drift_vel = nalpha * buf._drift_velocity[noffset + f] + alpha[f] * drift_vel[f];
				//d_alpha[f] += c * dot((nalpha - alpha[f]) * u + rel_drift_vel, F_K_xi) * lambda;
				if (f == 0)
					sum1 += c * dot((nalpha - alpha[f]) * u + rel_drift_vel, F_K_xi) * lambda;
				if (f == 1)
					sum2 += c * dot((nalpha - alpha[f]) * u + rel_drift_vel, F_K_xi) * lambda;
			}*/
			nalpha0 = buf._alpha[noffset + 0]; nalpha1 = buf._alpha[noffset + 1];
			rel_drift_vel0 = nalpha0 * buf._drift_velocity[noffset] + alpha0 * drift_vel0;
			rel_drift_vel1 = nalpha1 * buf._drift_velocity[noffset] + alpha1 * drift_vel1;
			sum0 += c * dot((nalpha0 - alpha0) * u + rel_drift_vel0, F_K_xi) * lambda;
			sum1 += c * dot((nalpha0 - alpha0) * u + rel_drift_vel0, F_K_xi) * lambda;

			//if (i == 96000)
			//	printf("i: %d, nidx: %d, lambda: %f, alpha2: %f, d_alpha: %f, u: %f, %f, %f, F_K_xi: %f, %f, %f, c: %f\n", i, pndx, lambda, d_alpha[2], (buf._alpha[noffset + 2] - alpha[2]),
			//		u.x,u.y,u.z,F_K_xi.x,F_K_xi.y,F_K_xi.z,c);
		}
	}
	dalpha0 = sum0;
	dalpha1 = sum1;
	//return d_alpha;
}

inline __device__ float my_clamp(float a)
{
	return a > 0.0f ? (a < 1.0f ? a : 1.0f) : 0;
}
__global__ void computeDelAlpha(bufList buf, int numParticles, float time_step)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Initialize
	float3 pos = buf._position[i], vel = buf._mix_velocity[i];
	float3 drift_vel[MAX_PHASE_NUMBER];
	float alpha[MAX_PHASE_NUMBER], d_alpha[MAX_PHASE_NUMBER];
	const int muloff_i = i * MAX_PHASE_NUMBER;
	for (int f = 0; f < simData.phase_number; f++)
	{
		alpha[f] = buf._alpha[muloff_i + f];
		d_alpha[f] = 0.0f;
		drift_vel[f] = buf._drift_velocity[muloff_i + f];
	}

	// Sum divergence term
	float* contribute_ptr;
	for (int c = 0; c < 27; c++) {
		contribute_ptr = contributeDAlpha(i, buf, pos, vel, drift_vel, alpha, gc + simData.grid_search_offset[c]);
		for (int f = 0; f < simData.phase_number; f++) d_alpha[f] += -contribute_ptr[f];
	}

	//for (int c = 0; c < 27; c++) {
	//	float dalpha0 = 0.0f, float dalpha1 = 0.0f;
	//	contributeDAlphaChanged(i, buf, pos, vel, drift_vel[0],drift_vel[1], alpha[0], alpha[1], gc + simData.grid_search_offset[c], dalpha0, dalpha1);
	//	d_alpha[0] = dalpha0;
	//	d_alpha[1] = dalpha1;
	//}

	__syncthreads();

	for (int f = 0; f < simData.phase_number; f++)
	{
		d_alpha[f] *= time_step * simData.CubicSplineKern;
		buf._delta_alpha[muloff_i + f] = d_alpha[f];
		buf._alpha_advanced[muloff_i + f] = my_clamp(d_alpha[f] + alpha[f]);
	}
	//if (i % 3000 == 0)
	//	printf("i: %d, d_alpha: %f, %f, alpha: %f, %f\n",i, buf._delta_alpha[muloff_i + 2], buf._delta_alpha[muloff_i + 3], buf._alpha_advanced[muloff_i + 2], buf._alpha_advanced[muloff_i + 3]);
}

__global__ void computeLambda(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND) return;
	if (buf._active[i] == false) return;

	// Initialize
	float alpha[MAX_PHASE_NUMBER], d_alpha[MAX_PHASE_NUMBER], new_alpha[MAX_PHASE_NUMBER];
	const int muloff_i = i * MAX_PHASE_NUMBER;
	for (int f = 0; f < simData.phase_number; f++)
	{
		alpha[f] = buf._alpha[muloff_i + f];
		d_alpha[f] = buf._delta_alpha[muloff_i + f];
		new_alpha[f] = buf._alpha_advanced[muloff_i + f];
	}

	float temp_coef = 0.0f, coef = 10000.0f;
	for (int f = 0; f < simData.phase_number; f++)
	{
		if (new_alpha[f] < -0.001f) break;
		if (f == MAX_PHASE_NUMBER - 1) return;    // no negative value, return
	}

	bool negative[MAX_PHASE_NUMBER];
	for (int f = 0; f < simData.phase_number; f++)
	{
		if (new_alpha[f] < -0.001f) {
			negative[f] = true;
			temp_coef = alpha[f] / (-d_alpha[f]);
			coef = min(coef, temp_coef);
		}
		else negative[f] = false;
	}
	buf._lambda[i] *= max(coef, 0.0f);
}

__global__ void computeCorrection(bufList buf, int numParticles, float time_step)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	// Initialize
	float d_alpha[MAX_PHASE_NUMBER] = { 0.0f }, tmp_alpha[MAX_PHASE_NUMBER];
	register float alpha_sum = 0.0f, mrdens = 0.0f, mvis = 0.0f, rmass = 0.0f;
	float volume = buf._rest_mass[i] / buf._mix_rest_density[i];
	const int muloff_i = i * MAX_PHASE_NUMBER;
	for (int f = 0; f < simData.phase_number; f++)
	{
		alpha_sum += buf._alpha_advanced[muloff_i + f];
	}

	// raw alpha
	float dmass_sum = 0.0f;
	alpha_sum = 1.0f / alpha_sum;
	for (int f = 0; f < simData.phase_number; f++)
	{
		tmp_alpha[f] = buf._alpha_advanced[muloff_i + f] * alpha_sum;
		d_alpha[f] = tmp_alpha[f] - buf._alpha[muloff_i + f];

		mrdens += buf._alpha_advanced[muloff_i + f] * simData.phase_density[f];
		mvis += buf._alpha_advanced[muloff_i + f] * simData.phase_visc[f];
		rmass += buf._alpha_advanced[muloff_i + f] * simData.phase_density[f] * volume;

		dmass_sum += buf._delta_alpha[muloff_i + f] * simData.phase_density[f] * volume;
	}

	// pressure correction
	float d_p = 0.0f, rel_dens = pow(buf._mix_density[i] / mrdens, 2);
	for (int f = 0; f < simData.phase_number; f++) d_p -= 0.5f * simData.gas_constant * simData.phase_density[f] * (1.0f + rel_dens) * d_alpha[f];
	buf._mix_pressure[i] += d_p;

	// mu & rho_m correction
	buf._mix_rest_density[i] = mrdens;
	buf._viscosity[i] = mvis;
	//buf._delta_mass[i] = rmass - buf._rest_mass[i];
	buf._rest_mass[i] += dmass_sum;
}

__global__ void normalizeAlpha(bufList buf, int numParticles, float time_step)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;

	// Initialize
	register float alpha[MAX_PHASE_NUMBER] = { 0.0f }, new_alpha[MAX_PHASE_NUMBER], m_frac[MAX_PHASE_NUMBER] = { 0.0f };
	register float alpha_sum = 0.0f, mf_sum = 0.0f, mfdrho_sum = 0.0f, inv_dens = 1.0f / buf._mix_rest_density[i];
	float volume = buf._rest_mass[i] / buf._mix_rest_density[i];
	const int muloff_i = i * MAX_PHASE_NUMBER;

	// recompute mass fraction
	for (int f = 0; f < simData.phase_number; f++)
	{
		m_frac[f] = buf._alpha_advanced[muloff_i + f] * simData.phase_density[f] * inv_dens;
		mfdrho_sum += m_frac[f] / simData.phase_density[f];
		mf_sum += m_frac[f];
	}

	// recompute alpha
	for (int f = 0; f < simData.phase_number; f++)
	{
		alpha[f] = buf._alpha_advanced[muloff_i + f];
		new_alpha[f] = (m_frac[f] / simData.phase_density[f]) / mfdrho_sum;
		buf._alpha_advanced[muloff_i + f] = new_alpha[f];
		alpha_sum += new_alpha[f];
	}

	buf._particle_radius[i] = 0.5f * pow(volume, 1.0f / 3.0f);
}

__device__ float3 contributeForce(int i, bufList buf, float3 pos, float3 vel, tensor& K, int cell)
{
	if (buf.num_particle_grid[cell] == 0) return make_float3(0.0f, 0.0f, 0.0f);

	float3 dist, sum = make_float3(0.0f, 0.0f, 0.0f), K_xi, u;
	float3 pforce = make_float3(0.0f, 0.0f, 0.0f), vforce = make_float3(0.0f, 0.0f, 0.0f);
	float dsq, c, pVol = 1.0f / buf._mix_rest_density[i];
	register float r2 = simData.smooth_radius * simData.smooth_radius;

	int clast = buf.grid_offset[cell] + buf.num_particle_grid[cell];

	for (int cndx = buf.grid_offset[cell]; cndx < clast; cndx++) {
		int pndx = buf.sorted_grid_map[cndx];

		//if (pndx == i && buf._type[pndx] == BOUND) continue;
		if (pndx == i) continue;

		dist = pos - buf._position[pndx];
		dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
		if (dsq < r2 && dsq>0.0) {
			c = cubicKernel(dsq, r2);
			K_xi.x = K.tensor[0] * dist.x + K.tensor[1] * dist.y + K.tensor[2] * dist.z;
			K_xi.y = K.tensor[3] * dist.x + K.tensor[4] * dist.y + K.tensor[5] * dist.z;
			K_xi.z = K.tensor[6] * dist.x + K.tensor[7] * dist.y + K.tensor[8] * dist.z;
			if (buf._type[pndx] == FLUID) {
				c *= buf._rest_mass[pndx] * buf._inv_dens[pndx];
				u = buf._mix_velocity[pndx] - vel;
				pforce = 0.5f * c * (buf._mix_pressure[i] + buf._mix_pressure[pndx]) * K_xi * pVol;
				vforce = 2.0f * c * (buf._viscosity[i] + buf._viscosity[pndx]) * u * pVol * dot(dist, K_xi) / dsq;
			}
			else
			{
				if (buf._active[pndx] == true) {
					u = buf._mix_velocity[pndx] - vel;
					c *= buf._bound_phi[pndx] * buf._mix_rest_density[i] * buf._inv_dens[i];
					pforce = c * pVol * buf._mix_pressure[i] * K_xi;
					vforce = 2.0f * c * simData.viscosity * u * pVol * dot(dist, K_xi) / dsq;
				}
				else
				{
					pforce = make_float3(0.0f, 0.0f, 0.0f);
					vforce = make_float3(0.0f, 0.0f, 0.0f);
				}
			}
			sum += pforce + vforce;
		}
	}
	return sum;
}
__global__ void computeForce(bufList buf, int numParticles, float time_step)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	if (buf._type[i] == BOUND || buf._type[i] == RIGID) return;
	if (buf._active[i] == false) return;
	//if (buf._alpha_sum[i] < 0.01f) return;

	// Get search cell
	uint gc = buf.particle_grid_cellindex[i];
	if (gc == GRID_UNDEF) return;

	// Sum force
	float3 pos = buf._position[i], vel = buf._mix_velocity[i];
	register float3 force = make_float3(0.0f, 0.0f, 0.0f);
	tensor K = buf._tensor_K[i];
	const int muloff_i = i * MAX_PHASE_NUMBER;
	for (int c = 0; c < 27; c++) {
		force += contributeForce(i, buf, pos, vel, K, gc + simData.grid_search_offset[c]);
	}
	__syncthreads();
	float3 phforce = make_float3(0.0f, 0.0f, 0.0f);
	//for (int f = 0; f < simData.phase_number; f++) {
	//	phforce += simData.phase_density[f] * (1.0f / buf._mix_rest_density[i]) * buf._delta_alpha[muloff_i + f] / time_step * (buf._drift_velocity[muloff_i + f] + vel);
	//}
	buf._force[i] = force * simData.CubicSplineKern + phforce;
}

__global__ void advanceParticles(bufList buf, int numParticles, float time_step, int frame)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
    float3 color_water = make_float3(0.1,0.2,1.0);
    float3 color_sand  = make_float3(0.8,0.6,0.1);
    float3 color_iron  = make_float3(0.7,0.7,0.8);
    float3 phase_color[5] = { color_water, color_sand, color_iron };
    float3 init_color     = make_float3(0.0f, 0.0f, 0.0f);
	
    
	if (buf._type[i] == BOUND){
        buf._render_pos[i] = make_float3(0.0f, 0.0f, 0.0f);
        buf._color[i]      = make_float3(1.0f, 1.0f, 1.0f);
        return;
	}
	for (int fcount = 0; fcount < simData.phase_number; fcount++) {
        init_color += buf._alpha_advanced[i * MAX_PHASE_NUMBER + fcount] * phase_color[fcount];
	}
    buf._color[i] = init_color;
	/*if (i % 5000 == 0)
        printf("color: %f, %f, %f\n", init_color.x, init_color.y, init_color.z);*/
	if (buf._active[i] == false) return;

	if (buf._type[i] == RIGID)
	{
        ///if (i % 2000 == 0)
        ///    printf("%d, pos: %f\n", i, buf._position[i].x);
		if (frame< 180)
		{
			register float3 pos = buf._position[i], vel = buf._mix_velocity[i], a = simData.gravity;
			vel += time_step * a;
			pos += time_step * vel;
			buf._mix_velocity[i] = vel;
			buf._position[i] = pos;
            buf._render_pos[i]   = pos;
		}
        if (frame > 180 && frame < 800)
		{
			register float3 pos = buf._position[i], vel = make_float3(2.0f, 0.0f, 2.0f), a = make_float3(20.0, 0.0f, 20.5f);
			vel += time_step * a;
			pos += time_step * vel;
			buf._mix_velocity[i] = vel;
			buf._position[i] = pos;
            buf._render_pos[i]   = pos;
		}
		if (frame > 800)
		{

			return;
		}

		//if (i % 5000 == 0)
		//	printf("Rigid: %d, pos: %f, %f, %f\n", i, pos.x, pos.y, pos.z);
		return;
	}

	register float3 pos = buf._position[i], vel = buf._mix_velocity[i], a = buf._force[i] + simData.gravity+buf._external_force[i];
	vel += time_step * a;
	pos += time_step * vel;
	buf._position[i] = pos;
	buf._mix_velocity[i] = vel;
    buf._render_pos[i]   = pos;
}

__global__ void test(bufList buf, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= numParticles) return;
	
	if (i % 5000 == 0)
		printf("idx: %d, pos: %f, %f, %f\n", i, buf._position[i].x, buf._position[i].y, buf._position[i].z);
}

// ======================================== //