#define _USE_MATH_DEFINES  // for C++

#include "pbd_elastic_kernel.cuh"

#include <stdio.h>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include <thrust/tuple.h>
#include <math_constants.h>

#include "pbd_elastic_params.hpp"
#include "helper_math.hpp"


using namespace cooperative_groups;
namespace Physika {
	
	struct ElasticSolverParams;
	__constant__ ElasticSolverParams elastic_params;

   __host__ void setSimulationParams(ElasticSolverParams* p)
   {
       cudaError_t cudaStatus  = cudaMemcpyToSymbol(elastic_params, p, sizeof(ElasticSolverParams));
       //printf("volume %f \n", elastic_params.m_volume);
       if (cudaStatus != cudaSuccess){
           printf("fail1\n");
       }
   }

   __device__ mat3 outerProduct(const float3& A, const float3& B) {

		mat3 C;
		C.value[0] = A.x * B.x;
		C.value[1] = A.x * B.y;
		C.value[2] = A.x * B.z;
		C.value[3] = A.y * B.x;
		C.value[4] = A.y * B.y;
		C.value[5] = A.y * B.z;
		C.value[6] = A.z * B.x;
		C.value[7] = A.z * B.y;
		C.value[8] = A.z * B.z;
		return C;
	}

   __device__ int3 calcGridPosKernel(float3 p, float3 m_world_orgin,float3 m_cell_size)
    {
        int3 gridPos;
        gridPos.x = floor((p.x - m_world_orgin.x) / m_cell_size.x);
        gridPos.y = floor((p.y - m_world_orgin.y) / m_cell_size.y);
        gridPos.z = floor((p.z - m_world_orgin.z) / m_cell_size.z);
        return gridPos;
    }

    __device__ unsigned int calcGridHashKernel(int3 gridPos, uint3 m_grid_num)
    {
        gridPos.x = gridPos.x & (m_grid_num.x - 1);
        gridPos.y = gridPos.y & (m_grid_num.y - 1);
        gridPos.z = gridPos.z & (m_grid_num.z - 1);
        return gridPos.z * elastic_params.m_grid_num.x * elastic_params.m_grid_num.y + gridPos.y * elastic_params.m_grid_num.x + gridPos.x;
    }

    __global__ void calcParticlesHashKernel(
        unsigned int* gridParticleHash,
        float3*       pos,
        float3        m_world_orgin,
        float3		  m_cell_size,
		uint3		  m_grid_num,
        unsigned int  numParticles)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
            return;

        volatile float3 curPos    = pos[index];
        int3            gridPos   = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z), m_world_orgin, m_cell_size);
        unsigned int    hashValue = calcGridHashKernel(gridPos, m_grid_num);
        gridParticleHash[index]   = hashValue;
    }

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

    __device__ float kernelPoly6(float3 r, float m_sph_radius)
    {
		float sph_radius = m_sph_radius;
		float radius = length(r);
		float value = 0.f;
		if (radius <= sph_radius) {
			float l = (1 / sph_radius - radius * radius / (sph_radius * sph_radius * sph_radius));
			value = 315 / (64 * M_PI) * l * l * l;
		}
		return value;
	}

	__device__ float3 gradientKenrelPoly6(float3 r, float m_sph_radius)
    {
		float sph_radius = m_sph_radius;
		float radius = length(r);
		float3 value = make_float3(0.f, 0.f, 0.f);
		if ( radius <= sph_radius) {
            float deltex = powf(sph_radius, 2.0f) - powf(radius, 2.0f);
            float l      = -945 / (32 * M_PI * powf(sph_radius, 9.0f)) * pow(deltex, 2.0f);
			value = l * r;
		}
		return value;
	}

    __global__ void calSumKernelPolyCuda(
		float3* m_device_initial_pos,
		float* m_device_sum,
		unsigned int* grid,
		unsigned int* gridStart,
        float m_volume,
        float m_sph_radius,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;
		float3 initial_pos_i = m_device_initial_pos[index];
		float sum = 0.f;
 		int start = gridStart[index];
 		int end = gridStart[index + 1];
 #pragma unroll 25
 		for (int i = start; i < end; i++) {
 			int j = grid[i];
			if (j == index)
				continue;
 			float3 m_initial_pos_j = m_device_initial_pos[j];
 			float3 m_delte_x = initial_pos_i - m_initial_pos_j;
            sum += m_volume * kernelPoly6(m_delte_x, m_sph_radius);
 		}
		m_device_sum[index] = sum;
	}

	__global__ void calCorrGradientKenrelSumCuda(
        mat3*         m_device_L,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float3*       m_device_sph_kernel,
        float3*       m_device_sph_kernel_inv,
        float3*       m_device_external_force,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
		float		  m_sph_radius,
        int           m_num_particle
	)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= m_num_particle)
            return;

        int start = gridStart[index];
        int end   = gridStart[index + 1];

#pragma unroll 30
        for (int i = start; i < end; i++)
        {
            int j = grid[i];
            if (j == index)
                continue;

            m_device_sph_kernel[i]     = m_device_L[index] * CorrGradientKenrelPoly6(m_device_initial_pos, m_device_y, m_device_sum, m_sph_radius, index, j);
            m_device_sph_kernel_inv[i] = m_device_L[j] * CorrGradientKenrelPoly6(m_device_initial_pos, m_device_y, m_device_sum, m_sph_radius, j, index);
            /*if (index == 399)
            {
                printf("sph_kernel %f \n", m_device_sph_kernel[i]);
            }*/
		}

        //m_device_external_force[index] = make_float3(0.0f , 0.0f, 0.0f);
    }

    __global__ void calYCuda(
		float3* m_device_initial_pos,
		float3* m_deivce_y,
		unsigned int* grid,
		unsigned int* gridStart,
        float m_volume,
        float m_sph_radius,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		float3 initial_pos_i = m_device_initial_pos[index];	
		float max = m_sph_radius;

		float sum = 0.f;
		float3 gradient_sum = make_float3(0.f,0.f,0.f); 
        int start = gridStart[index];
        int end = gridStart[index + 1];
#pragma unroll 25
        for(int i = start; i < end; i++ ) {
            int j = grid[i];
            if (j == index)
                continue;
            float3 initial_pos_j = m_device_initial_pos[j];
            float3 deltex = initial_pos_i - initial_pos_j;
            sum += m_volume * kernelPoly6(deltex, m_sph_radius);
            gradient_sum += m_volume * gradientKenrelPoly6(deltex, m_sph_radius);
            /*if (index == 2501)
            {	
				printf("dt: %f\n", elastic_params.lame_first);
				printf("volume: %f \n", elastic_params.m_volume);
                printf("deltex:%f,%f,%f \n", deltex.x, deltex.y, deltex.z);
                printf("sum_y_kernel: %f \n", sum);
			}*/
        }

		m_deivce_y[index] = gradient_sum / sum;
     /*   if (index == 2501) {
			printf("gradient_sum:%f,%f,%f \n", gradient_sum.x, gradient_sum.y, gradient_sum.z);
			printf("sum_y: %f \n", sum);
			printf("y:%f,%f,%f \n", m_deivce_y[index].x, m_deivce_y[index].y, m_deivce_y[index].z);
		}*/
	}


    __global__ void calLCuda(
		float3* m_device_initial_pos,
		float3* m_device_y,
		float* m_device_sum,
		mat3* m_deivce_L,
		unsigned int* grid,
		unsigned int* gridStart,
		float m_sph_radius,
		float m_volume,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;
		float3 initial_pos_i = m_device_initial_pos[index];
		mat3 sum;
 		int start = gridStart[index];
 		int end = gridStart[index + 1];
		float3 m_y = m_device_y[index];
		float m_sum = m_device_sum[index];
 #pragma unroll 25
 		for (int i = start; i < end; i++) {
 			int j = grid[i];
 			if (j == index)
 				continue;
 			mat3 t;
 			float3 initial_pos_j = m_device_initial_pos[j];
 			float3 delte_x_ij = initial_pos_i - initial_pos_j;
 			float3 delte_x_ji = -delte_x_ij;
			t = outerProduct(CorrGradientKenrelPoly6(
				m_device_initial_pos,
				m_device_y,
				m_device_sum,
				m_sph_radius,
				index,
				j) , delte_x_ji);
 			sum += t;
 		}
		sum *= m_volume;
		sum = sum.inverse();
		m_deivce_L[index] = sum;
         //if (index == 5) {
        	////printf("volume :%f \n", elastic_params.m_volume);
        	//printf("index: %d ,LLL :%f,%f,%f,%f,%f,%f,%f,%f,%f \n", index, sum.value[0], sum.value[1], sum.value[2], sum.value[3], sum.value[4], sum.value[5], sum.value[6], sum.value[7], sum.value[8]);
         //}
	}

    __device__ float3 CorrGradientKenrelPoly6(
		float3* m_device_initial_pos,
		float3* m_device_y,
		float* m_device_sum,
		float m_sph_radius,
		int index,
		int j
	) {

		float3 initial_pos_i = m_device_initial_pos[index];
		float3 initial_pos_j = m_device_initial_pos[j];
		float3 delte_x = initial_pos_i - initial_pos_j;
		float3 initial_y = m_device_y[index];
		float sum = m_device_sum[index];
        float3 gradient_kernel = gradientKenrelPoly6(delte_x, m_sph_radius);
		float3 value = (gradient_kernel - initial_y) / sum;
		return value;
	}

    __global__ void calGradientDeformationCuda(
		float3* m_device_sph_kernel,
		float3* m_device_next_pos,
		float3* m_device_initial_pos,
		float3* m_device_y,
		float* m_device_sum,
		mat3* m_device_L,
		mat3* m_device_F,
		unsigned int* grid,
		unsigned int* gridStart,
		float m_volume,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;
	
		float3 next_pos_i = m_device_next_pos[index];
		float3 initial_pos_i = m_device_initial_pos[index];
		mat3 sum;
        int start = gridStart[index];	
        int end = gridStart[index + 1];
        #pragma unroll 30
        for(int i = start; i < end; i++ ) {
            int j = grid[i];
            if (j == index)
                continue;
            float3 next_pos_j = m_device_next_pos[j];
            float3 delte_x = next_pos_j - next_pos_i;
            float3 kernel = m_device_sph_kernel[i];
            sum += outerProduct(delte_x, kernel) ;
        }
        m_device_F[index] = sum * m_volume;
         //if (index == 399) {
        	////printf("volume :%f \n", elastic_params.m_volume);
        	//printf("F : %f, %f, %f, %f, %f, %f, %f, %f, %f \n", m_device_F[index].value[0], m_device_F[index].value[1],m_device_F[index].value[2],m_device_F[index].value[3],m_device_F[index].value[4], m_device_F[index].value[5],m_device_F[index].value[6],m_device_F[index].value[7],m_device_F[index].value[8]);
         //}
	}

    __global__ void calRCuda(
		mat3* m_device_F,
		mat3* m_device_R,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		mat3 S, R;
		m_device_F[index].polarDecomposition(R, S);
		m_device_R[index] = R;
        /*if (index == 399) {
			printf("R :%f,%f,%f,%f,%f,%f,%f,%f,%f \n", m_device_R[index].value[0], m_device_R[index].value[1], m_device_R[index].value[2], m_device_R[index].value[3], \
				m_device_R[index].value[4], m_device_R[index].value[5], m_device_R[index].value[6], m_device_R[index].value[7], m_device_R[index].value[8]);
		}*/
	}

    __global__ void calPCuda(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		mat3* m_device_P,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;
		
		mat3 tranpose = m_device_F[index].transpose();
		tranpose = tranpose.inverse();
		float det = abs(m_device_F[index].determinant());

		mat3 anisotropy_P;
        //float3 stiffness = make_float3(0.35f, 1.f, 1.f);
		float3 b1 = make_float3(1.f, 0.f, 0.f);
		float3 b2 = make_float3(0.f, 1.f, 0.f);
		float3 b3 = make_float3(0.f, 0.f, 1.f);

		float beta1 = length(m_device_F[index] * b1);
		float beta2 = length(m_device_F[index] * b2);
		float beta3 = length(m_device_F[index] * b3);

		anisotropy_P = m_device_F[index] * outerProduct(b1, b1) * beta1 * (1 - 1 / (m_anisotropy.x * m_anisotropy.x)) + \
						m_device_F[index] * outerProduct(b2, b2) * beta2 * (1 - 1 / (m_anisotropy.y * m_anisotropy.y)) + \
						m_device_F[index] * outerProduct(b3, b3) * beta3 * (1 - 1 / (m_anisotropy.z * m_anisotropy.z));
		

		mat3 isotropy_P_neo_hookean = m_device_F[index] * elastic_params.lame_second - tranpose * elastic_params.lame_second + tranpose * __logf(det) * elastic_params.lame_first;
		mat3 I ;
		mat3 isotropy_P_corotated = (m_device_F[index] - m_device_R[index]) * elastic_params.lame_second * 2 + m_device_R[index] * (m_device_R[index].transpose() * m_device_F[index] - I.identity()).trace() * elastic_params.lame_first;
        m_device_P[index]         = isotropy_P_neo_hookean + isotropy_P_corotated + anisotropy_P;
	}

    	__global__ void calEnergyCuda(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		float* m_device_energy,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		mat3 tranpose = m_device_F[index].transpose();
		float det = abs(m_device_F[index].determinant());
		mat3 temp = tranpose * m_device_F[index];

		//float3 stiffness = make_float3(0.35f, 1.f, 1.f);
		float3 b1 = make_float3(1.f, 0.f, 0.f);
		float3 b2 = make_float3(0.f, 1.f, 0.f);
		float3 b3 = make_float3(0.f, 0.f, 1.f);

		float beta1 = length(m_device_F[index] * b1);
		float beta2 = length(m_device_F[index] * b2);
		float beta3 = length(m_device_F[index] * b3);

		float anisotropy_energy = ((m_anisotropy.x * m_anisotropy.x - 1) / 2 - __logf(m_anisotropy.x)) * beta1 + \
			((m_anisotropy.y * m_anisotropy.y - 1) / 2 - __logf(m_anisotropy.y)) * beta2 + \
			((m_anisotropy.z * m_anisotropy.z - 1) / 2 - __logf(m_anisotropy.z)) * beta3;
		mat3 I;
		float temp1 = (m_device_F[index] - m_device_R[index]).frobeniusNorm();
		float temp2 = (m_device_R[index].transpose() * m_device_F[index] - I.identity()).trace();
        float neo_energy = abs(elastic_params.lame_second / 2 * (temp.trace() - 3) - elastic_params.lame_second * __logf(det) + elastic_params.lame_first / 2 * __logf(det) * __logf(det));
        float cor_energy = temp1 * temp1 * elastic_params.lame_second + elastic_params.lame_first / 2 * temp2 * temp2;

		m_device_energy[index] = neo_energy + cor_energy + anisotropy_energy;
	}

    __global__ void calLMCuda(
        float* m_device_energy,
        float* m_device_lm,
        int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		m_device_lm[index] = 0.f;
		
	}
	
	
    __global__ void solveDelteLagrangeMultiplierCuda(
		float3* m_device_sph_kernel,
		float* m_device_energy,
		float* m_device_lm,
		float* m_device_delte_lm,
		float* m_device_sum,
		float3* m_device_initial_pos,
		float3* m_device_y,
		mat3* m_device_L,
		mat3* m_device_P,
		unsigned int* grid,
		unsigned int* gridStart,
        float m_h,
		float m_volume,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		float sum = 0.f;
		float aphla = 1 / (m_h * m_h);
		mat3 m_L = m_device_L[index];
		mat3 m_P = m_device_P[index];

			
        float3 kernel_sum = make_float3(0.f, 0.f, 0.f);
        int start = gridStart[index];
        int end = gridStart[index + 1];
        #pragma unroll 30
        for(int i = start; i < end; i++ ) {
            int j = grid[i];
            if (j == index)
                continue;
            float3 kernel = m_device_sph_kernel[i];
            kernel *= m_volume;
            kernel = m_P * kernel;
            kernel_sum += kernel;
            sum += dot(kernel, kernel);
        }
		
		m_device_delte_lm[index] = -(m_device_energy[index] + aphla * m_device_lm[index]) / (sum + aphla);
       /* if (index == 399) {
            printf("detle_lm :%f \n", m_device_delte_lm[index]);
        }*/
	}


    __global__ void solveDelteDistanceCuda(
		float3* m_device_sph_kernel,
		float3* m_device_sph_kernel_inv,
		mat3* m_device_P,
		mat3* m_device_L,
		float* m_device_delte_lm,
		float* m_device_sum,
		float3* m_device_delte_x,
		float3* m_device_initial_pos,
		float3* m_device_y,
		unsigned int* grid,
		unsigned int* gridStart,
		float m_volume,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;

		mat3 p_i =  m_device_P[index];
		float delte_lm_i = m_device_delte_lm[index];
		float3 sum = make_float3(0.f,0.f,0.f);
		int start = gridStart[index];
		int end = gridStart[index + 1];

#pragma unroll 30
		for(int i = start; i < end; i++ ) {
			int j = grid[i];
			if (j == index)
				continue;
			mat3 m_L_j = m_device_L[j];
			mat3 p_j = m_device_P[j];

			float delte_lm_j = m_device_delte_lm[j];
			float3 kernel_ij = m_device_sph_kernel[i];
			float3 kernel_ji = m_device_sph_kernel_inv[i];
			kernel_ij = p_i * kernel_ij * delte_lm_i;
			kernel_ji = p_j * kernel_ji * delte_lm_j;
			sum += kernel_ji - kernel_ij;
		}
        m_device_delte_x[index] = sum * m_volume; 
		/*if (index == 399) {
            printf("detle_x :%f,%f,%f \n", sum.x, sum.y, sum.z);
        }*/
	}

    __global__ void updateDeltxlmCuda(
		float3* m_deviceDeltex,
		float3* m_deviceNextPos,
		float* m_device_delte_lm,
		float* m_device_lm,
		int m_num_particle
	) {
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= m_num_particle) return;
		m_deviceNextPos[index] += m_deviceDeltex[index] / 6;
		m_device_lm[index] += m_device_delte_lm[index] / 2;
	}

    __global__ void updateCuda(
        float3* m_deviceNextPos,
        float3* m_devicePos,
        float3* m_deviceVel,
        float*  m_device_phase,
        int m_numParticle,
        float deltaT
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m_numParticle) return;
        if (m_device_phase[i] == 3.f)
            return;

        /*if (i % 2250 < 2205)
        {*/
            m_deviceVel[i] = (m_deviceNextPos[i] - m_devicePos[i]) / deltaT;
            m_devicePos[i] = m_deviceNextPos[i];
        //}
    }

    __global__ void advectCuda(
        float3* m_deviceNextPos,
        float3* m_devicePos,
        float3* m_deviceVel,
        float3* m_deviceDeltex,
        float3* m_deviceExternalForce,
        float*  m_device_phase,
        float3 acc,
        int m_numParticle,
        float deltaT
    ) {
		const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= m_numParticle) return;
        if (m_device_phase[i] == 3.f)
            return;

		m_deviceDeltex[i] = make_float3(0.0f,0.0f,0.0f);
        float3 force      = make_float3(0.f,-30.f, 0.f);
        
        /*if (i % 2250 < 45)
        {
            m_deviceVel[i] += acc * deltaT + force * deltaT;
        }
        else*/
        acc += m_deviceExternalForce[i] / elastic_params.m_mass;
        m_deviceVel[i] += acc * deltaT;

        m_deviceNextPos[i] = m_devicePos[i] + m_deviceVel[i] * deltaT;
	}

	__global__ void stretchCuda(
        float3* m_device_pos,
        float3* m_device_next_pos,
        float3* m_device_vel,
        float3* m_device_deltex,
        int     frame,
        int     m_num_particle)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= m_num_particle)
            return;

        /*if (index < 2500) {
            m_device_vel[index] = make_float3(-0.5f, 0.f, 0.f);
        }*/
        if (index >= m_num_particle - 2500)
        {
            m_device_vel[index]      = make_float3(20.f, 0.f, 0.f);
            m_device_next_pos[index] = m_device_pos[index] + m_device_vel[index];
        }
        else
            m_device_next_pos[index] = m_device_pos[index] + m_device_vel[index];
        
    }

    __global__ void solveBoundaryConstraintCuda(
			float3* m_deviceNextPos,
			float3* m_deviceVel,
			float3 LB,
			float3 RT,
			float radius
		) {
		const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (m_deviceNextPos[i].x < LB.x)
			m_deviceNextPos[i].x = LB.x ;
		if (m_deviceNextPos[i].x > RT.x)
			m_deviceNextPos[i].x = RT.x ;
		if (m_deviceNextPos[i].y < LB.y) {
			m_deviceNextPos[i].y = LB.y;
			//m_deviceVel[i] = make_float3(0.0, 0.0, 0.0);
		}
		if (m_deviceNextPos[i].y > RT.y)
			m_deviceNextPos[i].y = RT.y;
		if (m_deviceNextPos[i].z < LB.z)
			m_deviceNextPos[i].z = LB.z ;
		if (m_deviceNextPos[i].z > RT.z)
			m_deviceNextPos[i].z = RT.z ;
	}

	__global__ void calCollisionDeltexCuda(
        float3*       m_deviceInitialPos,
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        float*        m_device_phase,
        unsigned int* m_device_index,
        unsigned int* cellStart,
        unsigned int* cellEnd,
        unsigned int* gridParticleHash,
		float		  m_sph_radius,
		float3		  m_world_orgin,
		float3		  m_celll_size,
		uint3		  m_grid_num,
        int           m_num_particle)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= m_num_particle)
            return;

        int    initial_index = m_device_index[index];
        float3 curPos        = m_deviceNextPos[initial_index];
        int3   gridPos       = calcGridPosKernel(curPos, m_world_orgin, m_celll_size);
        float3 sumDeltx      = make_float3(0.0f, 0.0f, 0.0f);
        int    sum           = 0;

#pragma unroll 3
        for (int z = -1; z <= 1; ++z)
        {
#pragma unroll 3
            for (int y = -1; y <= 1; ++y)
            {
#pragma unroll 3
                for (int x = -1; x <= 1; ++x)
                {
                    int3         neighbourGridPos   = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
                    unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos, m_grid_num);
                    unsigned int startIndex         = cellStart[neighbourGridIndex];
                    if (startIndex == 0xffffffff)
                        continue;
                    unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 20
                    for (unsigned int indexj = startIndex; indexj < endIndex; indexj++)
                    {
                        if (indexj == index)
                            continue;
                        int    initial_index_j = m_device_index[indexj];
                        float3 rest            = m_deviceInitialPos[initial_index] - m_deviceInitialPos[initial_index_j];
                        float3 dis             = m_deviceNextPos[initial_index] - m_deviceNextPos[initial_index_j];

                        if (m_device_phase[initial_index] == 0.f || m_device_phase[initial_index] == 3.f )
                            continue;
                        if (length(rest) > m_sph_radius && length(dis) < m_sph_radius / 2 && m_device_phase[initial_index] == 1.f && (m_device_phase[initial_index_j] == 2.f || m_device_phase[initial_index_j] == 1.f))
                        {
                            float3 n     = (dis) / length(dis);
                            float3 deltx = (m_sph_radius / 2 - length(dis)) * n;
                            sumDeltx += 0.5 * deltx;
                        }
                        else if (length(rest) > m_sph_radius && length(dis) < m_sph_radius && m_device_phase[initial_index] == 2.f && (m_device_phase[initial_index_j] == 2.f || m_device_phase[initial_index_j] == 1.f))
                        {
                            float3 n     = (dis) / length(dis);
                            float3 deltx = (m_sph_radius - length(dis)) * n;
                            sumDeltx += 0.5 * deltx;
                        }
                        else if (length(dis) < m_sph_radius / 2 && m_device_phase[initial_index] == 1.f && m_device_phase[initial_index_j] == 3.f)
                        {
                            float3 n     = (dis) / length(dis);
                            float3 deltx = (m_sph_radius / 2 - length(dis)) * n;
                            sumDeltx += deltx;
                        }
                        else if (length(dis) < m_sph_radius && m_device_phase[initial_index] == 2.f && m_device_phase[initial_index_j] == 3.f)
                        {
                            float3 n     = (dis) / length(dis);
                            float3 deltx = (m_sph_radius - length(dis)) * n;
                            sumDeltx += deltx;
                        }
                    }
                }
            }
        }
        m_deviceDeltex[index] = sumDeltx;
    }
    __global__ void updateCollDeltxCuda(
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        unsigned int* m_device_index,
        int           m_num_particle)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= m_num_particle)
            return;

        int initial_index = m_device_index[index];
        m_deviceNextPos[initial_index] += m_deviceDeltex[index];
    }

}