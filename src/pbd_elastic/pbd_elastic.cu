#include "pbd_elastic_kernel.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>



namespace Physika {

	struct ElasticSolverParams;

    void getLastCudaError(const char* errorMessage) {
		// check cuda last error.
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			std::cout << "getLastCudaError() CUDA error : "
				<< errorMessage << " : " << "(" << static_cast<int>(err) << ") "
				<< cudaGetErrorString(err) << ".\n";
		}
	}

    void setParamters(ElasticSolverParams* p) {
        setSimulationParams(p);
        getLastCudaError("set params error");
        // cudaMemcpyToSymbol(params, p, sizeof(Params));
    }

	void computeHash(
        unsigned int* gridParticleHash,
        float*        pos,
        float3        m_world_orgin,
        float3        m_cell_size,
        uint3          m_grid_num,
        int           numParticles)
    {
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks  = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

        // launch the kernel.
        calcParticlesHashKernel<<<numBlocks, numThreads>>>(
            gridParticleHash,
            ( float3* )pos,
            m_world_orgin,
            m_cell_size,
            m_grid_num,
            numParticles);
    }

    void sortParticles(
        unsigned int* deviceGridParticleHash,
        unsigned int* m_device_index,
        unsigned int  numParticles)
    {
        thrust::device_ptr<unsigned int> ptrindex(m_device_index);

        thrust::sort_by_key(
            thrust::device_ptr<unsigned int>(m_device_index),
            thrust::device_ptr<unsigned int>(m_device_index + numParticles),
            thrust::make_zip_iterator(thrust::make_tuple(ptrindex)));

        thrust::sort_by_key(
            thrust::device_ptr<unsigned int>(deviceGridParticleHash),
            thrust::device_ptr<unsigned int>(deviceGridParticleHash + numParticles),
            thrust::make_zip_iterator(thrust::make_tuple(ptrindex)));
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

    void update(
		float3* m_deviceNextPos,
		float3* m_devicePos,
		float3* m_deviceVel,
        float*  m_device_phase,
		float deltaT,
		int m_numParticles
	) {
		int block_size = 256;
		updateCuda <<< (m_numParticles - 1) / block_size + 1, block_size>>> (
            m_deviceNextPos, m_devicePos, m_deviceVel, m_device_phase, m_numParticles, deltaT);
	}

	void advect(
		float3* m_devicePos,
		float3* m_deviceNextPos,
		float3* m_deviceVel,
		float3* m_deviceDeltex,
        float3*  m_deviceExternalForce,
        float*  m_device_phase,
		float3 acc,
		float deltaT,
		int m_numParticles
	) {
		int block_size = 256;

		advectCuda <<< (m_numParticles - 1) / block_size + 1, block_size >>> (
            m_deviceNextPos, m_devicePos, m_deviceVel, m_deviceDeltex, m_deviceExternalForce, m_device_phase, acc, m_numParticles, deltaT
			);
	}
    
    void stretch(
        float3* m_device_pos,
        float3* m_device_next_pos,
        float3* m_device_vel,
        float3* m_device_deltex,
        int     frame,
        int     m_num_particle)
    {
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks  = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

        stretchCuda<<<numBlocks, numThreads>>>(
            m_device_pos,
            m_device_next_pos,
            m_device_vel,
            m_device_deltex,
            frame,
            m_num_particle);
    }

    void solveBoundaryConstraint(
		float3* m_deviceNextPos,
		float3* m_deviceVel,
		float3 LB,
		float3 RT,
		float radius,
		int m_numParticle
	) {
		int block_size = 256;
		solveBoundaryConstraintCuda <<< (m_numParticle - 1) / block_size + 1, block_size >>> (
		m_deviceNextPos, m_deviceVel, LB, RT, radius);
	}

    void calY(
        float3*       m_device_initial_pos,
        float3*       m_deivce_y,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        float         m_sph_radius,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calYCuda <<<numBlocks,numThreads>>> (
			m_device_initial_pos,
			m_deivce_y,
			grid,
			gridStart,
            m_volume,
            m_sph_radius,
			m_num_particle
		);
	}

	void calL(
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float*        m_device_sum,
        mat3*         m_deivce_L,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_sph_radius,
        float         m_volume,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calLCuda <<<numBlocks,numThreads>>> (
			m_device_initial_pos,
			m_device_y,
			m_device_sum,
			m_deivce_L,
			grid,
			gridStart,
            m_sph_radius,
            m_volume,
			m_num_particle
		);
	}

	void calSumKernelPoly(
        float3*       m_device_initial_pos,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        float         m_sph_radius,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calSumKernelPolyCuda <<<numBlocks,numThreads>>> (
			m_device_initial_pos,
			m_device_sum,
			grid,
			gridStart,
            m_volume,
            m_sph_radius,
			m_num_particle
		);
	}

	void calGradientDeformation(
        float3*       m_device_sph_kernel,
        float3*       m_device_next_pos,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float*        m_device_sum,
        mat3*         m_device_L,
        mat3*         m_device_F,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calGradientDeformationCuda <<<numBlocks,numThreads>>> (
			m_device_sph_kernel,
			m_device_next_pos,
			m_device_initial_pos,
			m_device_y,
			m_device_sum,
			m_device_L,
			m_device_F,
			grid,
			gridStart,
            m_volume,
			m_num_particle
		);

	}

	void calR(
		mat3* m_device_F,
		mat3* m_device_R,
		int m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calRCuda <<<numBlocks, numThreads >>> (
			m_device_F,
			m_device_R,
			m_num_particle
			);
	}


	void calP(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		mat3* m_device_P,
		int m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calPCuda <<<numBlocks,numThreads>>> (
            m_anisotropy,
			m_device_F, 
			m_device_R,
			m_device_P,
			m_num_particle
		);
	}

	void calEnergy(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		float* m_device_energy,
		int m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calEnergyCuda <<<numBlocks,numThreads>>> (
            m_anisotropy,
			m_device_F,
			m_device_R,
			m_device_energy,
			m_num_particle
		);
	}

	void calLM(
		float* m_device_energy,
		float* m_device_lm,
		int m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		calLMCuda <<<numBlocks,numThreads>>> (
			m_device_energy,
			m_device_lm,
			m_num_particle
		);
	}

	void solveDelteLagrangeMultiplier(
        float3*       m_device_sph_kernel,
        float*        m_device_energy,
        float*        m_device_lm,
        float*        m_device_delte_lm,
        float*        m_device_sum,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        mat3*         m_device_L,
        mat3*         m_device_P,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_h,
        float         m_volume,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 256;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		solveDelteLagrangeMultiplierCuda <<<numBlocks,numThreads>>> (
			m_device_sph_kernel,
			m_device_energy,
			m_device_lm,
			m_device_delte_lm,
			m_device_sum,
			m_device_initial_pos,
			m_device_y,
			m_device_L,
			m_device_P,
			grid,
			gridStart,
			m_h,
            m_volume,
			m_num_particle
		);
	}
	
	void solveDelteDistance(
		 float3*       m_device_sph_kernel,
        float3*       m_device_sph_kernel_inv,
        mat3*         m_device_P,
        mat3*         m_device_L,
        float*        m_device_delte_lm,
        float*        m_device_sum,
        float3*       m_device_delte_x,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        int           m_num_particle
	) {
		unsigned int numThreads, numBlocks;
		numThreads = 512;
		numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

		solveDelteDistanceCuda <<<numBlocks,numThreads>>> (
			m_device_sph_kernel,
			m_device_sph_kernel_inv,
			m_device_P,
			m_device_L,
			m_device_delte_lm,
			m_device_sum,
			m_device_delte_x,
			m_device_initial_pos,
			m_device_y,
			grid,
			gridStart,
            m_volume,
			m_num_particle
		);
	}

    void updateDeltxlm(
        float3* m_deviceDeltex,
        float3* m_deviceNextPos,
        float* m_device_delte_lm,
        float* m_device_lm,
        int m_num_particle
    ){
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

        updateDeltxlmCuda <<<numBlocks,numThreads>>> (
            m_deviceDeltex,
            m_deviceNextPos,
            m_device_delte_lm,
            m_device_lm,
            m_num_particle
        );
    }

    void calCorrGradientKenrelSum(
        mat3*         m_device_L,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float3*       m_device_sph_kernel,
        float3*       m_device_sph_kernel_inv,
        float3*       m_device_external_force,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_sph_radius,
        int           m_num_particle
    ) {
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

        calCorrGradientKenrelSumCuda << <numBlocks, numThreads >> > (
            m_device_L,
            m_device_initial_pos,
            m_device_y,
            m_device_sph_kernel,
            m_device_sph_kernel_inv,
            m_device_external_force,
            m_device_sum,
            grid,
            gridStart,
            m_sph_radius,
            m_num_particle
        );

    }

	void calCollisionDeltex(
        float3*       m_deviceInitialPos,
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        float*        m_device_phase,
        unsigned int* m_device_index,
        unsigned int* cellStart,
        unsigned int* cellEnd,
        unsigned int* gridParticleHash,
        float         m_sph_radius,
        float3        m_world_orgin,
        float3        m_celll_size,
        uint3          m_grid_num,
        int           m_num_particle
    )
    {
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks  = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

        calCollisionDeltexCuda<<<numBlocks, numThreads>>>(
            m_deviceInitialPos,
            m_deviceDeltex,
            m_deviceNextPos,
            m_device_phase,
            m_device_index,
            cellStart,
            cellEnd,
            gridParticleHash,
            m_sph_radius,
            m_world_orgin,
            m_celll_size,
            m_grid_num,
            m_num_particle);
    }

    void updateCollDeltx(
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        unsigned int* m_device_index,
        int           m_num_particle)
    {
        unsigned int numThreads, numBlocks;
        numThreads = 256;
        numBlocks  = (m_num_particle % numThreads != 0) ? (m_num_particle / numThreads + 1) : (m_num_particle / numThreads);

        updateCollDeltxCuda<<<numBlocks, numThreads>>>(
            m_deviceDeltex,
            m_deviceNextPos,
            m_device_index,
            m_num_particle);
    }

}