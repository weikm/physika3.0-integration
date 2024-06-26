#ifndef __ELASTIC_KERNEL__

#define __ELASTIC_KERNEL__

#include <iostream>

#include "include/mat3.cuh"


namespace Physika {

    struct ElasticSolverParams;

    /**
     * @brief Set the Simulation Params object
     * 
     * @param p the simulation params object
     * @return __host__ 
     */
	__host__ void setSimulationParams(ElasticSolverParams* p);

    /**
	 * @brief cal the outer Product
	 *
	 * @param A
	 * @param B
	 * @param C
	 * @return __device__
	 */
	__device__ mat3 outerProduct(
        const float3& A,
        const float3& B);
    
    /**
     * @brief calucate the three dimensional grid to index using the world origin and cell size
     * 
     * @param p pos
     * @param m_world_orgin 
     * @param m_cell_size 
     * @return __device__ 
     */
    __device__ int3 calcGridPosKernel(
        float3 p, 
        float3 m_world_orgin, 
        float3 m_cell_size
    );
    
    /**
     * @brief calucate hash value of the grid
     * 
     * @param gridPos 
     * @param m_grid_num 
     * @return __device__ 
     */
    __device__ unsigned int calcGridHashKernel(
        int3 gridPos,
        uint3 m_grid_num
    );

    /**
     * @brief calucate particle hash value of the grid
     * 
     * @param gridParticleHash 
     * @param pos 
     * @param m_world_orgin 
     * @param m_cell_size 
     * @param m_grid_num 
     * @param numParticles 
     * @return __global__ 
     */
    __global__ void calcParticlesHashKernel(
        unsigned int* gridParticleHash,
        float3*       pos,
        float3        m_world_orgin,
        float3        m_cell_size,
        uint3         m_grid_num,
        unsigned int  numParticles);

    /**
     * @brief find the range of the grid
     * 
     * @param cellStart 
     * @param cellEnd 
     * @param gridParticleHash 
     * @param numParticles 
     * @return __global__ 
     */
    __global__ void findCellRangeKernel(
        unsigned int* cellStart,         
        unsigned int* cellEnd,          
        unsigned int* gridParticleHash,  
        unsigned int  numParticles);

    /**
     * @brief calucate the kernel of poly6 with radius
     * 
     * @param r 
     * @param m_sph_radius 
     * @return __device__ 
     */
    __device__ float kernelPoly6(float3 r, float m_sph_radius);

    /**
     * @brief calucate the gradient of the poly6 kernel
     * 
     * @param r 
     * @param m_sph_radius 
     * @return __device__ 
     */
    __device__ float3 gradientKenrelPoly6(float3 r, float m_sph_radius);
    
    /** @brief calucate the sum of the kernel for precompute
     * 
     * @param m_device_initial_pos 
     * @param m_device_sum 
     * @param grid 
     * @param gridStart 
     * @param m_volume 
     * @param m_sph_radius 
     * @param m_num_particle 
     * @return __global__ 
     */
     __global__ void calSumKernelPolyCuda(
        float3*       m_device_initial_pos,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        float         m_sph_radius,
        int           m_num_particle
	 );

    /** @brief calucate the y of the kernel for precompute
     * 
     * @param m_device_initial_pos 
     * @param m_deivce_y 
     * @param grid 
     * @param gridStart 
     * @param m_volume 
     * @param m_sph_radius 
     * @param m_num_particle 
     * @return __global__ */
    __global__ void calYCuda(
         float3*       m_device_initial_pos,
         float3*       m_deivce_y,
         unsigned int* grid,
         unsigned int* gridStart,
         float         m_volume,
         float         m_sph_radius,
         int           m_num_particle
	);
		
    /** @brief calucate the corrected sph kernel for precompute
     * 
     * @param m_device_initial_pos 
     * @param m_deivce_y 
     * @param m_device_sum 
     * @param m_deivce_L 
     * @param grid 
     * @param gridStart 
     * @param m_sph_radius 
     * @param m_volume 
     */
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
	);

    /**
	 * @brief cal the initial Corrected Kernel
	 *
	 * @param m_device_initial_pos
	 * @param m_device_y
	 * @param index_i
	 * @param index_j
	 * @param grid
	 * @param gridStart
	 * @return __device__
	 */
	 __device__ float3 CorrGradientKenrelPoly6(
		float3* m_device_initial_pos,
		float3* m_device_y,
		float* m_device_sum,
		float m_sph_radius,
		int index,
		int j
	);

    /**
     * @brief cal the sum for the corrected gradient SPH kernel
     * 
     * @param m_device_L 
     * @param m_device_initial_pos 
     * @param m_device_y 
     * @param m_device_sph_kernel 
     * @param m_device_sph_kernel_inv 
     * @param m_device_sum 
     * @param grid 
     * @param gridStart 
     * @param m_sph_radius 
     * @param m_num_particle 
     * @return __global__ 
     */
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
         float         m_sph_radius,
         int           m_num_particle
     );

    /**
	 * @brief cal the deformation gradient during simulation
	 * 
	 * @param m_device_next_pos 
	 * @param m_device_initial_pos 
	 * @param m_device_L 
	 * @param m_device_F 
	 * @param grid 
	 * @param gridStart 
	 * @param m_num_particle 
	 * @return __global__ */
   __global__ void calGradientDeformationCuda(
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
   );
    
    /** @brief cal the rotation matrix of the gradient SPH kernel
     * 
     * @param m_device_F 
     * @param m_device_R 
     * @param m_num_particle */
    __global__ void calRCuda(
		mat3* m_device_F,
		mat3* m_device_R,
		int m_num_particle
	);

    /**
     * @brief cal the PK1 matrix
     * 
     * @param m_device_F 
     * @param m_device_R 
     * @param m_device_P 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void calPCuda(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		mat3* m_device_P,
		int m_num_particle
	);

    /**
     * @brief cal the energy of the particle
     * 
     * @param m_device_F 
     * @param m_device_R 
     * @param m_device_energy 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void calEnergyCuda(
        float3 m_anisotropy,
		mat3* m_device_F,
		mat3* m_device_R,
		float* m_device_energy,
		int m_num_particle
	);

    /**
     * @brief initilaize the Lagrange multiplier
     * 
     * @param m_device_energy 
     * @param m_device_lm 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void calLMCuda(
		float* m_device_energy,
		float* m_device_lm,
		int m_num_particle
	);

    /**
     * @brief  cal the delta of the Lagrange multiplier of the XPBD
     * 
     * @param m_device_sph_kernel 
     * @param m_device_energy 
     * @param m_device_lm 
     * @param m_device_delte_lm 
     * @param m_device_sum 
     * @param m_device_initial_pos 
     * @param m_device_y 
     * @param m_device_L 
     * @param m_device_P 
     * @param grid 
     * @param gridStart 
     * @param m_h 
     * @param m_volume 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void solveDelteLagrangeMultiplierCuda(
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
	);

    /**
     * @brief cal the delta of the distance of the XPBD
     * 
     * @param m_device_sph_kernel 
     * @param m_device_sph_kernel_inv 
     * @param m_device_P 
     * @param m_device_L 
     * @param m_device_delte_lm 
     * @param m_device_sum 
     * @param m_device_delte_x 
     * @param m_device_initial_pos 
     * @param m_device_y 
     * @param grid 
     * @param gridStart 
     * @param m_volume 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void solveDelteDistanceCuda(
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
    );

    /**
     * @brief  update the Lagrange multiplier of the XPBD
     * 
     * @param m_deviceDeltex 
     * @param m_deviceNextPos 
     * @param m_device_delte_lm 
     * @param m_device_lm 
     * @param m_num_particle 
     * @return __global__ 
     */
    __global__ void updateDeltxlmCuda(
		float3* m_deviceDeltex,
		float3* m_deviceNextPos,
		float* m_device_delte_lm,
		float* m_device_lm,
		int m_num_particle
	);

    /**
     * @brief  update the position of the XPBD
     * 
     * @param m_deviceNextPos 
     * @param m_devicePos 
     * @param m_deviceVel 
     * @param m_numParticle 
     * @param deltaT 
     */
    __global__ void updateCuda(
        float3* m_deviceNextPos,
        float3* m_devicePos,
        float3* m_deviceVel,
        float*  m_device_phase,
        int m_numParticle,
        float deltaT
    );

    /**
     * @brief  advect the next position of  particle
     * 
     * @param m_deviceNextPos 
     * @param m_devicePos 
     * @param m_deviceVel 
     * @param m_deviceDeltex 
     * @param acc 
     * @param m_numParticle 
     * @param deltaT 
     */
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
    );


    __global__ void stretchCuda(
        float3* m_device_pos,
        float3* m_device_next_pos,
        float3* m_device_vel,
        float3* m_device_deltex,
        int     frame,
        int     m_num_particle
    );

    /**
     * @brief  solve the boundary constraint 
     * 
     * @param m_deviceNextPos 
     * @param m_deviceVel 
     * @param LB 
     * @param RT 
     * @param radius 
     */
    __global__ void solveBoundaryConstraintCuda(
        float3* m_deviceNextPos,
        float3* m_deviceVel,
        float3 LB,
        float3 RT,
        float radius
    );

    /**
     * @brief solve the collision distance of the particles
     * 
     * @param m_deviceInitialPos 
     * @param m_deviceDeltex 
     * @param m_deviceNextPos 
     * @param m_device_phase 
     * @param m_device_index 
     * @param cellStart 
     * @param cellEnd 
     * @param gridParticleHash 
     * @param m_sph_radius 
     * @param m_world_orgin 
     * @param m_celll_size 
     * @param m_grid_num 
     * @param m_num_particle 
     * @return __global__ 
     */
	__global__ void calCollisionDeltexCuda(
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
    );

    /**
     * @brief update the collision distance of the particles
     * 
     * @param m_deviceDeltex 
     * @param m_deviceNextPos 
     * @param m_device_index 
     * @param m_num_particle 
     * @return __global__ 
     */
	 __global__ void updateCollDeltxCuda(
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        unsigned int* m_device_index,
        int           m_num_particle
	 );
    }


#endif