//@author        : Long Shen
//@date          : 2023/10/7
//@description   :
//@version       : 1.0

#include <iostream>

#include "enum.hpp"
#include "const_pack.hpp"

namespace Physika {  // neighbor search api

/**
 * @brief  : reset data of neighbor-searcher
 *
 * @param[in]  : h_const  		host pointer of solver constant params
 * @param[in]  : cellStart      device pointer of the cell start array
 * @param[in]  : cellEnd   	    device pointer of the cell end array
 * @param[in]  : neighborNum    device pointer of neighbor num
 * @param[in]  : neighbors      device pointer of neighbors index array
 */
extern __host__ void
ns_resetDevPtr(ConstPack& h_const, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors);

/**
 * @brief  : [cuda-kernel-impl] calculate hash index of each particle by their pos
 *
 * @param[in]  : d_const  		device pointer of solver constant params
 * @param[in]  : particleIndices      device pointer of the particle index array
 * @param[in]  : cellIndices   	    device pointer of the cell index array
 * @param[in]  : pos    device pointer of cur particles' pos
 */
extern __global__ void
ns_calcParticleHashValue_CUDA(ConstPack* d_const, uint32_t* particleIndices, uint32_t* cellIndices, float3* pos);

/**
 * @brief  : [host-invoke] calculate hash index of each particle by their pos
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  		device pointer of solver constant params
 * @param[in]  : particleIndices      device pointer of the particle index array
 * @param[in]  : cellIndices   	    device pointer of the cell index array
 * @param[in]  : pos    device pointer of cur particles' pos
 */
extern __host__ void
ns_calcParticleHashValue(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, uint32_t* cellIndices, float3* pos);

/**
 * @brief  : sort the particles based on the hash value
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : particleIndices  device pointer of the particle index array
 * @param[in]  : cellIndices    device pointer of the cell index array
 */
extern __host__ void
ns_sortByHashValue(ConstPack& h_const, uint32_t* particleIndices, uint32_t* cellIndices);

/**
 * @brief  : [cuda-kernel-impl] reorder the particle data based on the sorted hash value
 *
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : cellIndices  device pointer of the cell index array
 * @param[in]  : cellStart    device pointer of the cell start array
 * @param[in]  : cellEnd    device pointer of the cell end array
 */
extern __global__ void
ns_findCellRange_CUDA(ConstPack* d_const, uint32_t* cellIndices, uint32_t* cellStart, uint32_t* cellEnd);

/**
 * @brief  : [host-invoke] reorder the particle data based on the sorted hash value
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : cellIndices  device pointer of the cell index array
 * @param[in]  : cellStart    device pointer of the cell start array
 * @param[in]  : cellEnd    device pointer of the cell end array
 */
extern __host__ void
ns_findCellRange(ConstPack& h_const, ConstPack* d_const, uint32_t* cellIndices, uint32_t* cellStart, uint32_t* cellEnd);

/**
 * @brief  : [cuda-kernel-impl] reorder the particle data based on the sorted hash value
 *
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : cellOffsets  device pointer of the cell offset
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : cellStart    device pointer of the cell start array
 * @param[in]  : cellEnd    device pointer of the cell end array
 * @param[in]  : neighborNum    device pointer of neighbor num
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : pos    device pointer of cur particles' pos
 */
extern __global__ void
ns_findNeighbors_CUDA(ConstPack* d_const, int3* cellOffsets, uint32_t* particleIndices, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors, float3* pos);

/**
 * @brief  : [host-invoke] reorder the particle data based on the sorted hash value
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : cellOffsets  device pointer of the cell offset
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : cellStart    device pointer of the cell start array
 * @param[in]  : cellEnd    device pointer of the cell end array
 * @param[in]  : neighborNum    device pointer of neighbor num
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : pos    device pointer of cur particles' pos
 */
extern __host__ void
ns_findNeighbors(ConstPack& h_const, ConstPack* d_const, int3* cellOffsets, uint32_t* particleIndices, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors, float3* pos);

}

namespace Physika {  // algorithm api, description is the same as above. only [host-invoke] api described.

/**
 * @brief  : [cuda-kernel-impl] compute volume of boundary particles for fluid-rigid coupling
 */
extern __global__ void
algo_computeRigidParticleVolume_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume);

/**
 * @brief  : [host-invoke] compute volume of boundary particles for fluid-rigid coupling
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in/out]  : volume    device pointer of particle volume
 */
extern __host__ void
algo_computeRigidParticleVolume(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume);

/**
 * @brief  : [cuda-kernel-impl] compute external force
 */
extern __global__ void
algo_computeExtForce_CUDA(ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float3* vel, float3* acc);

/**
 * @brief  : [host-invoke] compute external force
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in]  : vel    device pointer of particle velocity
 * @param[in/out]  : acc    device pointer of particle accelerate
 */
extern __host__ void
algo_computeExtForce(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* ext_force, float3* predictPos, float3* vel, float3* acc);

/**
 * @brief  : [cuda-kernel-impl] compute density of fluid particles
 */
extern __global__ void
algo_computeDensity_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume, float* density);

/**
 * @brief  : [host-invoke] compute density of fluid particles
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in]  : volume    device pointer of particle volume
 * @param[in/out]  : density    device pointer of particle density
 */
extern __host__ void
algo_computeDensity(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume, float* density);

/**
 * @brief  : [cuda-kernel-impl] compute delta_x using PBF density constraint
 */
extern __global__ void
algo_computeDxFromDensityConstraint_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* volume, float* density, float3* predictPos, float3* dx, float* lam);

/**
 * @brief  : [host-invoke] compute delta_x using PBF density constraint
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : volume    device pointer of particle volume
 * @param[in]  : density    device pointer of particle density
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in/out]  : dx    device pointer of particles' delta_x
 * @param[in/out]  : lam    device pointer of param of PBF density constraint' solve
 */
extern __host__ void
algo_computeDxFromDensityConstraint(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* volume, float* density, float3* predictPos, float3* dx, float* lam);

/**
 * @brief  : [cuda-kernel-impl] advect pos from delta_x
 */
extern __global__ void
algo_applyDx_CUDA(ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float3* pos, float3* vel);

/**
 * @brief  : [host-invoke] advect pos from delta_xnt
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in/out]  : pos    device pointer of particles' pos
 * @param[in/out]  : vel    device pointer of particle velocity
 */
extern __host__ void
algo_applyDx(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float3* pos, float3* vel);

/**
 * @brief  : [cuda-kernel-impl] compute cross-viscous force
 */
extern __global__ void
algo_computeVisForce_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float* vis);

/**
 * @brief  : [cuda-kernel-impl] compute cross-viscosity
 */
extern __global__ void
algo_computeNNVis_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float* vis, float* shearRate);

/**
 * @brief  : [cuda-kernel-impl] compute XSPH viscous force
 */
extern __global__ void
algo_computeVisForceXSPH_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float3* acc);

/**
 * @brief  : [host-invoke] advect pos from delta_xnt
 *
 * @param[in]  : h_const  host pointer of solver constant params
 * @param[in]  : d_const  device pointer of solver constant params
 * @param[in]  : neighbors      device pointer of neighbors index array
 * @param[in]  : particleIndices    device pointer of the particle index array
 * @param[in]  : material  device pointer of the particle simMaterial
 * @param[in]  : density    device pointer of particle density
 * @param[in]  : predictPos    device pointer of mid particles' pos
 * @param[in]  : vel    device pointer of particle velocity
 * @param[in/out]  : acc    device pointer of particle accelerate
 * @param[in/out]  : vis    device pointer of particle cross-viscosity
 * @param[in/out]  : shearRate    device pointer of particle shear-rate
 */
extern __host__ void
algo_computeVisForce(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float3* acc, float* vis, float* shearRate);

}  // namespace Physika
