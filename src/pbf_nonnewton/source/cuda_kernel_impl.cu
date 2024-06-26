//@author        : Long Shen
//@date          : 2023/10/8
//@description   :
//@version       : 1.0

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "cuda_kernel_api.cuh"
#include "cuda_device.cuh"
#include "cuda_tool.cuh"

namespace Physika {

extern __host__ void
ns_resetDevPtr(ConstPack& h_const, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors)
{
    static size_t size1 = h_const.ns_cellNum;
    static size_t size2 = h_const.total_particle_num;
    static size_t size3 = h_const.total_particle_num * h_const.ns_maxNeighborNum;

    cudaMemset(cellStart, UINT_MAX, size1 * sizeof(uint32_t));
    cudaMemset(cellEnd, UINT_MAX, size1 * sizeof(uint32_t));
    cudaMemset(neighborNum, 0, size2 * sizeof(uint32_t));
    cudaMemset(neighbors, UINT_MAX, size3 * sizeof(uint32_t));
}

extern __global__ void
ns_calcParticleHashValue_CUDA(ConstPack* d_const, uint32_t* particleIndices, uint32_t* cellIndices, float3* pos)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto cellPos = getCellPos(pos[i], d_const->ns_sceneLB, d_const->ns_cellLength);
    if (cellIsAvailable(cellPos, d_const->ns_gridSize))
    {
        uint32_t cellId    = getCellId(cellPos, d_const->ns_gridSize);
        cellIndices[i]     = cellId;
        particleIndices[i] = i;
    }
}

extern __host__ void
ns_calcParticleHashValue(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, uint32_t* cellIndices, float3* pos)
{
    ns_calcParticleHashValue_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                      particleIndices,
                                                                                      cellIndices,
                                                                                      pos);

    cudaGetLastError_t("ns_calcParticleHashValue() error.");
}

extern __host__ void
ns_sortByHashValue(ConstPack& h_const, uint32_t* particleIndices, uint32_t* cellIndices)
{
    thrust::device_ptr<uint32_t> keys_dev_ptr(cellIndices);
    thrust::device_ptr<uint32_t> values_dev_ptr(particleIndices);

    // 使用thrust::sort_by_key函数根据键进行排序
    thrust::sort_by_key(keys_dev_ptr, keys_dev_ptr + h_const.total_particle_num, values_dev_ptr);
}

extern __global__ void
ns_findCellRange_CUDA(ConstPack* d_const, uint32_t* cellIndices, uint32_t* cellStart, uint32_t* cellEnd)
{
    uint32_t i     = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pre_i = i - 1;
    if (i >= d_const->total_particle_num)
        return;

    uint32_t curCellId = cellIndices[i];
    if (i == 0)
        cellStart[curCellId] = 0;
    else
    {
        uint32_t preCellId = cellIndices[pre_i];
        if (curCellId != preCellId)
        {
            cellStart[curCellId] = i;
            cellEnd[preCellId]   = pre_i;
        }

        if (i == d_const->total_particle_num - 1)
            cellEnd[curCellId] = i;
    }
}

extern __host__ void
ns_findCellRange(ConstPack& h_const, ConstPack* d_const, uint32_t* cellIndices, uint32_t* cellStart, uint32_t* cellEnd)
{
    ns_findCellRange_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                              cellIndices,
                                                                              cellStart,
                                                                              cellEnd);

    cudaGetLastError_t("ns_findCellRange_CUDA() error.");
}

extern __global__ void
ns_findNeighbors_CUDA(ConstPack* d_const, int3* cellOffsets, uint32_t* particleIndices, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors, float3* pos)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i        = particleIndices[i];
    auto pos_i      = pos[p_i];
    auto pn_index   = p_i * d_const->ns_maxNeighborNum;
    int3 curCellPos = getCellPos(pos[p_i], d_const->ns_sceneLB, d_const->ns_cellLength);
    for (int t = 0; t < 27; ++t)
    {
        auto offset  = cellOffsets[t];
        int3 cellPos = curCellPos + offset;
        auto cellId  = getCellId(cellPos, d_const->ns_gridSize);
        if (cellIsAvailable(cellPos, d_const->ns_gridSize) && cellIsActivated(cellId, cellStart))
        {
            for (uint32_t j = cellStart[cellId]; j <= cellEnd[cellId]; ++j)
            {
                auto p_j   = particleIndices[j];
                auto pos_j = pos[p_j];
                if (length(pos_i - pos_j) <= d_const->ns_cellLength)
                {
                    if (neighborNum[p_i] < d_const->ns_maxNeighborNum)
                    {
                        auto ind_offset                  = neighborNum[p_i]++;
                        neighbors[pn_index + ind_offset] = p_j;
                    }
                }
            }
        }
    }
}

extern __host__ void
ns_findNeighbors(ConstPack& h_const, ConstPack* d_const, int3* cellOffsets, uint32_t* particleIndices, uint32_t* cellStart, uint32_t* cellEnd, uint32_t* neighborNum, uint32_t* neighbors, float3* pos)
{
    ns_findNeighbors_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                              cellOffsets,
                                                                              particleIndices,
                                                                              cellStart,
                                                                              cellEnd,
                                                                              neighborNum,
                                                                              neighbors,
                                                                              pos);

    cudaGetLastError_t("ns_findNeighbors_CUDA() error.");
}

extern __global__ void
algo_computeRigidParticleVolume_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;
    if (material[p_i] == SimMaterial::BOUND || material[p_i] == SimMaterial::RIGID)
    {

        auto  pos_i = predictPos[p_i];
        float delta = 0;

        for (unsigned int p_j = neighbors[neib_ind], t = 0;
             p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
             ++t, p_j = neighbors[neib_ind + t])
        {

        TASK_DOMAIN: {
            auto pos_j = predictPos[p_j];

            if (material[p_j] == SimMaterial::BOUND || material[p_j] == SimMaterial::RIGID)
                delta += cubic_value(length(pos_i - pos_j), d_const->sph_h);
        };
        }

        volume[p_i] = 1 / delta;
    }
}

extern __host__ void
algo_computeRigidParticleVolume(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume)
{
    algo_computeRigidParticleVolume_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                             neighbors,
                                                                                             particleIndices,
                                                                                             material,
                                                                                             predictPos,
                                                                                             volume);
    cudaGetLastError_t("algo_computeRigidParticleVolume_CUDA() error.");
}

extern __global__ void
algo_computeExtForce_CUDA(ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* ext_force, float3* predictPos, float3* vel, float3* acc)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
        return;

    // Compute gravity
    acc[p_i] = d_const->gravity + ext_force[p_i];

    // Compute surface tension
    // ...

    // __syncthreads();

    // Apply acc
    vel[p_i] += acc[p_i] * d_const->dt;
    predictPos[p_i] += vel[p_i] * d_const->dt;
}

extern __host__ void
algo_computeExtForce(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* ext_force, float3* predictPos, float3* vel, float3* acc)
{
    algo_computeExtForce_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                  particleIndices,
                                                                                  material,
                                                                                  ext_force,
                                                                                  predictPos,
                                                                                  vel,
                                                                                  acc);

    cudaGetLastError_t("algo_computeExtForce_CUDA() error.");
}

extern __global__ void
algo_computeDensity_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume, float* density)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
        return;

    density[p_i] = 0.0;
    auto pos_i   = predictPos[p_i];

    for (unsigned int p_j = neighbors[neib_ind], t = 0;
         p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
         ++t, p_j = neighbors[neib_ind + t])
    {

    TASK_DOMAIN: {
        auto pos_j = predictPos[p_j];

        if (material[p_j] == SimMaterial::FLUID)
            density[p_i] += d_const->rest_mass * cubic_value(length(pos_i - pos_j), d_const->sph_h);

        else if (material[p_j] == SimMaterial::RIGID || material[p_j] == SimMaterial::BOUND)
            density[p_i] +=
                d_const->rest_density * volume[p_j] * cubic_value(length(pos_i - pos_j), d_const->sph_h);
    };
    }

    density[p_i] = max(density[p_i], d_const->rest_density);
}

extern __host__ void
algo_computeDensity(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float* volume, float* density)
{
    algo_computeDensity_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                 neighbors,
                                                                                 particleIndices,
                                                                                 material,
                                                                                 predictPos,
                                                                                 volume,
                                                                                 density);
    cudaGetLastError_t("algo_computeDensity_CUDA() error.");
}

extern __global__ void
algo_computeDxFromDensityConstraint_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* volume, float* density, float3* predictPos, float3* dx, float* lam)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
        return;

    // Compute lambda
    lam[p_i] *= 0;
    float c_dens_w = (density[p_i] / d_const->rest_density - 1) / d_const->rest_mass;  // eq.(7) up-hand-side
    if (c_dens_w != 0)
    {

        float3 c_dens_grad    = { 0, 0, 0 };
        float  sum_grads_norm = 1e-6;
        auto   pos_i          = predictPos[p_i];
        for (unsigned int p_j = neighbors[neib_ind], t = 0; p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
             ++t, p_j                                  = neighbors[neib_ind + t])
        {
        TASK1_DOMAIN: {
            auto pos_j = predictPos[p_j];
            if (material[p_j] == SimMaterial::FLUID)
            {
                auto c_grad = d_const->rest_mass / d_const->rest_density * cubic_gradient(pos_i - pos_j, d_const->sph_h);
                c_dens_grad += c_grad;
                sum_grads_norm += length(c_grad) * length(c_grad) / d_const->rest_mass;
            }
            else if (material[p_j] == SimMaterial::RIGID || material[p_j] == SimMaterial::BOUND)
            {
                auto c_grad = volume[p_j] * cubic_gradient(pos_i - pos_j, d_const->sph_h);
                c_dens_grad += c_grad;
                sum_grads_norm += length(c_grad) * length(c_grad) / d_const->rest_mass;
            }
        };
        }

        sum_grads_norm +=
            length(c_dens_grad) * length(c_dens_grad) / d_const->rest_mass;  // eq.(7) down-hand-side
        lam[p_i] = -c_dens_w / sum_grads_norm;                               // eq.(7)
    }

    __syncthreads();

    // Compute dx
    dx[p_i] *= 0;
    auto pos_i = predictPos[p_i];
    for (unsigned int p_j = neighbors[neib_ind], t = 0; p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
         ++t, p_j                                  = neighbors[neib_ind + t])
    {

    TASK2_DOMAIN: {
        auto pos_j = predictPos[p_j];

        auto s_corr = -d_const->pbf_Ks * pow(cubic_value(length(pos_i - pos_j), d_const->sph_h) / cubic_value(d_const->pbf_Dq, d_const->sph_h), 4);
        s_corr      = 0;

        if (material[p_j] == SimMaterial::FLUID)
        {

            dx[p_i] += d_const->rest_mass / d_const->rest_density * (lam[p_i] + lam[p_j] + s_corr) * cubic_gradient(pos_i - pos_j, d_const->sph_h);  // eq.(8+)
        }
        else if (material[p_j] == SimMaterial::RIGID || material[p_j] == SimMaterial::BOUND)
        {
            dx[p_i] +=
                volume[p_j] * (lam[p_i] + lam[p_i] + s_corr) * cubic_gradient(pos_i - pos_j, d_const->sph_h);  // eq.(8+)
        }
    };
    }

    __syncthreads();

    // Correct predictPos
    predictPos[p_i] += dx[p_i];
}

extern __host__ void
algo_computeDxFromDensityConstraint(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* volume, float* density, float3* predictPos, float3* dx, float* lam)
{
    algo_computeDxFromDensityConstraint_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                                 neighbors,
                                                                                                 particleIndices,
                                                                                                 material,
                                                                                                 volume,
                                                                                                 density,
                                                                                                 predictPos,
                                                                                                 dx,
                                                                                                 lam);

    cudaGetLastError_t("algo_computeDxFromDensityConstraint_CUDA() error.");
}

extern __global__ void
algo_applyDx_CUDA(ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float3* pos, float3* vel)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i = particleIndices[i];

    if (material[p_i] == SimMaterial::FLUID)
    {
        vel[p_i] = (predictPos[p_i] - pos[p_i]) / d_const->dt;
        pos[p_i] = predictPos[p_i];
    }
}

extern __host__ void
algo_applyDx(ConstPack& h_const, ConstPack* d_const, uint32_t* particleIndices, SimMaterial* material, float3* predictPos, float3* pos, float3* vel)
{
    algo_applyDx_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                          particleIndices,
                                                                          material,
                                                                          predictPos,
                                                                          pos,
                                                                          vel);

    cudaGetLastError_t("algo_applyDx_CUDA() error.");
}

extern __global__ void
algo_computeVisForce_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float* vis)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
        return;

    auto   pos_i     = predictPos[p_i];
    auto   vel_i     = vel[p_i];
    float  dimFactor = 10;
    float  h2_001    = 0.01 * d_const->sph_h * d_const->sph_h;
    float3 da{ 0, 0, 0 };

    for (unsigned int p_j = neighbors[neib_ind], t = 0;
         p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
         ++t, p_j = neighbors[neib_ind + t])
    {

    TASK_DOMAIN: {
        auto pos_j = predictPos[p_j];
        auto vel_j = vel[p_j];

        auto vel_ij = vel_i - vel_j;
        auto pos_ij = pos_i - pos_j;

        if (material[p_j] == SimMaterial::FLUID)
        {
            da += (vis[p_j] + vis[p_i]) / 2 * dimFactor * d_const->rest_mass / density[p_j] * dot(vel_ij, pos_ij) / (length(pos_ij) * length(pos_ij) + h2_001) * cubic_gradient(pos_ij, d_const->sph_h);
        }
        //        if (material[p_j] == SimMaterial::RIGID || material[p_j] == SimMaterial::BOUND)
        //        {
        //            da += d_const->cross_visBound * dimFactor * d_const->rest_mass / d_const->rest_density * dot(vel_ij, pos_ij) / (length(pos_ij) * length(pos_ij) + h2_001) * cubic_gradient(pos_ij, d_const->sph_h);
        //        }
    };
    }

    vel[p_i] += 0.8 * da * d_const->dt;
}

extern __global__ void
algo_computeVisForceXSPH_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float3* acc)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
        return;

    auto   pos_i = predictPos[p_i];
    auto   vel_i = vel[p_i];
    float3 dv{ 0, 0, 0 };

    for (unsigned int p_j = neighbors[neib_ind], t = 0;
         p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
         ++t, p_j = neighbors[neib_ind + t])
    {

    TASK_DOMAIN: {
        auto pos_j = predictPos[p_j];
        auto vel_j = vel[p_j];

        if (material[p_j] == SimMaterial::FLUID)
            dv += 0.5 * d_const->rest_mass / density[p_j] * (vel_j - vel_i) * cubic_value(length(pos_i - pos_j), d_const->sph_h);
    };
    }

    vel[p_i] += dv;
}

extern __global__ void
algo_computeNNVis_CUDA(ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float* vis, float* shearRate)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_const->total_particle_num)
        return;

    auto p_i      = particleIndices[i];
    auto neib_ind = p_i * d_const->ns_maxNeighborNum;

    if (material[p_i] != SimMaterial::FLUID)
    {
        vis[p_i] = d_const->cross_visBound;
        return;
    }

    float gradV[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    auto  vel_i    = vel[p_i];
    auto  pos_i    = predictPos[p_i];

    for (unsigned int p_j = neighbors[neib_ind], t = 0;
         p_j != UINT_MAX && t < d_const->ns_maxNeighborNum;
         ++t, p_j = neighbors[neib_ind + t])
    {

    TASK_DOMAIN: {
        auto  vel_j = vel[p_j];
        auto  pos_j = predictPos[p_j];
        float d_grad[9];

        outer_product(d_grad, d_const->rest_mass / density[p_j] * (vel_j - vel_i), cubic_gradient(pos_i - pos_j, d_const->sph_h));
        sum_f9(gradV, d_grad);
    };
    }

    grad_sum_gradT(gradV);
    shearRate[p_i] = sqrt(0.5 * pow(trace(gradV), 2));
    vis[p_i]       = d_const->cross_visInf + (d_const->cross_vis0 - d_const->cross_visInf) / (1 + pow(d_const->cross_K * shearRate[p_i], d_const->cross_N));
}

extern __host__ void
algo_computeVisForce(ConstPack& h_const, ConstPack* d_const, uint32_t* neighbors, uint32_t* particleIndices, SimMaterial* material, float* density, float3* predictPos, float3* vel, float3* acc, float* vis, float* shearRate)
{
    algo_computeNNVis_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                               neighbors,
                                                                               particleIndices,
                                                                               material,
                                                                               density,
                                                                               predictPos,
                                                                               vel,
                                                                               vis,
                                                                               shearRate);

    algo_computeVisForce_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
                                                                                  neighbors,
                                                                                  particleIndices,
                                                                                  material,
                                                                                  density,
                                                                                  predictPos,
                                                                                  vel,
                                                                                  vis);

    //    algo_computeVisForceXSPH_CUDA<<<h_const.ns_blockNum, h_const.ns_threadPerBlock>>>(d_const,
    //                                                                                      neighbors,
    //                                                                                      particleIndices,
    //                                                                                      material,
    //                                                                                      density,
    //                                                                                      predictPos,
    //                                                                                      vel,
    //                                                                                      acc);

    cudaGetLastError_t("algo_computeVisForce_CUDA() error.");
}

}  // namespace Physika
