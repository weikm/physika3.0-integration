//@author        : Long Shen
//@date          : 2023/10/8
//@description   :
//@version       : 1.0

#ifndef PHYSIKA_CUDA_DEVICE_CUH
#define PHYSIKA_CUDA_DEVICE_CUH

#include "helper_math.hpp"

namespace Physika {

/**
 * @brief sph cubic kernel value
 *
 * @param[in] r_norm length of (pos_i-pos_j)
 * @param[in] h sph smoothing radius
 *
 * @return    cubic value
 */
__device__ inline float
cubic_value(const float r_norm, const float h)
{
    const float PI         = 3.14159265;
    const float cubicSigma = 8.f / PI / static_cast<float>(std::pow(h, 3));

    float res  = 0.0;
    float invH = 1 / h;
    float q    = r_norm * invH;

    if (q <= 1)
    {
        if (q <= 0.5)
        {
            auto q2 = q * q;
            auto q3 = q2 * q;
            res     = static_cast<float>(cubicSigma * (6.0 * q3 - 6.0 * q2 + 1));
        }
        else
        {
            res = static_cast<float>(cubicSigma * 2 * std::pow(1 - q, 3));
        }
    }

    return res;
}

/**
 * @brief sph cubic kernel gradient
 *
 * @param[in] r vector of (pos_i-pos_j)
 * @param[in] h sph smoothing radius
 *
 * @return    cubic gradient
 */
__device__ inline float3
cubic_gradient(const float3& r, const float h)
{
    const float PI         = 3.14159265;
    const float cubicSigma = 8.f / PI / static_cast<float>(std::pow(h, 3));

    auto  res  = make_float3(0, 0, 0);
    float invH = 1 / h;
    float q    = length(r) * invH;

    if (q < 1e-6 || q > 1)
        return res;

    float3 grad_q = r / (length(r) * h);
    if (q <= 0.5)
        res = (6 * (3 * q * q - 2 * q)) * grad_q * cubicSigma;
    else
    {
        auto factor = 1 - q;
        res         = -6 * factor * factor * grad_q * cubicSigma;
    }

    return res;
}

/**
 * @brief math helper, floor [float3] to [int3]
 *
 * @param[in] v 3d-vector of float
 *
 * @return    3d-vector of int
 */
__device__ inline int3
floor_to_int3(const float3& v)
{
    return make_int3(static_cast<int>(floor(v.x)),
                     static_cast<int>(floor(v.y)),
                     static_cast<int>(floor(v.z)));
}

/**
 * @brief compute cell pos by particle pos
 *
 * @param[in] pos particle pos
 * @param[in] sceneLB left-bottom of the scene
 * @param[in] cellLength cell length
 *
 * @return    cell pos, 3d-vector of int
 */
__device__ inline int3
getCellPos(const float3& pos, const float3& sceneLB, const float cellLength)
{
    int3 cellPos = floor_to_int3((pos - sceneLB) / cellLength);
    return cellPos;
}

/**
 * @brief compute cell id by cell pos
 *
 * @param[in] cellPos cell pos
 * @param[in] gridSize size of background grid
 *
 * @return    cell id, uint32_t
 */
__device__ inline uint32_t
getCellId(const int3& cellPos, const uint3& gridSize)
{
    uint32_t cellId =
        cellPos.z * (gridSize.y * gridSize.x) + cellPos.y * gridSize.x + cellPos.x;
    return cellId;
}

/**
 * @brief check if cur cell is available
 *
 * @param[in] cellPos cell pos
 * @param[in] gridSize size of background grid
 *
 * @return    true if available, false otherwise
 */
__device__ inline bool
cellIsAvailable(const int3& cellPos, const uint3& gridSize)
{
    auto cellStartPos = make_int3(0, 0, 0);
    auto cellEndPos   = make_int3(gridSize);

    return (cellPos.x >= cellStartPos.x && cellPos.y >= cellStartPos.y && cellPos.z >= cellStartPos.z && cellPos.x <= cellEndPos.x && cellPos.y <= cellEndPos.y && cellPos.z <= cellEndPos.z);
}

/**
 * @brief check if cur cell is activated
 *
 * @param[in] cellId cell id
 * @param[in] cellStart device pointer of the cell start array
 *
 * @return    true if cell is not empty, false otherwise
 */
__device__ inline bool
cellIsActivated(const uint32_t cellId, const uint32_t* cellStart)
{
    return (cellStart[cellId] != UINT_MAX);
}

}  // namespace Physika

#endif  // PHYSIKA_CUDA_DEVICE_CUH
