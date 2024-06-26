//@author        : Long Shen
//@date          : 2023/10/8
//@description   :
//@version       : 1.0

#ifndef PHYSIKA_CONST_PACK_HPP
#define PHYSIKA_CONST_PACK_HPP

#include <iostream>
#include <vector_types.h>

#ifndef CROSS_N
#define CROSS_N 0
#endif

namespace Physika {

/**
 * @brief constant params of the solver
 */
struct ConstPack
{
    uint32_t ns_threadPerBlock{ 0 };
    uint32_t ns_blockNum{ 0 };
    float    ns_cellLength{ 0 };
    uint3    ns_gridSize{ 0, 0, 0 };
    uint32_t ns_cellNum{ 0 };
    float3   ns_sceneLB{ -10, -10, -10 };
    float3   ns_sceneSize{ 20, 20, 20 };
    uint32_t ns_maxNeighborNum{ 35 };

    float3   gravity{ 0, -9.8f, 0 };
    float    rest_density{ 0.f };
    float    rest_mass{ 0.f };
    float    sph_h{ 0.f };
    float    pbf_Ks{ 0.f };
    float    pbf_Dq{ 0.f };
    float    cross_vis0{ 0.001 };
    float    cross_visInf{ 0.001 };
    float    cross_visBound{ 0.001 };
    float    cross_K{ 2 };
    float    cross_N{ CROSS_N };
    uint32_t total_particle_num{ 0 };
    float    dt{ 0.f };
};

}  // namespace Physika

#endif  // PHYSIKA_CONST_PACK_HPP
