#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/distance_field3d.hpp"

namespace Physika {

__global__ void PointsSdfCollision_device(vec3f* positions, int* points_num, float* radius, DistanceField3D* sdf, int* num_collisions, int* m_collision_id, vec3f* m_collision_normal, float* m_collision_distance)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= points_num[0])
        return;
    vec3f point = positions[tid];
    vec3f normal;
    float dis;
    sdf->getDistance(point, dis, normal);
    normal *= -1.0f;
    if (dis <= radius[0])
    {
        int id                   = atomicAdd(&num_collisions[0], 1) - 1;
        m_collision_id[id]       = tid;
        m_collision_distance[id] = dis;
        m_collision_normal[id]   = normal;
    }
}

void PointsSdfCollision(vec3f* positions, int* points_num, float* radius, DistanceField3D* sdf, int* num_collisions, int* m_collision_id, vec3f* m_collision_normal, float* m_collision_distance) {
    PointsSdfCollision_device<<<1, 1024>>>(positions, points_num, radius, sdf, num_collisions, m_collision_id, m_collision_normal, m_collision_distance);
}

}