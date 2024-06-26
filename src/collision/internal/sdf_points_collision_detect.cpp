#include "collision/interface/sdf_points_collision_detect.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.hpp>

namespace Physika {

extern void PointsSdfCollision(vec3f* positions, int* points_num, float* radius, DistanceField3D* sdf, int* num_collisions, int* m_collision_id, vec3f* m_collision_normal, float* m_collision_distance);

SDFPointsCollisionDetect::SDFPointsCollisionDetect(vec3f* points, DistanceField3D* sdf, float radius, int num)
{
	m_points = points;
    m_distanceField = sdf;
    m_radius = radius;
    m_num_points = num;
}

SDFPointsCollisionDetect::SDFPointsCollisionDetect():m_radius(0.0f),m_num_points(0)  {}
SDFPointsCollisionDetect::~SDFPointsCollisionDetect() {}

void SDFPointsCollisionDetect::setPoints(vec3f* points)
{
    m_points = points;
}

void SDFPointsCollisionDetect::setRadius(float radius) 
{
    m_radius = radius;
}

void SDFPointsCollisionDetect::setDistanceField(DistanceField3D* sdf)
{
    m_distanceField = sdf;
}

    void SDFPointsCollisionDetect::getCollisionResult(int& collisionNum, int*& collisionIds, vec3f*& collisionNormals, float*& collisionDistance) const
{

    collisionNum      = m_num_collisions;
    collisionIds      = m_collision_id;
    collisionNormals  = m_collision_normal;
    collisionDistance = m_collision_distance;
    return;
}

int SDFPointsCollisionDetect::getCollisionNums() const
{
    return m_num_collisions;
}

const int* SDFPointsCollisionDetect::getCollisionIds() const
{
    return m_collision_id;
}

const vec3f* SDFPointsCollisionDetect::getCollisionNormals() const
{
    return m_collision_normal;
}

const float* SDFPointsCollisionDetect::getCollisionDistance() const
{
    return m_collision_distance;
}

bool SDFPointsCollisionDetect::execute() {
    PointsSdfCollision(m_points, &m_num_points, &m_radius, m_distanceField, &m_num_collisions, m_collision_id, m_collision_normal, m_collision_distance);
    return true;
}
    
}