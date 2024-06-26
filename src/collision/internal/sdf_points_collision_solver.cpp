#include "collision/interface/sdf_points_collision_solver.hpp"
#include "collision/interface/collidable_sdf.hpp"
#include "collision/interface/collidable_points.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.hpp>
//#include "sm_20_atomic_functions.hpp"                                                                                                                                             
namespace Physika {

extern void PointsSdfCollision(vec3f* positions, int* points_num, float* radius, DistanceField3D* sdf, int* num_collisions, int* m_collision_id, vec3f* m_collision_normal, float* m_collision_distance);

SDFPointsCollisionSolver::SDFPointsCollisionSolver()
    : Solver(), m_is_init(false)
{
    m_maxCollision = 1000;
    //cudaMalloc(( void** )&m_num_collisions, sizeof(int));
    //cudaMalloc(( void** )m_collision_id, sizeof(int)*max_collisionNum);
    //cudaMalloc(( void** )m_collision_normal, sizeof(vec3f) * max_collisionNum);
    //cudaMalloc(( void** )m_collision_distance, sizeof(float) * max_collisionNum);
}

SDFPointsCollisionSolver::~SDFPointsCollisionSolver()
{
    this->reset();
}

bool SDFPointsCollisionSolver::initialize()
{
    std::cout << "SDFPointsCollisionSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool SDFPointsCollisionSolver::isInitialized() const
{
    std::cout << "SDFPointsCollisionSolver::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

// TODO
bool SDFPointsCollisionSolver::reset()
{
    std::cout << "SDFPointsCollisionSolver::reset() sets the solver to newly constructed state.\n";
    m_is_init   = false;

    return true;
}

// TODO
bool SDFPointsCollisionSolver::run()
{
    return step();
}

bool SDFPointsCollisionSolver::step()
{
    std::cout << "SDFPointsCollisionSolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "error: solver not initialized.\n";
        return false;
    }
    if (!m_sdf_points_object)
    {
        std::cout << "error: SDF or points not attached to the solver.\n";
        return false;
    }
    std::cout << "SDFPointsCollisionSolver::run() applied to SDF " << m_sdf_points_object->id() << ".\n";
    doCollision();
    return true;
}

bool SDFPointsCollisionSolver::isApplicable(const Object* object) const
{
    std::cout << "SDFPointsCollisionSolver::isApplicable() checks if object has CollidableMeshComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<SDFPointsCollisionComponent>();
}

bool SDFPointsCollisionSolver::attachObject(Object* object)
{
    std::cout << "SDFPointsCollisionSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<SDFPointsCollisionComponent>())
    {
        std::cout << "    object attached as SDFPointsCollision.\n";
    }
    m_sdf_points_object = object;
    return true;
}

bool SDFPointsCollisionSolver::detachObject(Object* object)
{
    std::cout << "SDFPointsCollisionSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_sdf_points_object == object)
    {
        m_sdf_points_object = nullptr;
        std::cout << "    SDF Points obejct detached.\n";
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void SDFPointsCollisionSolver::clearAttachment()
{
    std::cout << "SDFPointsCollisionSolver::clearAttachment() clears the target list of the solver.\n";
    m_sdf_points_object = nullptr;
}

bool SDFPointsCollisionSolver::doCollision()
{
    auto             sdfComponent    = m_sdf_points_object->getComponent<CollidableSDFComponent>();
    auto             pointsComponent = m_sdf_points_object->getComponent<CollidablePointsComponent>();
    auto             num             = m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_num;

    vec3f*           positions;
    int*             points_num;
    DistanceField3D* sdf;
    REAL* radius;

    cudaMalloc(( void** )&positions, sizeof(vec3f) * num);
    cudaMalloc(( void** )&points_num, sizeof(int));
    cudaMalloc(( void** )&sdf, sizeof(sdf));
    cudaMalloc(( void** )&radius, sizeof(float));
    cudaMalloc(( void** )&m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_id, 10000 * sizeof(int));
    cudaMalloc(( void** )&m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_normal, 10000 * sizeof(vec3f));
    cudaMalloc(( void** )&m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_distance, 10000 * sizeof(float));


    cudaMemcpy(positions, pointsComponent->m_pos, sizeof(vec3f) * pointsComponent->m_num, cudaMemcpyHostToDevice);
    cudaMemcpy(points_num, &pointsComponent->m_num, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sdf, sdfComponent->m_sdf, sizeof(sdf), cudaMemcpyHostToDevice);
    cudaMemcpy(radius, &pointsComponent->m_radius, sizeof(REAL), cudaMemcpyHostToDevice);
    // a simple sdf collision detect for cpu

    PointsSdfCollision(positions, points_num, radius, sdf, &m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_num_collisions, m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_id, m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_normal, m_sdf_points_object->getComponent<SDFPointsCollisionComponent>()->m_collision_distance);
    //cudaMemcpy(&m_num_collisions, num_collisions, sizeof(int), cudaMemcpyDeviceToHost)
    //cudaMemcpy(m_collision_id, &m_collision_id, sizeof(int) * max_collisionNum, cudaMemcpyDeviceToHost);
    //cudaMemcpy(m_collision_normal, &m_collision_normal, sizeof(vec3f) * max_collisionNum, cudaMemcpyDeviceToHost);
    //cudaMemcpy(m_collision_distance, &m_collision_distance, sizeof(float) * max_collisionNum, cudaMemcpyDeviceToHost);
    return true;
}

}