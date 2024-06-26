#include "framework/object.hpp"
#include "collision/interface/mesh_hf_collision_solver.hpp"
#include "collision/interface/mesh_hf_collision_detect.hpp"
#include "collision/interface/collidable_hf.hpp"
#include "collision/interface/collidable_trianglemesh.hpp"
namespace Physika {
MeshHeightFieldCollisionSolver::MeshHeightFieldCollisionSolver()
    : Solver(), m_is_init(false), m_mesh_heightField_object(nullptr)
{
}

MeshHeightFieldCollisionSolver::~MeshHeightFieldCollisionSolver()
{
    this->reset();
}

bool MeshHeightFieldCollisionSolver::initialize()
{
    std::cout << "MeshHeightFieldCollisionSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool MeshHeightFieldCollisionSolver::isInitialized() const
{
    std::cout << "MeshRayCollisionSolverr::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool MeshHeightFieldCollisionSolver::reset()
{
    std::cout << "MeshHeightFieldCollisionSolver::reset() sets the solver to newly constructed state.\n";
    m_mesh_heightField_object->reset();
    m_cur_time      = 0.0;
    m_is_init       = false;
    return true;
}

bool MeshHeightFieldCollisionSolver::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_mesh_heightField_object)
    {
        return false;
    }
    // Update till termination

    float totaltime = 0.f;
    int   step_id   = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double      dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        cudaEvent_t start, stop;
        float       milliseconds = 0.f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        m_cur_time += dt;
        m_config.m_dt = dt;
        step();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        totaltime += milliseconds;
        ++step_id;
        printf("step_id: %d , frame time: %f ms ! frame: %f ! \n", step_id, totaltime / step_id, 1000 * step_id / totaltime);
    }
    return true;
}

bool MeshHeightFieldCollisionSolver::step()
{
    auto &meshes      = m_mesh_heightField_object->getComponent<MeshHeightFieldCollisionComponent>()->m_trimesh;
    auto &heightField = m_mesh_heightField_object->getComponent<MeshHeightFieldCollisionComponent>()->m_heightField;

    MeshHfCollisionDetect solver(*meshes, *heightField);
    solver.setMaxCollision(this->m_maxCollision);
    solver.execute();
    solver.getResult(m_mesh_heightField_object->getComponent<MeshHeightFieldCollisionComponent>()->m_collision_id, 
                        m_mesh_heightField_object->getComponent<MeshHeightFieldCollisionComponent>()->m_collision_normal,
                     m_mesh_heightField_object->getComponent<MeshHeightFieldCollisionComponent>()->m_counter);
    return true;
}

bool MeshHeightFieldCollisionSolver::isApplicable(const Object* object) const
{
    std::cout << "MeshHeightFieldCollisionSolver::isApplicable() checks if object has MeshHeightFieldCollisionComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<MeshHeightFieldCollisionComponent>();
}

bool MeshHeightFieldCollisionSolver::attachObject(Object* object)
{
    std::cout << "MeshHeightFieldCollisionSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }

     m_mesh_heightField_object = object;

    return true;
}

bool MeshHeightFieldCollisionSolver::detachObject(Object* object)
{
    std::cout << "MeshHeightFieldCollisionSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_mesh_heightField_object == object)
    {
        m_mesh_heightField_object = nullptr;
        std::cout << "    Mesh heightField detached.\n";
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void MeshHeightFieldCollisionSolver::clearAttachment()
{
    std::cout << "MeshHeightFieldCollisionSolver::clearAttachment() clears the target list of the solver.\n";
    m_mesh_heightField_object = nullptr;
}

//int MeshHeightFieldCollisionSolver::getMaxCollision()
//{
//    return 0;
//}
//
//void MeshHeightFieldCollisionSolver::setMaxCollision(int maxCollision)
//{
//}

}  // namespace Physika