#include "collision/interface/particles_collision_solver.hpp"

namespace Physika {

ParticlesCollisionSolver::ParticlesCollisionSolver()
{
}


ParticlesCollisionSolver::~ParticlesCollisionSolver()
{
    this->reset();
}

bool ParticlesCollisionSolver::initialize()
{
    std::cout << "ParticlesCollisionSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool ParticlesCollisionSolver::isInitialized() const
{
    std::cout << "ParticlesCollisionSolverr::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool ParticlesCollisionSolver::reset()
{
    std::cout << "ParticlesCollisionSolver::reset() sets the solver to newly constructed state.\n";
    m_is_init = false;
    m_particles->reset();

    return true;
}

bool ParticlesCollisionSolver::run()
{
    return step();
}

bool ParticlesCollisionSolver::step()
{
    std::cout << "ParticlesCollisionSolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "error: solver not initialized.\n";
        return false;
    }
    if (!m_particles)
    {
        std::cout << "error: particles not attached to the solver.\n";
        return false;
    }
    std::cout << "ParticlesCollisionSolver::run() applied to mesh " << m_particles->id() << ".\n";
    doCollision();
    return true;
}

bool ParticlesCollisionSolver::isApplicable(const Object* object) const
{
    std::cout << "ParticlesCollisionSolver::isApplicable() checks if object has CollidablePointsComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<ParticlesCollisionComponent>();
}

bool ParticlesCollisionSolver::attachObject(Object* object)
{
    std::cout << "ParticlesCollisionSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<ParticlesCollisionComponent>())
    {
        std::cout << "    object attached as collidable points.\n";
        m_particles = object;
    }

    return true;
}

bool ParticlesCollisionSolver::detachObject(Object* object)
{
    std::cout << "ParticlesCollisionSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_particles == object)
    {
        m_particles = nullptr;
        std::cout << "    Particles detached.\n";
        return true;
    }

    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void ParticlesCollisionSolver::clearAttachment()
{
    std::cout << "ParticlesCollisionSolver::clearAttachment() clears the target list of the solver.\n";
    m_particles = nullptr;
}

bool ParticlesCollisionSolver::doCollision()
{
    m_particles->getComponent<ParticlesCollisionComponent>()->m_pairs.clear();
    auto& radius = m_particles->getComponent<ParticlesCollisionComponent>()->m_radius;
    auto& points = m_particles->getComponent<ParticlesCollisionComponent>()->m_pos;
    int   num    = m_particles->getComponent<ParticlesCollisionComponent>()->m_num;

    for (int i = 0; i < num; i++)
    {
        vec3f p = points[i];
        for (int j = i + 1; j < num; j++)
        {
            vec3f q = points[j];
            if ((p - q).length() < radius)
            {
                m_particles->getComponent<ParticlesCollisionComponent>()->m_pairs.push_back(id_pair(i, j, false));
            }
        }
    }

    return true;

}

}  // namespace Physika