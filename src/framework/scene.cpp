/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-14
 * @description: implementation of Scene class
 * @version    : 1.0
 */

#include "scene.hpp"

#include <algorithm>
#include <limits>
#include "object.hpp"
#include "solver.hpp"
#include "world.hpp"

namespace Physika {

Scene::Scene()
    : m_id(std::numeric_limits<std::uint64_t>::max())
    , m_attach_info_outdated(true)
{
    World::instance().connectSolverRemoveListener<&Scene::removeSolverById>(*this);  // connect listener with signal
}

Scene::~Scene()
{
    World::instance().disconnectSolverRemoveListener<&Scene::removeSolverById>(*this);  // disconnect listener with signal
}

std::uint64_t Scene::id() const
{
    return m_id;
}

bool Scene::reset()
{
    bool scene_ret = true;
    for (auto obj : m_objects)
    {
        auto ret = obj->reset();
        scene_ret &= ret;
    }
    for (auto solver : m_solvers)
    {
        auto ret = solver->reset();
        scene_ret &= ret;
    }
    return scene_ret;
}

bool Scene::addObject(Object* object)
{
    if (!object)
        return false;

    if (object->getHostScene() != nullptr)  // object in another scene
        return false;

    if (std::find(m_objects.begin(), m_objects.end(), object) != m_objects.end())  // already in scene
        return true;

    m_objects.push_back(object);
    object->setHostScene(this);
    m_attach_info_outdated = true;
    return true;
}

bool Scene::addObjectById(std::uint64_t id)
{
    Object* obj = World::instance().getObjectById(id);
    return this->addObject(obj);
}

bool Scene::removeObject(Object* object)
{
    if (!object)
        return false;

    auto iter = std::find(m_objects.begin(), m_objects.end(), object);
    if (iter == m_objects.end())  // not in scene
        return false;

    m_objects.erase(iter);
    object->setHostScene(nullptr);
    m_attach_info_outdated = true;
    return true;
}

bool Scene::removeObjectById(std::uint64_t id)
{
    Object* obj = this->getObjectById(id);
    return this->removeObject(obj);
}

void Scene::removeAllObjects()
{
    for (auto obj : m_objects)
        obj->setHostScene(nullptr);
    m_objects.clear();
    m_attach_info_outdated = true;
}

std::uint64_t Scene::objectNum() const
{
    return m_objects.size();
}

const Object* Scene::getObjectById(std::uint64_t obj_id) const
{
    auto iter = std::find_if(m_objects.begin(), m_objects.end(), [obj_id](Object* obj) { return obj->id() == obj_id; });
    if (iter == m_objects.end())
        return nullptr;
    return *iter;
}

Object* Scene::getObjectById(std::uint64_t obj_id)
{
    auto iter = std::find_if(m_objects.begin(), m_objects.end(), [obj_id](Object* obj) { return obj->id() == obj_id; });
    if (iter == m_objects.end())
        return nullptr;
    return *iter;
}

bool Scene::addSolver(Solver* solver)
{
    if (!solver)
        return false;

    if (std::find(m_solvers.begin(), m_solvers.end(), solver) != m_solvers.end())
        return true;

    m_solvers.push_back(solver);
    m_attach_info_outdated = true;
    return true;
}

bool Scene::addSolverById(std::uint64_t id)
{
    Solver* solver = World::instance().getSolverById(id);
    return this->addSolver(solver);
}

bool Scene::removeSolver(Solver* solver)
{
    if (!solver)
        return false;

    auto iter = std::find(m_solvers.begin(), m_solvers.end(), solver);
    if (iter == m_solvers.end())  // not in scene
        return false;

    m_solvers.erase(iter);
    m_attach_info_outdated = true;
    return true;
}

bool Scene::removeSolverById(std::uint64_t id)
{
    Solver* solver = this->getSolverById(id);
    return this->removeSolver(solver);
}

void Scene::removeAllSolvers()
{
    m_solvers.clear();
    m_attach_info_outdated = true;
}

std::uint64_t Scene::solverNum() const
{
    return m_solvers.size();
}

const Solver* Scene::getSolverById(std::uint64_t solver_id) const
{
    auto iter = std::find_if(m_solvers.begin(), m_solvers.end(), [solver_id](Solver* solver) { return solver->id() == solver_id; });
    if (iter == m_solvers.end())
        return nullptr;
    return *iter;
}

Solver* Scene::getSolverById(std::uint64_t solver_id)
{
    auto iter = std::find_if(m_solvers.begin(), m_solvers.end(), [solver_id](Solver* solver) { return solver->id() == solver_id; });
    if (iter == m_solvers.end())
        return nullptr;
    return *iter;
}

bool Scene::step()
{
    if (m_attach_info_outdated)
    {
        auto ret = updateSolverAttachInfo();
        if (!ret)
            return false;
    }

    for (auto solver : m_solvers)
    {
        if (!solver)
            return false;

        if (solver->isInitialized() == false)
            solver->initialize();

        auto ret = solver->step();
        if (!ret)
            return false;
    }
    return true;
}

bool Scene::run()
{
    if (m_attach_info_outdated)
    {
        auto ret = updateSolverAttachInfo();
        if (!ret)
            return false;
    }

    for (auto solver : m_solvers)
    {
        if (!solver)
            return false;

        if (solver->isInitialized() == false)
            solver->initialize();

        auto ret = solver->run();
        if (!ret)
            return false;
    }
    return true;
}

void Scene::setId(std::uint64_t id)
{
    m_id = id;
}

bool Scene::updateSolverAttachInfo()
{
    if (!m_attach_info_outdated)
        return true;

    // attach applicable objects to solvers
    for (auto solver : m_solvers)
    {
        if (!solver)
            return false;

        solver->clearAttachment();

        for (auto object : m_objects)
        {
            if (!object)
                return false;

            if (solver->isApplicable(object))
            {
                auto ret = solver->attachObject(object);
                if (!ret)
                    return false;
            }
        }
    }
    return true;
}

}  // namespace Physika