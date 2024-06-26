/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-14
 * @description: implementation of World class
 * @version    : 1.0
 */

#include "world.hpp"

#include <algorithm>
#include <limits>
#include "scene.hpp"
#include "object.hpp"
#include "solver.hpp"

#include "src/entt/meta/resolve.hpp"

using entt::literals::operator"" _hs;
namespace Physika {

// implement Solver::id() here because it is highly related to the trick that we use its pointer address as id
std::uint64_t Solver::id() const
{
    return reinterpret_cast<std::uint64_t>(this);
}

World::World()
    : m_rm_solver_signal(), m_rm_solver_sink{ m_rm_solver_signal }
{
    entt::locator<entt::meta_ctx>::value_or();
    m_meta_context = entt::locator<entt::meta_ctx>::handle();
}

World::~World()
{
    this->clear();
}

World& World::instance()
{
    static World instance;
    return instance;
}

void World::clear()
{
    m_registry.clear();
    for (auto solver : m_solvers)
        delete solver;
    m_solvers.clear();
    m_rm_solver_sink.disconnect();  // disconnect all listeners
}

bool World::reset()
{
    bool ret = true;
    // call reset() method of all scenes
    auto scene_list = m_registry.view<std::shared_ptr<Scene>>();
    for (auto scene_id : scene_list)
    {
        auto scene = this->getSceneById(static_cast<std::uint64_t>(scene_id));
        if (!scene)
        {
            ret = false;
            continue;
        }
        ret &= scene->reset();
    }
    // call reset() method of all objects
    ret &= this->resetAllObjects();
    // call reset() method of all solvers
    for (auto solver : m_solvers)
    {
        ret &= solver->reset();
    }

    return ret;
}

Scene* World::createScene()
{
    auto  entity    = m_registry.create();
    auto& scene_ptr = m_registry.emplace<std::shared_ptr<Scene>>(entity, new Scene());
    scene_ptr->setId(static_cast<std::uint64_t>(entity));
    return scene_ptr.get();
}

bool World::destroyScene(Scene* scene)
{
    if (!scene)
        return false;

    m_registry.destroy(static_cast<PhysikaEntityType>(scene->id()));
    return true;
}

void World::destroyAllScenes()
{
    auto scene_list = m_registry.view<std::shared_ptr<Scene>>();
    for (auto scene : scene_list)
        m_registry.destroy(scene);
}

std::uint64_t World::sceneNum() const
{
    auto scene_list = m_registry.view<std::shared_ptr<Scene>>();
    return scene_list.size();
}

const Scene* World::getSceneById(std::uint64_t scene_id) const
{
    if (m_registry.valid(static_cast<PhysikaEntityType>(scene_id)) == false)
        return nullptr;
    return m_registry.get<std::shared_ptr<Scene>>(static_cast<PhysikaEntityType>(scene_id)).get();
}

Scene* World::getSceneById(std::uint64_t scene_id)
{
    if (m_registry.valid(static_cast<PhysikaEntityType>(scene_id)) == false)
        return nullptr;
    return m_registry.get<std::shared_ptr<Scene>>(static_cast<PhysikaEntityType>(scene_id)).get();
}

Object* World::createObject()
{
    auto  entity  = m_registry.create();
    auto& obj_ptr = m_registry.emplace<std::shared_ptr<Object>>(entity, new Object());
    obj_ptr->setId(static_cast<std::uint64_t>(entity));
    return obj_ptr.get();
}

bool World::destroyObject(Object* object)
{
    if (!object)
        return false;

    // remove from host scene
    auto host_scene = object->getHostScene();
    if (host_scene)
        host_scene->removeObject(object);

    m_registry.destroy(static_cast<PhysikaEntityType>(object->id()));
    return true;
}

bool World::destroyAllObjects()
{
    auto obj_list = m_registry.view<std::shared_ptr<Object>>();
    auto ret      = true;
    for (auto obj_id : obj_list)
    {
        auto obj = this->getObjectById(static_cast<std::uint64_t>(obj_id));
        ret &= this->destroyObject(obj);
    }
    return ret;
}

bool World::resetObject(Object* object)
{
    if (!object)
        return false;

    return this->resetObject(object->id());
}

bool World::resetObject(std::uint64_t obj_id)
{
    // call reset() method of all components of the object
    auto obj_com_type = entt::type_id<std::shared_ptr<Object>>();
    auto obj_entity   = static_cast<PhysikaEntityType>(obj_id);
    for (auto [com_info, storage] : m_registry.storage())
    {
        if (storage.contains(obj_entity))
        {
            if (obj_com_type == storage.type())
                continue;  // skip the special component type: Object
            entt::meta_type com_type = entt::resolve(storage.type());
            if (com_type)
            {
                entt::meta_func reset_func = com_type.func("reset"_hs);
                entt::meta_func get_func   = com_type.func("get"_hs);
                if (reset_func && get_func)
                {
                    auto com_instance = get_func.invoke({}, entt::forward_as_meta(m_registry), obj_entity);
                    if (!com_instance)
                        return false;
                    auto ret = reset_func.invoke(com_instance);
                    if (!ret)
                        return false;
                }
                else
                    return false;
            }
            else
                return false;
        }
    }
    return true;
}

bool World::resetAllObjects()
{
    // call reset() method of all objects
    auto obj_list = m_registry.view<std::shared_ptr<Object>>();
    auto ret      = true;
    for (auto obj : obj_list)
    {
        ret &= this->resetObject(static_cast<std::uint64_t>(obj));
    }

    return ret;
}

std::uint64_t World::objectNum() const
{
    auto obj_list = m_registry.view<std::shared_ptr<Object>>();
    return obj_list.size();
}

const Object* World::getObjectById(std::uint64_t obj_id) const
{
    if (m_registry.valid(static_cast<PhysikaEntityType>(obj_id)) == false)
        return nullptr;

    return m_registry.get<std::shared_ptr<Object>>(static_cast<PhysikaEntityType>(obj_id)).get();
}

Object* World::getObjectById(std::uint64_t obj_id)
{
    if (m_registry.valid(static_cast<PhysikaEntityType>(obj_id)) == false)
        return nullptr;

    return m_registry.get<std::shared_ptr<Object>>(static_cast<PhysikaEntityType>(obj_id)).get();
}

std::uint64_t World::getObjectId(const Object* obj) const
{
    if (!obj)
        return std::numeric_limits<std::uint64_t>::max();

    return obj->id();
}

bool World::removeAllComponentsFromObject(Object* object)
{
    if (!object)
        return false;

    return this->removeAllComponentsFromObject(object->id());
}

bool World::removeAllComponentsFromObject(std::uint64_t obj_id)
{
    auto obj = this->getObjectById(obj_id);
    if (!obj)
        return false;

    auto obj_com_type = entt::type_id<std::shared_ptr<Object>>();
    for (auto [com_info, storage] : m_registry.storage())
    {
        if (storage.contains(static_cast<PhysikaEntityType>(obj_id))
            && obj_com_type != storage.type())  // skip Object type, which is also an entity component
            storage.erase(static_cast<PhysikaEntityType>(obj_id));
    }
    obj->m_component_num = 0;
    return true;
}

std::uint64_t World::objectComponentNum(const Object* object) const
{
    if (!object)
        return 0;

    return object->componentNum();
}

std::uint64_t World::objectComponentNum(std::uint64_t obj_id) const
{
    auto obj = this->getObjectById(obj_id);
    return this->objectComponentNum(obj);
}

bool World::destroySolver(Solver* solver)
{
    if (!solver)
        return false;

    auto iter = std::find(m_solvers.begin(), m_solvers.end(), solver);
    if (iter != m_solvers.end())
        m_solvers.erase(iter);
    else
        return false;

    m_rm_solver_signal.publish(solver->id());  // publish a signal for scenes to remove references of the solver
    delete solver;
    return true;
}

void World::destroyAllSolvers()
{
    for (auto solver : m_solvers)
    {
        m_rm_solver_signal.publish(solver->id());  // publish a signal for scenes to remove references of the solver
        delete solver;
    }
    m_solvers.clear();
}

std::uint64_t World::solverNum() const
{
    return m_solvers.size();
}

const Solver* World::getSolverById(std::uint64_t solver_id) const
{
    auto iter = std::find_if(m_solvers.begin(), m_solvers.end(), [solver_id](Solver* solver) { return solver->id() == solver_id; });
    if (iter == m_solvers.end())
        return nullptr;
    return *iter;
}

Solver* World::getSolverById(std::uint64_t solver_id)
{
    auto iter = std::find_if(m_solvers.begin(), m_solvers.end(), [solver_id](Solver* solver) { return solver->id() == solver_id; });
    if (iter == m_solvers.end())
        return nullptr;
    return *iter;
}

bool World::isDanglingSolver(const Solver* solver) const
{
    auto iter = std::find(m_solvers.begin(), m_solvers.end(), const_cast<Solver*>(solver));
    return iter == m_solvers.end();
}

bool World::step()
{
    auto scene_list = m_registry.view<std::shared_ptr<Scene>>();
    for (auto scene : scene_list)
    {
        auto scene_ptr = m_registry.get<std::shared_ptr<Scene>>(scene);
        if (!scene_ptr)
            return false;
        auto ret = scene_ptr->step();
        if (!ret)
            return false;
    }
    return true;
}

bool World::run()
{
    auto scene_list = m_registry.view<std::shared_ptr<Scene>>();
    for (auto scene : scene_list)
    {
        auto scene_ptr = m_registry.get<std::shared_ptr<Scene>>(scene);
        if (!scene_ptr)
            return false;
        auto ret = scene_ptr->run();
        if (!ret)
            return false;
    }
    return true;
}

void World::registerSolver(Solver* solver)
{
    if (!solver)
        return;

    if (std::find(m_solvers.begin(), m_solvers.end(), solver) != m_solvers.end())
        return;

    m_solvers.push_back(solver);
}

void World::incrementObjectComponentNum(Object* object)
{
    if (object)
        object->m_component_num++;
}

void World::decrementObjectComponentNum(Object* object)
{
    if (object && object->m_component_num > 0)
        object->m_component_num--;
}

}  // namespace Physika
