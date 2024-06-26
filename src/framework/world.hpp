/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-08
 * @description: declaration of World class, which is a library-scope manager
 * @version    : 1.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include "src/entt/core/hashed_string.hpp"
#include "src/entt/entity/registry.hpp"
#include "src/entt/meta/factory.hpp"
#include "src/entt/signal/sigh.hpp"
#include "src/entt/locator/locator.hpp"

namespace Physika {

class Scene;
class Object;
class Solver;

/**
 * World is the registry/manager of everything: from concepts like Scenes, Objects, and Solvers,
 * to settings like logger, cpu memory management, and gpu memory management, and more.
 *
 * World is designed as a final class and a singleton.
 *
 * World takes ownership of everything created by it.
 *
 * Note on scene/object/solver id: a unique id once it is created, won't change during its lifecyle.
 *
 * Note on dangling solver: Those not created by World, e.g., created by explicit call of its constructor.
 * Physika does not forbid the creation of solver bypassing World (for flexbility),
 * however it is strongly recommended to use World as the registry.
 * There's no warranty of correct behaviors while using dangling solver, e.g., ids may coincide with
 * registered ones. The developers should code at their own risks in such cases.
 */
class World final
{
public:
    /**
     * @brief get the instance of the World
     *        first call of this method will return an empty world
     *
     * @return    reference to the world singleton
     */
    static World& instance();

    /**
     * @brief clear the world, destroy everything in it.
     *        MUST be called before program exits.
     */
    void clear();

    /**
     * @brief reset everything in the world to initial state (contents not removed)
     *
     * @return    true if success, otherwise return false
     */
    bool reset();

    //////////////////// Scene Management ////////////////////

    /**
     * @brief create an empty scene in the world
     * @return    pointer to the newly created scene, return nullptr if error occurs
     */
    Scene* createScene();

    /**
     * @brief destroy the given scene, only applicable to scene created by world
     *        All contents (objects/solvers) in the scene will be destroyed, remove them
     *        from the scene beforehand if they're still needed.
     *
     * @param[in] scene    the scene to be destroyed
     *
     * @return    true if the scene is successfully destroyed
     *            false if error occurs, e.g., scene is nullptr
     */
    bool destroyScene(Scene* scene);

    /**
     * @brief destroy all scenes (and everything therein) that are created by the world
     */
    void destroyAllScenes();

    /**
     * @brief get the total number of scenes in the world
     * @return    number of scenes
     */
    std::uint64_t sceneNum() const;

    /**
     * @brief get the scene with given id
     *
     * @param[in] scene_id    id of the scene
     *
     * @return    pointer to the scene with given id, return nullptr if not found
     */
    const Scene* getSceneById(std::uint64_t scene_id) const;
    Scene*       getSceneById(std::uint64_t scene_id);

    //////////////////// Object Management ////////////////////

    /**
     * @brief create an object in the world
     * @return    pointer to the newly created object, return nullptr if error occurs
     */
    Object* createObject();

    /**
     * @brief destroy the given object, only applicable to object created by world
     *        The object will also be removed from the host scene
     *
     * @param[in] object    the object to be destroyed
     *
     * @return    true if the object is successfully destroyed
     *            false if error occurs, e.g., object is nullptr
     */
    bool destroyObject(Object* object);

    /**
     * @brief destroy all objects that are created by the world
     *        References in the host scenes will also be removed
     *
     * @return    true if all objects are successfully destroyed
     *            false if error occurs
     */
    bool destroyAllObjects();

    /**
     * @brief reset the given object to initial state, the components are reset as well (not removed)
     *
     * @param[in] object    pointer to the object to be reset
     *
     * @return    true if the object is successfully reset
     *            false if error occurs, e.g., object is nullptr
     */
    bool resetObject(Object* object);

    /**
     * @brief reset the given object to initial state, the components are reset as well (not removed)
     *
     * @param[in] obj_id    id of the object to be reset
     *
     * @return    true if the object is successfully reset
     *            false if error occurs, e.g., the id is invalid
     */
    bool resetObject(std::uint64_t obj_id);

    /**
     * @brief reset all objects to initial state
     *
     * @return    true if all objects are successfully reset
     *            false if error occurs
     */
    bool resetAllObjects();

    /**
     * @brief get the total number of objects in the world
     * @return     number of objects
     */
    std::uint64_t objectNum() const;

    /**
     * @brief get the object with given id
     *
     * @param[in] obj_id    id of the object
     *
     * @return    pointer to the object with given id, return nullptr if not found
     */
    const Object* getObjectById(std::uint64_t obj_id) const;
    Object*       getObjectById(std::uint64_t obj_id);

    /**
     * @brief get the id of given object
     *
     * @param[in] obj    pointer of the object
     *
     * @return    id of the given object, return SIZE_T::MAX if error occurs, e.g., obj is nullptr
     */
    std::uint64_t getObjectId(const Object* obj) const;

    /**
     * @brief create and initialize a component of given type, add add it to specified object
     * @tparam ComponentType    type of the component
     * @tparam Args             variable argument list that constructor of ComponentType expects
     *
     * @param[in] object    pointer to the object that the component is added to
     * @param[in] args      arguments of ComponentType constructor
     *
     * @return    true if the component is successfully added
     *            false if error occurs, e.g., the component already exists
     */
    template <typename ComponentType, typename... Args>
    bool addComponentToObject(Object* object, Args&&... args);

    /**
     * @brief create and initialize a component of given type, add add it to specified object
     * @tparam ComponentType    type of the component
     * @tparam Args             variable argument list that constructor of ComponentType expects
     *
     * @param[in] obj_id    id of the object that the component is added to
     * @param[in] args      arguments of ComponentType constructor
     *
     * @return    true if the component is successfully added
     *            false if error occurs, e.g., the component already exists
     */
    template <typename ComponentType, typename... Args>
    bool addComponentToObject(std::uint64_t obj_id, Args&&... args);

    /**
     * @brief check whether the object has a component of specific type
     * @tparam ComponentType    type of the component
     * @param[in] object    pointer to the object
     *
     * @return true if the object has the component, otherwise return false
     */
    template <typename ComponentType>
    bool checkObjectHasComponent(const Object* object) const;

    /**
     * @brief check whether the object has a component of specific type
     * @tparam ComponentType    type of the component
     * @param[in] obj_id   id of the object
     *
     * @return true if the object has the component, otherwise return false
     */
    template <typename ComponentType>
    bool checkObjectHasComponent(std::uint64_t obj_id) const;

    /**
     * @brief remove specific component from the object
     * @tparam ComponentType    type of the componenet
     *
     * @param[in] object    pointer to the object that the component is removed from
     *
     * @return    true if the component is successfully removed
     *            false if error occurs, e.g., the object has no such component
     */
    template <typename ComponentType>
    bool removeComponentFromObject(Object* object);

    /**
     * @brief remove specific component from the object
     * @tparam ComponentType    type of the componenet
     *
     * @param[in] obj_id    id of the object that the component is removed from
     *
     * @return    true if the component is successfully removed
     *            false if error occurs, e.g., the object has no such component
     */
    template <typename ComponentType>
    bool removeComponentFromObject(std::uint64_t obj_id);

    /**
     * @brief remove all components from the object
     *
     * @param[in] object    pointer to the object that the components are removed from
     *
     * @return    true if all components are successfully removed
     *            false if error occurs, e.g., object is nullptr
     *
     */
    bool removeAllComponentsFromObject(Object* object);

    /**
     * @brief remove all components from the object
     *
     * @param[in] obj_id    id of the object that the components are removed from
     *
     * @return    true if all components are successfully removed
     *            false if error occurs, e.g., object is not registered
     *
     */
    bool removeAllComponentsFromObject(std::uint64_t obj_id);

    /**
     * @brief get the total component number of an object
     *
     * @param[in] object    pointer of the object
     *
     * @return    the component number, return 0 if error occurs
     */
    std::uint64_t objectComponentNum(const Object* object) const;

    /**
     * @brief get the total component number of an object
     *
     * @param[in] obj_id    id of the object
     *
     * @return    the component number, return 0 if error occurs
     */
    std::uint64_t objectComponentNum(std::uint64_t obj_id) const;

    /**
     * @brief get the component with given type from an object
     * @tparam ComponentType    the component type
     *
     * @param[in] object    pointer to the object
     *
     * @return    pointer to the component, return nullptr if not found
     */
    template <typename ComponentType>
    const ComponentType* getObjectComponent(const Object* object) const;

    template <typename ComponentType>
    ComponentType* getObjectComponent(const Object* object);

    /**
     * @brief get the component with given type from an object
     * @tparam ComponentType    the component type
     *
     * @param[in] obj_id    id of the object
     *
     * @return    pointer to the component, return nullptr if not found
     */
    template <typename ComponentType>
    const ComponentType* getObjectComponent(std::uint64_t obj_id) const;

    template <typename ComponentType>
    ComponentType* getObjectComponent(std::uint64_t obj_id);

    //////////////////// Solver Management ////////////////////

    /**
     * @brief create a solver with given type
     *
     * @tparam SolverType    type of of the solver to be created
     * @tparam Args          variable argument list to solver constructor
     *
     * @param[in] args    argument list to solver constructor
     *
     * @return    pointer to the newly created solver, return nullptr if error occurs
     */
    template <typename SolverType, typename... Args>
    SolverType* createSolver(Args&&... args);

    /**
     * @brief destroy the given solver, only applicable to solver created by world
     *        Ths solver will also be removed from the host scene
     *
     * @param[in] solver    pointer to the solver to be destroyed
     *
     * @return    true if the solver is successfully destroyed
     *            false if error occurs, e.g., the solver is not created by world API
     */
    bool destroySolver(Solver* solver);

    /**
     * @brief destroy all solvers that are created by the world
     *        References in the host scene will also be removed
     */
    void destroyAllSolvers();

    /**
     * @brief get the number of registered solvers
     *
     * @return    number of solvers managed by World
     */
    std::uint64_t solverNum() const;

    /**
     * @brief get the solver with given id
     *
     * @param[in] solver_id    id of the solver
     *
     * @return    pointer to the solver with given id, return nullptr if not found
     */
    const Solver* getSolverById(std::uint64_t solver_id) const;
    Solver*       getSolverById(std::uint64_t solver_id);

    /**
     * @brief check whether given solver is a dangling solver
     *        Dangling means the solver is not created by World API
     *        Nullptr is considered dangling
     *
     * @param[in] solver    pointer to the solver to be checked
     *
     * @return    true if the solver is dangling, false otherwise
     */
    bool isDanglingSolver(const Solver* solver) const;

    //////////////////// Update Methods ////////////////////

    /**
     * @brief run the step() methods of all scenes in the world
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool step();

    /**
     * @brief run the run() methods of all scenes in the world
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool run();

public:  // advanced/internal methods && data that need to be public
    enum class PhysikaEntityType : std::uint64_t
    {
    };

    /**
     * @brief connect/disconnect a listener for m_rm_solver_signal
     *
     * @tparam    Candidate            Function or member to connect to the signal
     * @tparam    Type                 type of class or type of payload, if any
     * @param[in] value_or_instance    a valid object that fits the purpose, if anay
     */
    template <auto Candidate, typename... Type>
    void connectSolverRemoveListener(Type&&... value_or_instance);

    template <auto Candidate, typename... Type>
    void disconnectSolverRemoveListener(Type&&... value_or_instance);

private:
    World();
    ~World();
    // disable copy&&move
    World(const World&)            = delete;
    World(World&&)                 = delete;
    World& operator=(const World&) = delete;
    World& operator=(World&&)      = delete;

    void registerSolver(Solver* solver);  // add a solver to registry
    void incrementObjectComponentNum(Object* object);
    void decrementObjectComponentNum(Object* object);

private:
    entt::basic_registry<PhysikaEntityType>     m_registry;          //!< ECS registry
    std::vector<Solver*>                        m_solvers;           //!< registry of solvers
    entt::sigh<bool(std::uint64_t)>             m_rm_solver_signal;  //!< signal handler for detroying solvers
    entt::sink<entt::sigh<bool(std::uint64_t)>> m_rm_solver_sink;    //!< sink for rm_solver signals && listeners
    entt::locator<entt::meta_ctx>::node_type    m_meta_context;      //!< context of entt, for across boundary (dll) use
};

//////////////////// implementation of template methods ////////////////////

template <typename ComponentType>
ComponentType& get(entt::basic_registry<World::PhysikaEntityType>& registry, World::PhysikaEntityType entity)
{
    return registry.template get<ComponentType>(entity);
}

template <typename ComponentType, typename... Args>
bool World::addComponentToObject(Object* object, Args&&... args)
{
    if (!object)
        return false;

    return this->addComponentToObject<ComponentType>(this->getObjectId(object), std::forward<Args>(args)...);
}

template <typename ComponentType, typename... Args>
bool World::addComponentToObject(std::uint64_t obj_id, Args&&... args)
{
    if (m_registry.any_of<ComponentType>(static_cast<PhysikaEntityType>(obj_id)))
        return false;

    auto obj = this->getObjectById(obj_id);
    if (!obj)
        return false;
    this->incrementObjectComponentNum(obj);

    entt::locator<entt::meta_ctx>::reset(m_meta_context);  // make sure the same context is used
    // Note: wierd that 'using entt::literals::operator"" _hs' leads to C3688 error on MSVC, but using entire namespace won't.
    using namespace entt::literals;
    // using entt::literals::operator"" _hs;
    m_registry.emplace<ComponentType>(static_cast<PhysikaEntityType>(obj_id), std::forward<Args>(args)...);
    entt::meta<ComponentType>().template func<&ComponentType::reset>("reset"_hs)  // register reset() method to reflection system
        .template func<&get<ComponentType>, entt::as_ref_t>("get"_hs);            // register get() method to reflection system
    return true;
}

template <typename ComponentType>
bool World::checkObjectHasComponent(const Object* object) const
{
    if (!object)
        return false;

    return this->checkObjectHasComponent<ComponentType>(this->getObjectId(object));
}

template <typename ComponentType>
bool World::checkObjectHasComponent(std::uint64_t obj_id) const
{
    return m_registry.any_of<ComponentType>(static_cast<PhysikaEntityType>(obj_id));
}

template <typename ComponentType>
bool World::removeComponentFromObject(Object* object)
{
    if (!object)
        return false;

    return this->removeComponentFromObject<ComponentType>(this->getObjectId(object));
}

template <typename ComponentType>
bool World::removeComponentFromObject(std::uint64_t obj_id)
{
    if (!m_registry.any_of<ComponentType>(static_cast<PhysikaEntityType>(obj_id)))
        return false;

    auto obj = this->getObjectById(obj_id);
    if (!obj)
        return false;
    this->decrementObjectComponentNum(obj);

    m_registry.erase<ComponentType>(static_cast<PhysikaEntityType>(obj_id));
    return true;
}

template <typename ComponentType>
const ComponentType* World::getObjectComponent(const Object* object) const
{
    if (!object)
        return nullptr;

    return this->getObjectComponent<ComponentType>(this->getObjectId(object));
}

template <typename ComponentType>
ComponentType* World::getObjectComponent(const Object* object)
{
    if (!object)
        return nullptr;

    return this->getObjectComponent<ComponentType>(this->getObjectId(object));
}

template <typename ComponentType>
const ComponentType* World::getObjectComponent(std::uint64_t obj_id) const
{
    return m_registry.try_get<ComponentType>(static_cast<PhysikaEntityType>(obj_id));
}

template <typename ComponentType>
ComponentType* World::getObjectComponent(std::uint64_t obj_id)
{
    return m_registry.try_get<ComponentType>(static_cast<PhysikaEntityType>(obj_id));
}

template <typename SolverType, typename... Args>
SolverType* World::createSolver(Args&&... args)
{
    SolverType* solver = new SolverType(std::forward<Args>(args)...);
    this->registerSolver(solver);
    return solver;
}

template <auto Candidate, typename... Type>
void World::connectSolverRemoveListener(Type&&... value_or_instance)
{
    m_rm_solver_sink.connect<Candidate>(value_or_instance...);
}

template <auto Candidate, typename... Type>
void World::disconnectSolverRemoveListener(Type&&... value_or_instance)
{
    m_rm_solver_sink.disconnect<Candidate>(value_or_instance...);
}

}  // namespace Physika