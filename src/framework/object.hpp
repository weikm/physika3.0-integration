/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-13
 * @description: declaration of Object class, the subject of solvers
 * @version    : 1.0
 */

#pragma once

#include <cstdint>

#include "world.hpp"

namespace Physika {

class Scene;

/**
 * Object is the building block of a scene, and is the target that the solvers apply to.
 * Object is composed of Components, which are data entries of the object.
 * Object cannot have 2 components with the same type.
 *
 * Requirement on the ComponentType: must have a reset() method to reset its data
 *
 * Object can be added to one Scene. An object must be removed from current scene before it
 * can be added to another scene.
 *
 * A solver only applies to an object that has specific component type(s) it needs.
 *
 */
class Object final
{
public:
    ~Object();
    /**
     * @brief    get the unique id of the object, which won't change once the object is created
     *
     * @return   the id of the object
     */
    std::uint64_t id() const;

    /**
     * @brief    reset the object to initial state, the components are reset as well (not removed)
     *           the object stay in the scene
     *
     * @return   true if object is successfully reset, otherwise return false
     */
    bool reset();

    /**
     * @brief create and initialize a component of given type, and add it to the object
     * @tparam ComponentType    type of the component
     * @tparam Args             variable argument list that constructor of ComponentType expects
     *
     * @param[in] args    arguments of ComponentType constructor
     *
     * @return    true if the component is successfully added
     *            false if error occurs, e.g., the component already exists
     */
    template <typename ComponentType, typename... Args>
    bool addComponent(Args&&... args);

    /**
     * @brief check whether the object has a component of specific type
     * @tparam ComponentType    type of the component
     *
     * @return    true if the object has the specific component, otherwise return false
     */
    template <typename ComponentType>
    bool hasComponent() const;

    /**
     * @brief remove specific component from the object
     * @tparam ComponentType    type of the component
     *
     * @return    true if the component is successfully removed
     *            false if error occurs, e.g., the object has no such component
     */
    template <typename ComponentType>
    bool removeComponent();

    /**
     * @brief remove all components of the object
     *
     * @return    true if all components are successfully removed
     *            false if error occurs
     *
     */
    bool removeAllComponents();

    /**
     * @brief get the total number of components
     *
     * @return    the component number
     */
    std::uint64_t componentNum() const;

    /**
     * @brief get the component with given type
     * @tparam ComponentType    the component type
     *
     * @return    pointer to the component, return nullptr if not found
     */
    template <typename ComponentType>
    const ComponentType* getComponent() const;

    template <typename ComponentType>
    ComponentType* getComponent();

    /**
     * @brief get the scene that the object is in
     *
     * @return    pointer to the host scene, return nullptr if the object is not in any scene
     */
    const Scene* getHostScene() const;
    Scene*       getHostScene();

private:
    Object();  // make constructor private to force creation through World API

    // disable copy&&move
    Object(const Object&)            = delete;
    Object(Object&&)                 = delete;
    Object& operator=(const Object&) = delete;
    Object& operator=(Object&&)      = delete;

    void setId(std::uint64_t id);
    void setHostScene(Scene* scene);

private:
    friend class World;
    friend class Scene;

    std::uint64_t m_id;             //!< unique id of the object
    std::uint64_t m_component_num;  //!< number of component the object has
    Scene*        m_host_scene;     //!< the scene that the object is in
};

//////////////////// implementation of template methods ////////////////////
template <typename ComponentType, typename... Args>
bool Object::addComponent(Args&&... args)
{
    return World::instance().addComponentToObject<ComponentType>(this->id(), std::forward<Args>(args)...);
}

template <typename ComponentType>
bool Object::hasComponent() const
{
    return World::instance().checkObjectHasComponent<ComponentType>(this->id());
}

template <typename ComponentType>
bool Object::removeComponent()
{
    return World::instance().removeComponentFromObject<ComponentType>(this->id());
}

template <typename ComponentType>
const ComponentType* Object::getComponent() const
{
    return World::instance().getObjectComponent<ComponentType>(this->id());
}

template <typename ComponentType>
ComponentType* Object::getComponent()
{
    return World::instance().getObjectComponent<ComponentType>(this->id());
}

}  // namespace Physika