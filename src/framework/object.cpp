/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-14
 * @description: implementation of Object class
 * @version    : 1.0
 */

#include "object.hpp"

#include <limits>

#include "scene.hpp"

namespace Physika {

Object::Object()
    : m_id(std::numeric_limits<std::uint64_t>::max())
    , m_component_num(0)
    , m_host_scene(nullptr)
{
    // id initialized to a nearly invalid value to minimize the chance of coninciding with world registry
}

Object::~Object()
{
}

std::uint64_t Object::id() const
{
    return m_id;
}

bool Object::reset()
{
    return World::instance().resetObject(this->id());
}

bool Object::removeAllComponents()
{
    auto ret        = World::instance().removeAllComponentsFromObject(this->id());
    m_component_num = 0;
    return ret;
}

std::uint64_t Object::componentNum() const
{
    return m_component_num;
}

const Scene* Object::getHostScene() const
{
    return m_host_scene;
}

Scene* Object::getHostScene()
{
    return m_host_scene;
}

void Object::setId(std::uint64_t id)
{
    m_id = id;
}

void Object::setHostScene(Scene* scene)
{
    m_host_scene = scene;
}

}  // namespace Physika