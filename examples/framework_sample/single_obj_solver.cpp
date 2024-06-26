/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: implementation of SingleObjSolver class, a sample solver that demonstrates
 *               how to define a solver that applies to single object
 * @version    : 1.0
 */

#include "single_obj_solver.hpp"

#include <iostream>
#include "framework/object.hpp"

namespace Physika {

SingleObjSolver::SingleObjSolver()
    : Solver(), m_is_init(false), m_object(nullptr), m_cur_time(0.0)
{
}

SingleObjSolver::~SingleObjSolver()
{
    this->reset();
}

bool SingleObjSolver::initialize()
{
    std::cout << "SingleObjSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool SingleObjSolver::isInitialized() const
{
    std::cout << "SingleObjSolver::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool SingleObjSolver::reset()
{
    std::cout << "SingleObjSolver::reset() sets the solver to newly initialized state.\n";
    m_is_init             = true;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_object              = nullptr;
    m_cur_time            = 0.0;
    return true;
}

bool SingleObjSolver::step()
{
    std::cout << "SingleObjSolver::step() updates the solver state by a time step.\n";
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_object)
    {
        std::cout << "    error: no object attached to the solver.\n";
        return false;
    }
    // Do the step stuff here.
    m_cur_time += m_config.m_dt;
    std::cout << "SingleObjSolver::step() applied to object " << m_object->id() << ".\n";
    return true;
}

bool SingleObjSolver::run()
{
    std::cout << "SingleObjSolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_object)
    {
        std::cout << "    error: no object attached to the solver.\n";
        return false;
    }
    // Update till termination
    int step_id = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        std::cout << "    step " << step_id << ": " << m_cur_time << " -> " << m_cur_time + dt << "\n";
        m_cur_time += dt;
        ++step_id;
    }
    std::cout << "SingleObjSolver::run() applied to object " << m_object->id() << ".\n";
    return true;
}

bool SingleObjSolver::isApplicable(const Object* object) const
{
    std::cout << "SingleObjSolver::isApplicable() checks if object has DummyComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<DummyComponent>();
}

bool SingleObjSolver::attachObject(Object* object)
{
    std::cout << "SingleObjSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }
    m_object = object;
    return true;
}

bool SingleObjSolver::detachObject(Object* object)
{
    std::cout << "SingleObjSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_object != object)
    {
        std::cout << "    error: object is not attached to the solver.\n";
        return false;
    }
    m_object = nullptr;
    return true;
}

void SingleObjSolver::clearAttachment()
{
    std::cout << "SingleObjSolver::clearAttachment() clears the target list of the solver.\n";
    m_object = nullptr;
}

void SingleObjSolver::config(const SingleObjSolver::SolverConfig& config)
{
    std::cout << "SingleObjSolver::config() setups the configuration of the solver.\n";
    m_config = config;
}

}  // namespace Physika