/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: implementation of DummySFISolver class, a sample solver that demonstrates
 *               how to define a solver that needs objects of different kind to run
 * @version    : 1.0
 */

#include "dummy_sfi_solver.hpp"

#include <iostream>
#include "framework/object.hpp"

namespace Physika {

DummySFISolver::DummySFISolver()
    : Solver(), m_is_init(false), m_fluid(nullptr), m_solid(nullptr), m_cur_time(0.0)
{
}

DummySFISolver::~DummySFISolver()
{
    this->reset();
}

bool DummySFISolver::initialize()
{
    std::cout << "DummySFISolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool DummySFISolver::isInitialized() const
{
    std::cout << "DummySFISolver::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool DummySFISolver::reset()
{
    std::cout << "DummySFISolver::reset() sets the solver to newly initialized state.\n";
    m_is_init             = true;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_fluid               = nullptr;
    m_solid               = nullptr;
    m_cur_time            = 0.0;
    return true;
}

bool DummySFISolver::step()
{
    std::cout << "DummySFISolver::step() updates the solver state by a time step.\n";
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_solid || !m_fluid)
    {
        std::cout << "    error: solid or fluid not attached to the solver.\n";
        return false;
    }
    // Do the step stuff here.
    m_cur_time += m_config.m_dt;
    std::cout << "DummySFISolver::step() applied to fluid " << m_fluid->id() << " and solid " << m_solid->id() << ".\n";
    return true;
}

bool DummySFISolver::run()
{
    std::cout << "DummySFISolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_solid || !m_fluid)
    {
        std::cout << "    error: solid or fluid not attached to the solver.\n";
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
    std::cout << "DummySFISolver::run() applied to fluid " << m_fluid->id() << " and solid " << m_solid->id() << ".\n";
    return true;
}

bool DummySFISolver::isApplicable(const Object* object) const
{
    std::cout << "DummySFISolver::isApplicable() checks if object has SolidComponent or FluidComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<SolidComponent>() || object->hasComponent<FluidComponent>();
}

bool DummySFISolver::attachObject(Object* object)
{
    std::cout << "DummySFISolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<SolidComponent>())
    {
        std::cout << "    object attached as solid.\n";
        m_solid = object;
    }
    if (object->hasComponent<FluidComponent>())
    {
        std::cout << "    object attached as fluid.\n";
        m_fluid = object;
    }

    return true;
}

bool DummySFISolver::detachObject(Object* object)
{
    std::cout << "DummySFISolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_fluid == object)
    {
        m_fluid = nullptr;
        std::cout << "    Fluid detached.\n";
        return true;
    }
    if (m_solid == object)
    {
        m_solid = nullptr;
        std::cout << "    Solid detached.\n";
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void DummySFISolver::clearAttachment()
{
    std::cout << "DummySFISolver::clearAttachment() clears the target list of the solver.\n";
    m_solid = nullptr;
    m_fluid = nullptr;
}

void DummySFISolver::config(const DummySFISolver::SolverConfig& config)
{
    std::cout << "DummySFISolver::config() setups the configuration of the solver.\n";
    m_config = config;
}

}  // namespace Physika