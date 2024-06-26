/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-10-18
 * @description: implementation of DummyCudaSolver class (host code part), a sample solver that demonstrates
 *               the use of cuda in solver developement
 * @version    : 1.0
 */

#include "dummy_cuda_solver.hpp"

#include <iostream>
#include "framework/object.hpp"

namespace Physika {

#ifndef BUILD_WITH_CUDA
ThrustComponent::ThrustComponent()
{
}

ThrustComponent::~ThrustComponent()
{
}

ThrustComponent::ThrustComponent(const ThrustComponent& comp)
    : m_host_pos(comp.m_host_pos)
    , m_host_vel(comp.m_host_vel)
{
}

ThrustComponent& ThrustComponent::operator=(const ThrustComponent& comp)
{
    if (comp == *this)
        return *this;

    m_host_pos = comp.m_host_pos;
    m_host_vel = comp.m_host_vel;

    return *this;
}

bool ThrustComponent::operator==(const ThrustComponent& comp) const
{
    return m_host_pos == comp.m_host_pos
           && m_host_vel == comp.m_host_vel;
}

void ThrustComponent::reset()
{
    m_host_pos.assign(m_host_pos.size(), 0.0);
    m_host_vel.assign(m_host_vel.size(), 0.0);
}

void ThrustComponent::resize(size_t size, double value)
{
    m_host_pos.clear();
    m_host_vel.clear();
    m_host_pos.resize(size, value);
    m_host_vel.resize(size, value);
}

bool ThrustComponent::checkDataSizeConsistency()
{
    if (m_host_pos.size() != m_host_vel.size())
        return false;

    return true;
}

#endif

DummyCudaSolver::DummyCudaSolver()
    : Solver(), m_is_init(false), m_object(nullptr), m_cur_time(0.0)
{
}

DummyCudaSolver::~DummyCudaSolver()
{
    this->reset();
}

bool DummyCudaSolver::initialize()
{
    std::cout << "DummyCudaSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool DummyCudaSolver::isInitialized() const
{
    std::cout << "DummyCudaSolver::isInitialized() gets the initialization status of the solver.\n";
    return m_is_init;
}

bool DummyCudaSolver::reset()
{
    std::cout << "DummyCudaSolver::reset() sets the solver to newly initialized state.\n";
    m_is_init             = true;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_object              = nullptr;
    m_cur_time            = 0.0;
    return true;
}

bool DummyCudaSolver::step()
{
    std::cout << "DummyCudaSolver::step() updates the solver state by a time step.\n";
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
    bool ret = (m_config.m_use_gpu ? stepGPU(m_config.m_dt) : stepCPU(m_config.m_dt));
    std::cout << "DummyCudaSolver::step() applied to object " << m_object->id() << ".\n";
    return ret;
}

bool DummyCudaSolver::run()
{
    std::cout << "DummyCudaSolver::run() updates the solver till termination criteria are met.\n";
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
    int  step_id = 0;
    bool ret     = true;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        std::cout << "    step " << step_id << ": " << m_cur_time << " -> " << m_cur_time + dt << "\n";
        m_cur_time += dt;
        ret = (m_config.m_use_gpu ? stepGPU(dt) : stepCPU(dt));
        ++step_id;
    }
    std::cout << "DummyCudaSolver::run() applied to object " << m_object->id() << ".\n";
    return ret;
}

bool DummyCudaSolver::isApplicable(const Object* object) const
{
    std::cout << "DummyCudaSolver::isApplicable() checks if object has ThrustComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<ThrustComponent>();
}

bool DummyCudaSolver::attachObject(Object* object)
{
    std::cout << "DummyCudaSolver::attachObject() set the target of the solver.\n";
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

bool DummyCudaSolver::detachObject(Object* object)
{
    std::cout << "DummyCudaSolver::detachObject() remove the object from target list of the solver.\n";
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

void DummyCudaSolver::clearAttachment()
{
    std::cout << "DummyCudaSolver::clearAttachment() clears the target list of the solver.\n";
    m_object = nullptr;
}

void DummyCudaSolver::config(const DummyCudaSolver::SolverConfig& config)
{
    std::cout << "DummyCudaSolver::config() setups the configuration of the solver.\n";
    m_config = config;
}

bool DummyCudaSolver::stepCPU(double dt)
{
    auto* comp = m_object->getComponent<ThrustComponent>();
    if (!comp->checkDataSizeConsistency())
        return false;

    size_t data_num = comp->m_host_pos.size();
    for (size_t i = 0; i < data_num; ++i)
    {
        comp->m_host_vel[i] += m_config.m_gravity * dt;
        comp->m_host_pos[i] += comp->m_host_vel[i] * dt;
    }
#ifdef BUILD_WITH_CUDA
    comp->copyHostToDevice();
#endif
    return true;
}

bool DummyCudaSolver::stepGPU(double dt)
{

#ifndef BUILD_WITH_CUDA
    std::cout << "DummyCudaSolver::stepGPU() is only available when BUILD_WITH_CUDA is defined.\n";
    return false;
#else
    auto* comp = m_object->getComponent<ThrustComponent>();
    if (!comp->checkDataSizeConsistency())
        return false;

    size_t data_num        = comp->m_host_pos.size();
    int    threadsPerBlock = 256;
    int    blocksPerGrid   = (data_num + threadsPerBlock - 1) / threadsPerBlock;
    stepKernelWrapper(blocksPerGrid, threadsPerBlock, dt, m_config.m_gravity, comp);
    comp->copyDeviceToHost();
    return true;
#endif
}

}  // namespace Physika