/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-10-18
 * @description: implementation of DummyCudaSolver class (device code part), a sample solver that demonstrates
 *               the use of cuda in solver developement
 * @version    : 1.0
 */

#include "dummy_cuda_solver.hpp"

#include "cuda_kernels.cuh"

namespace Physika {

ThrustComponent::ThrustComponent()
{
}

ThrustComponent::~ThrustComponent()
{
}

ThrustComponent::ThrustComponent(const ThrustComponent& comp)
    : m_host_pos(comp.m_host_pos)
    , m_host_vel(comp.m_host_vel)
    , m_device_pos(comp.m_device_pos)
    , m_device_vel(comp.m_device_vel)
{
}

ThrustComponent& ThrustComponent::operator=(const ThrustComponent& comp)
{
    if (comp == *this)
        return *this;

    m_host_pos   = comp.m_host_pos;
    m_host_vel   = comp.m_host_vel;
    m_device_pos = comp.m_device_pos;
    m_device_vel = comp.m_device_vel;

    return *this;
}

bool ThrustComponent::operator==(const ThrustComponent& comp) const
{
    return m_host_pos == comp.m_host_pos
           && m_host_vel == comp.m_host_vel
           && m_device_pos == comp.m_device_pos
           && m_device_vel == comp.m_device_vel;
}

void ThrustComponent::reset()
{
    m_host_pos.assign(m_host_pos.size(), 0.0);
    m_host_vel.assign(m_host_vel.size(), 0.0);
    m_device_pos.assign(m_device_pos.size(), 0.0);
    m_device_vel.assign(m_device_vel.size(), 0.0);
}

void ThrustComponent::resize(size_t size, double value)
{
    m_host_pos.clear();
    m_host_vel.clear();
    m_device_pos.clear();
    m_device_vel.clear();
    m_host_pos.resize(size, value);
    m_host_vel.resize(size, value);
    m_device_pos.resize(size, value);
    m_device_vel.resize(size, value);
}

void ThrustComponent::copyHostToDevice()
{
    m_device_pos = m_host_pos;
    m_device_vel = m_host_vel;
}

void ThrustComponent::copyDeviceToHost()
{
    m_host_pos.resize(m_device_pos.size());
    m_host_vel.resize(m_device_vel.size());
    m_host_pos.assign(m_device_pos.begin(), m_device_pos.end());
    m_host_vel.assign(m_device_vel.begin(), m_device_vel.end());
}

bool ThrustComponent::checkDataSizeConsistency()
{
    if (m_host_pos.size() != m_host_vel.size() || m_host_pos.size() != m_device_pos.size() || m_host_pos.size() != m_device_vel.size())
        return false;

    return true;
}

void DummyCudaSolver::stepKernelWrapper(int blocks_per_grid, int threads_per_block, double dt, double gravity, ThrustComponent* comp)
{
    size_t data_num = comp->m_host_pos.size();
    stepKernel<<<blocks_per_grid, threads_per_block>>>(dt, gravity, data_num, thrust::raw_pointer_cast(comp->m_device_vel.data()), thrust::raw_pointer_cast(comp->m_device_pos.data()));
}

}  // namespace Physika