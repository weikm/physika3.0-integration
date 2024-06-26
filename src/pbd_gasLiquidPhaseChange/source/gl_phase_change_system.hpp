#pragma once
#include "thermal_pbf_solver.hpp"
namespace Physika {
class BoilingSystem
{
public:
    BoilingSystem(
        std::shared_ptr<ThermoParticles>& fluidModel,
        std::shared_ptr<ThermoParticles>& boundariesModel,
        std::shared_ptr<ThermoParticles>& heatenParticle,
        std::shared_ptr<ThermalPBFSolver>& solver,
        const float3 spaceSize,
        const float cellLength,
        const float smoothRadius,
        const float dt,
        const float mass,
        const float rho0,
        const float rhoBoundary,
        const float dynamicVisc,
        const float3 gravity,
        const float sphSurfaceTensionIntensity,
        const float sphAirPressure,
        const int3 cellSize);
    BoilingSystem(const BoilingSystem&) = delete;
    BoilingSystem& operator=(const BoilingSystem&) = delete;

    int fluidSize() const{
        return (*m_fluids).size();
    }
    int boundarySize() const{
        return (*m_boundaries).size();
    }
    int particlesSize() const{
        return (*m_fluids).getCapacity() + (*m_boundaries).getCapacity();
    }
    auto getFluids() const{
        return static_cast<const std::shared_ptr<ThermoParticles>>(m_fluids);
    }
    auto getBoundaries() const{
        return static_cast<const std::shared_ptr<ThermoParticles>>(m_boundaries);
    }
    auto getHeaten() const {
        return static_cast<const std::shared_ptr<ThermoParticles>>(m_heaten);
    }
    auto getColorGradient() const {
        return m_solver->getColorGradient();
    }
    // STEP 
    float solveStep();
    void  resetExternalForce();
    bool  setExternalForce(const std::vector<float3>& ex_acc_source);
    bool  setVelocity(const std::vector<float3>& ex_vel_source);

    ~BoilingSystem() noexcept{}
private:
    int steps = 0;
    // Obejects
    std::shared_ptr<ThermoParticles> m_fluids;
    std::shared_ptr<ThermoParticles> m_boundaries;
    std::shared_ptr<ThermoParticles> m_heaten;
    std::shared_ptr<ThermalPBFSolver> m_solver; // Dynamic Solver
    DataArray<uint32_t> cellStartFluid; // For neighbor searching
    DataArray<uint32_t> cellStartBoundary;
    DataArray<uint32_t> cellStartHeaten;
    
    // Simulation parameters
    float3 m_spaceSize;
    const float m_cellLength;
    const float m_smoothRadius;
    float m_dt;
    const float m_rho0;
    const float m_rhoBoundary;
    const float m_dynamicVisc;
    const float3 m_gravity;
    const float m_surfaceTensionIntensity;
    const float m_airPressure;
    const int3 m_cellSize;
    const float m_startTemp;

    const float m_latenHeat; // latent heat for phase-change
    DataArray<int> bufferInt; // swap buffer
 
    void boundaryMassSampling();

    void neighborSearch(const std::shared_ptr<ThermoParticles>& particles, DataArray<uint32_t>& cellStart);
};

}