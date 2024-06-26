/**
 * @file gas_liquid_phase_change_solver.cpp
 *
 * @author Wang Qianwei (729596003@qq.com)
 * @date 2023-11-17
 * @brief This file implements a solver for simulating gas-liquid phase change processes, designed to model
 *        the transition between gas and liquid states.
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The gas-liquid phase change solver is a computational tool used for predicting the behavior of materials
 * undergoing phase transitions between gas and liquid phases. It takes the initial properties and conditions
 * of the material as input and calculates the evolution of its phase state over time, providing insights into
 * the material's phase change dynamics.
 *
 * This solver is an integral part of the Physika framework and is designed to seamlessly integrate with other
 * components within the framework, facilitating comprehensive material modeling and simulation capabilities.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, and vector_functions.
 * It also includes "gas_liquid_phase_change_solver.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To utilize this solver, it requires the specification of initial material properties and environmental conditions,
 * which are encapsulated within the GasLiquidComponent. The solver then computes the evolution of the material's
 * phase state, providing access to the resulting properties for further analysis or visualization.
 *
 *
 * @section WARNING
 *
 * It is essential to ensure that the CUDA environment is correctly configured and operational, as the solver relies on
 * CUDA for efficient computation of phase change processes.
 *
 */

#include "gas_liquid_phase_change_solver.hpp"

#include <iostream>
#include <string>
#include <fstream>

#include "framework/object.hpp"
#include "pbd_gasLiquidPhaseChange/source/Util/helper_math.h"

namespace Physika {
void GasLiquidPhaseChangeComponent::reset()
{
    m_bInitialized = false;
    m_fluid_initialized = false;
    m_space_initialized = false;
    m_sim_system   = nullptr;
}

GasLiquidPhaseChangeComponent::GasLiquidPhaseChangeComponent()
    : m_sim_system(nullptr), m_bInitialized(false), m_fluid_initialized(false), m_space_initialized(false)
{
    m_config                                         = new GasLiquidPhaseChangeParams();
}

void GasLiquidPhaseChangeComponent::initialize()
{
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    if (!(m_fluid_initialized && m_space_initialized))
    {
        if (!m_fluid_initialized)
            std::cout << "Fluid object not initialized.\n";
        if (!m_space_initialized)
            std::cout << "World boundary not initialized.\n";
        return;
    }
    auto space_size = m_config->space_size;

    m_fluid_capacity = m_host_pos_fluid.size() * 2;
    m_num_particles  = m_fluid_capacity;

    float sph_cell_size = m_config->sph_cell_size;

    const auto compact_size = 2 * make_int3(ceil(space_size.x / sph_cell_size), ceil(space_size.y / sph_cell_size), ceil(space_size.z / sph_cell_size));
    // front and back
    for (auto i = 0; i < compact_size.x; ++i)
    {
        for (auto j = 0; j < compact_size.y; ++j)
        {
            auto x = make_float3(i, j, 0) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
            x = make_float3(i, j, compact_size.z - 1) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
        }
    }
    // top and bottom
    for (auto i = 0; i < compact_size.x; ++i)
    {
        for (auto j = 0; j < compact_size.z - 2; ++j)
        {
            auto x = make_float3(i, 0, j + 1) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
            m_host_pos_heaten.push_back(0.99f * x + 0.005f * space_size);
            m_host_tem_heaten.push_back(160.0f);
            if (j % 16 || i % 16 || j == 0 || i == 0)
                m_host_nucleation_marks.push_back(1);
            else
                m_host_nucleation_marks.push_back(0);
            x = make_float3(i, compact_size.y - 1, j + 1) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
            m_host_pos_heaten.push_back(0.99f * x + 0.005f * space_size);
            m_host_tem_heaten.push_back(20.0f);
            m_host_nucleation_marks.push_back(1);
        }
    }
    // left and right
    for (auto i = 0; i < compact_size.y - 2; ++i)
    {
        for (auto j = 0; j < compact_size.z - 2; ++j)
        {
            auto x = make_float3(0, i + 1, j + 1) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
            x = make_float3(compact_size.x - 1, i + 1, j + 1) / make_float3(compact_size - make_int3(1)) * space_size;
            m_host_pos_boundary.push_back(0.99f * x + 0.005f * space_size);
        }
    }

    printf("Copy data to device.\n");
    auto fluid_particles = std::make_shared<ThermoParticles>(m_host_pos_fluid, m_host_vel_fluid, m_host_tem_fluid, m_num_particles);

    auto heaten_particles   = std::make_shared<ThermoParticles>(m_host_pos_heaten, m_host_tem_heaten, m_host_nucleation_marks);

    auto boundary_particles = std::make_shared<ThermoParticles>(m_host_pos_boundary);

    printf("Init Solver.\n");
    std::shared_ptr<ThermalPBFSolver> p_solver;
    p_solver       = std::make_shared<ThermalPBFSolver>(fluid_particles->getCapacity(), 3);
    printf("Init System. \n");

    m_sim_system        = std::make_shared<BoilingSystem>
        (fluid_particles, boundary_particles, heaten_particles, p_solver, 
            m_config->space_size, m_config->sph_cell_size, m_config->sph_smoothing_radius, m_config->dt, 
            m_config->rest_mass, m_config->rest_rho, m_config->rest_rho_boundary, m_config->sph_visc, 
            m_config->sph_g, m_config->sph_surface_tension_intensity, m_config->sph_air_pressure, m_config->cell_size);    
    
    // Init velocity
    //m_sim_system->setVelocity(m_host_vel_fluid);
    
    m_bInitialized = true;
    printf("Init system end.\n");
}

void GasLiquidPhaseChangeComponent::freeMemory()
{
    m_sim_system.reset();
    m_bInitialized = false;
    m_space_initialized = false;
    m_fluid_initialized = false;
}
// Set parameters and fluid particles data
/**
* Set parameters and fluid particles data by these methods.
*/

bool GasLiquidPhaseChangeComponent::initializeParticleRadius(float radius) {
    if (m_bInitialized)
    {
        std::cout << "Failed to initiailize particle radius: Component already initialized.\n";
        return false;
    }
    m_config->setParticleSize(radius);
    return true;
}

bool GasLiquidPhaseChangeComponent::initializeWorldBoundary(float3 space_size) {
    if (!m_space_initialized)
    {
        m_host_pos_boundary.clear();
        m_host_pos_heaten.clear();
        m_host_tem_heaten.clear();
        m_host_nucleation_marks.clear();

        m_config->setSpaceSize(space_size);
        m_space_initialized  = true;

        return true;
    }
    std::cout << "Already initialized: World boudary.\n";
    return false;
}

bool GasLiquidPhaseChangeComponent::initializePartilcePosition(const std::vector<float3>& pos) {
    if (!m_fluid_initialized)
    {
        m_host_pos_fluid.clear();
        m_host_tem_fluid.clear();
        m_host_pos_fluid.assign(pos.begin(), pos.end());
        m_host_tem_fluid.assign(m_host_pos_fluid.size(), m_config->m_fluid_tem);
        m_host_vel_fluid.assign(m_host_pos_fluid.size(), make_float3(0.0f));
        m_fluid_initialized  = true;
        return true;
    }
    std::cout << "Already initialized: Fluid position.\n";
    return false;
}

bool GasLiquidPhaseChangeComponent::initializeLiquidInstance(float3 lb, float3 cube_size, float sampling_size)
{
    if (!m_fluid_initialized)
    {
        m_host_pos_fluid.clear();
        m_host_tem_fluid.clear();
        std::vector<float3> pos;
        int num_x = cube_size.x / sampling_size;
        int num_y = cube_size.y / sampling_size;
        int num_z = cube_size.z / sampling_size;
        for (auto i = 0; i < num_y; ++i)
        {
            for (auto j = 0; j < num_x; ++j)
            {
                for (auto k = 0; k < num_z; ++k)
                {
                    auto x = make_float3(lb.x + sampling_size * j,
                                         lb.y + sampling_size * i,
                                         lb.z + sampling_size * k);
                    pos.push_back(x);
                }
            }
        }
        m_host_pos_fluid.assign(pos.begin(), pos.end());
        m_host_tem_fluid.assign(m_host_pos_fluid.size(), m_config->m_fluid_tem);
        m_host_vel_fluid.assign(m_host_pos_fluid.size(), make_float3(0.0f));
        m_fluid_initialized = true;
        return true;
    }
    std::cout << "Already initialized: Fluid position.\n";
    return false;
}

bool GasLiquidPhaseChangeComponent::initializePartilceVelocity(const std::vector<float3>& init_vel)
{
    if (!(m_fluid_initialized && m_space_initialized))
    {
        if (!m_fluid_initialized)
            std::cout << "Failed to set velocity: Fluid position not initialized.\n";
        if (!m_space_initialized)
            std::cout << "Failed to set velocity: World boundary not initialized.\n";
        return false;
    }
    if (!m_bInitialized)
    {
        if (m_host_vel_fluid.size() != init_vel.size())
        {
            std::cout << "Failed to set velocity: Particle size mismatching.\n";
            return false;
        }
        m_host_vel_fluid.assign(init_vel.begin(), init_vel.end());
        return true;
    }
    std::cout << "Failed to set velocity: Already initialized: Fluid object.\n";
    return false;
}

float GasLiquidPhaseChangeComponent::getBoilingPoint() {
    return m_config->m_boilingPoint;
}

float GasLiquidPhaseChangeComponent::getLatentHeat()
{
    return m_config->m_latenHeat;
}

    /*
* Get data ptr from device(GPU)
*/
float* GasLiquidPhaseChangeComponent::getFluidParticlePositionPtr() {
    if (m_bInitialized)
    {
        return ( float* )m_sim_system->getFluids()->getPosPtr();
    }
    return nullptr;
}

float* GasLiquidPhaseChangeComponent::getFluidParticleTemperaturePtr() {
    if (m_bInitialized)
    {
        return m_sim_system->getFluids()->getTempPtr();
    }
    return nullptr;
}

int* GasLiquidPhaseChangeComponent::getFluidParticleTypePtr() {
    if (m_bInitialized)
    {
        return m_sim_system->getFluids()->getTypePtr();
    }
    return nullptr;
}

float* GasLiquidPhaseChangeComponent::getFluidParticleVelocityPtr() {
    if (m_bInitialized)
    {
        return ( float* )m_sim_system->getFluids()->getVelPtr();
    }
    return nullptr;
}

float* GasLiquidPhaseChangeComponent::getFluidExternalForcePtr()
{
    if (m_bInitialized)
    {
        return ( float* )m_sim_system->getFluids()->getExternalForcePtr();
    }
    return nullptr;
}

int GasLiquidPhaseChangeComponent::getFluidParticleNum() {
    if (m_bInitialized)
    {
        return m_sim_system->getFluids()->size();
    }
    return 0;
}

GasLiquidPhaseChangeComponent::~GasLiquidPhaseChangeComponent()
{
    // free gpu memory
    m_sim_system = nullptr;
    delete m_config;
    if (m_bInitialized)
    {
        m_bInitialized = false;
    }
}
/*
* GasLiquidPhaseChangeSolver
*/

GasLiquidPhaseChangeSolver::GasLiquidPhaseChangeSolver()
    : m_is_init(false), m_gas_liquid_phase_change(nullptr), m_cur_time(0)
{
}

GasLiquidPhaseChangeSolver::~GasLiquidPhaseChangeSolver()
{
}

bool GasLiquidPhaseChangeSolver::initialize()
{
    if (m_is_init)
        return true;
    if (m_gas_liquid_phase_change == nullptr)
    {
        std::cout << "ERROR: Must set gas fluid phase change particle object first.\n";
        return false;
    }
    if (m_gas_liquid_phase_change->hasComponent<GasLiquidPhaseChangeComponent>() == false)
    {
        std::cout << "ERROR: gas_liquid_phase_change object has no GasLiquidPhaseChange component.\n";
        return false;
    }
    GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();
    if (gas_liquid_phase_change_component == nullptr)
    {
        std::cout << "ERROR: no GasLiquidPhaseChange component.\n";
        return false;
    }

    m_is_init = true;
    std::cout << "GasLiquidPhaseChange solver initialized successfully.\n";
    return true;
}

bool GasLiquidPhaseChangeSolver::isInitialized() const
{
    return m_is_init;
}

bool GasLiquidPhaseChangeSolver::reset()
{
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_gas_liquid_phase_change   = nullptr;
    m_cur_time            = 0.0;
    if (m_gas_liquid_phase_change != nullptr)
    {
        m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>()->reset();
    }
    return true;
}


bool GasLiquidPhaseChangeSolver::step()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (!m_gas_liquid_phase_change->hasComponent<GasLiquidPhaseChangeComponent>())
        return false;
    GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();

    if (gas_liquid_phase_change_component == nullptr)
        return false;
    auto milliseconds = gas_liquid_phase_change_component->m_sim_system->solveStep();
    m_config.m_total_time += milliseconds;

    //setParameters(m_params);

    return true;
}

bool GasLiquidPhaseChangeSolver::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_gas_liquid_phase_change)
    {
        return false;
    }
    // Update till termination

    float sim_run_time = 0.0;
    int   step_id      = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        //***time count start
        cudaEvent_t start, stop;
        float       elapsedTime;
        // create CUDA event
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // record start time
        cudaEventRecord(start, 0);
        step();
        // record end time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // count running time
        cudaEventElapsedTime(&elapsedTime, start, stop);

        // destroy CUDA event
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        sim_run_time += elapsedTime;        
        //***time count end

        ++step_id;
    }
    std::cout << "Run Time: " << sim_run_time << " ms" << std::endl;
    return true;
}

bool GasLiquidPhaseChangeSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<GasLiquidPhaseChangeComponent>();
}

bool GasLiquidPhaseChangeSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<GasLiquidPhaseChangeComponent>())
    {
        m_gas_liquid_phase_change                   = object;
        GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();
        if (!gas_liquid_phase_change_component->m_bInitialized)
        {
            std::cout << "GasLiquidPhaseChangeComponent not initialized.\n";
            return false;
        }
        std::cout << "object attached as gas liquid phase change system.\n";
    }
    initialize();
    return true;
}

bool GasLiquidPhaseChangeSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_gas_liquid_phase_change == object)
    {
        m_gas_liquid_phase_change = nullptr;
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void GasLiquidPhaseChangeSolver::clearAttachment()
{
    m_gas_liquid_phase_change = nullptr;
}

void GasLiquidPhaseChangeSolver::config(SolverConfig& config)
{

}

float* GasLiquidPhaseChangeSolver::getParticlePositionPtr()
{
    if (m_gas_liquid_phase_change != nullptr)
    {
        if (m_gas_liquid_phase_change->hasComponent<GasLiquidPhaseChangeComponent>())
        {
            GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();
            if (gas_liquid_phase_change_component != nullptr)
                return (float*)(gas_liquid_phase_change_component->m_sim_system->getFluids()->getPosPtr());
        }
    }   
    return nullptr;
}

void GasLiquidPhaseChangeSolver::getParticleRadius(float& particleRadius)
{
    if (!m_is_init)
        particleRadius = 0;
    if (m_gas_liquid_phase_change != nullptr)
    {
        if (m_gas_liquid_phase_change->hasComponent<GasLiquidPhaseChangeComponent>())
        {
            GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();
            if (gas_liquid_phase_change_component != nullptr)
                particleRadius = gas_liquid_phase_change_component->m_config->sph_radius;
        }
    }  
}
bool GasLiquidPhaseChangeSolver::setPartilceExternalForce(std::vector<float3>& ex_force)
{
    if (!m_is_init)
    {
        std::cout << "Failed to set external force: Solver not initialized.\n";
        return false;
    }
    if (m_gas_liquid_phase_change != nullptr)
    {
        if (m_gas_liquid_phase_change->hasComponent<GasLiquidPhaseChangeComponent>())
        {
            GasLiquidPhaseChangeComponent* gas_liquid_phase_change_component = m_gas_liquid_phase_change->getComponent<GasLiquidPhaseChangeComponent>();
            if (gas_liquid_phase_change_component != nullptr)
            {
                if (!gas_liquid_phase_change_component->m_bInitialized)
                {
                    std::cout << "Failed to set external force: Component not initialized.\n";
                    return false;
                }
                else
                {
                    if(gas_liquid_phase_change_component->m_sim_system->setExternalForce(ex_force))
                        return true;
                }
            }
                
        }
    }  
    std::cout << "Failed to set external force: Memory copying error.\n";
    return false;
        
}
}  // namespace Physika