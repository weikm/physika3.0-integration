/**
 * @file granular_particle_solver.cpp
 *
 * @author Yuanmu Xu (xyuan1517@gmail.com)
 * @date 2023-07-17
 * @brief This file implements a granular particle solver, primarily intended for dry sand particles.
 * @version 1.0
 * 
 * @author Haikai Zeng (haok_z@126.com)
 * @date 2023-11-18
 * @brief This file implements a granular particle solver, primarily intended for dry sand particles, and multi-scale particle coupling is achieved.
 *
 * @section DESCRIPTION
 *
 * The granular particle solver is a numerical tool for predicting the behavior of granular systems,
 * specifically dry sand particles, and enables the coupling of different fine-grained particles.
 * It operates by taking the initial positions and invMass of the particles as input,
 * and outputs their positions and invMass at the subsequent time step.
 *
 * This solver is part of the Physika framework and is built to work with other components within the framework.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, chrono, iomanip, filesystem and vector_functions.
 * It also includes "granular_particle_solver.hpp", "granular_params.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To use this solver, you need to provide the initial positions and invMass of the particles(By defining the GranularComponent).
 * The solver then calculates the positions of the particles at the next time step, which can be retrieved for further use or visualization.
 *
 * The solver can handle errors during computation. Specifically, it includes a method 'getLastCudaErrorGranular' to
 * retrieve the last CUDA error message.
 *
 * @section WARNING
 *
 * Ensure that the CUDA environment is properly set up, as the solver relies on it for computation.
 *
 */

#include "granular_particle_solver_integration.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <vector_functions.h>

#include "granular_params.hpp"
//#include "framework/object.hpp"

namespace Physika {
/**
 * @brief  : get last cuda error
 * @param[in]  : errorMessage
 */
extern void getLastCudaErrorGranular(const char* errorMessage);

/**
 * @brief  : set simulation parameters (copy the params from CPU to GPU)
 *  This function will be called before each simulation step to ensure that
 *  the modified simulation parameters can take effect immediately
 *
 * @param[in]  : new simulation parameters
 */
extern void setParameters(GranularSimulateParams* hostParams);

/**
 * @brief  : compute hash value for each particle
 *
 * @param[in]  : gridParticleHash  the grid hash pointer
 * @param[in]  : pos               pointer of the particle position array
 * @param[in]  : numParticles
 */
extern void computeHashGranular(
    unsigned int* gridParticleHash,
    float*        pos,
    int           numParticles);

/**
 * @brief  : sort the particles based on the hash value
 *
 * @param[in]  : deviceGridParticleHash  the grid hash pointer
 * @param[in]  : numParticles
 * @param[in/out]  : devicePos               pointer of the particle position array
 * @param[in/out]  : deviceVel               pointer of the particle velocity array
 * @param[in/out]  : devicePredictedPos      pointer of the particle predicted position array
 * @param[in.out]  : phase                   pointer of the particle phase array
 */
extern void sortParticlesGranular(
    unsigned int* deviceGridParticleHash,
    unsigned int  numParticles,
    float*        devicePos,
    float*        deviceVel,
    float*        devicePredictedPos,
    float*        phase);

/**
 * @brief  : reorder the particle data based on the sorted hash value
 *
 * @param[in]  : cellStart  		     pointer of the cell start array
 * @param[in]  : cellEnd    		     pointer of the cell end array
 * @param[in]  : gridParticleHash   	 pointer of the hash array
 * @param[in]  : numParticles         number of particles
 * @param[in]  : numCell              number of grid cells
 */
extern void findCellRangeGranular(
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCell);

/**
 * @brief : advect particles
 *
 * @param[in] : position 	  pointer of the particle position array
 * @param[in] : velocity 	  pointer of the particle velocity array
 * @param[in] : predictedPos pointer of the particle predicted position array
 * @param[in] : phase        pointer of the particle phase array
 * @param[in] : deltaTime    time step
 * @param[in] : numParticles number of particles
 */
extern void granularAdvection(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    float*       phase,
    float*       height,
    float        unit_height,
    int          height_x_num,
    int          height_z_num,
    float        deltaTime,
    unsigned int numParticles);

/**
 * @brief : update particle velocity and position
 *
 * @param[in] : position 	  pointer of the particle position array
 * @param[in] : velocity 	  pointer of the particle velocity array
 * @param[in] : phase        pointer of the particle phase array
 * @param[in] : deltaTime    time step
 * @param[in] : numParticles number of particles
 * @param[in/out] : predictedPos pointer of the particle predicted position array
 */
extern void updateVelAndPosGranular(
    float4*      position,
    float4*      velocity,
    float*       phase,
    float        deltaTime,
    unsigned int numParticles,
    float4*      predictedPos);
/**
 * @brief : solver the constrains for each particles
 *
 * @param[in/out] : position 	  pointer of the particle position array
 * @param[in/out] : velocity 	  pointer of the particle velocity array
 * @param[in] : predictedPos pointer of the particle predicted position array
 * @param[in] : phase        pointer of the particle phase array
 * @param[in] : cellStart
 * @param[in] : cellEnd
 * @param[in] : gridParticleHash
 * @param[in] : numParticles number of particles
 * @param[in] : numCells     number of grid cells
 */
extern void solveDistanceConstrainGranluar(
    float4*       postion,
    float4*       velocity,
    float3*       deltaPos,
    float4*       predictedPos,
    float*        phase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

/**
 * @brief : solver the collision constrains for particles which are in contact with the collision object
 *
 * @param[in/out] : position 	  pointer of the particle position array
 * @param[in/out] : predictedPos pointer of the particle predicted position array
 * @param[in]   : phase        pointer of the particle phase array
 * @param[in]   : collision_particle_id article id which are in contact with the collision object
 * @param[in]   : numCollisionParticles the number of particles which are in contact with the collision object
 */
extern void solverColisionConstrainGranular(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles);

#define M_PI 3.1415926535
/**
 * @brief  : a simple vector3(int) struct
 */
struct MyIntVec3
{
    int x, y, z;
};
/**
 * @brief  : a simple vector3(float) struct
 */
struct MyFloatVec3
{
    float x, y, z;
};

//std::chrono::time_point<std::chrono::system_clock> lastExecutionTime;

void GranularParticleComponentIntegration::reset()
{
    freeMemory();
    m_host_pos.resize(0);
    m_host_vel.resize(0);
    m_host_phase.resize(0);
}

void GranularParticleComponentIntegration::addInstance(float particle_radius, float3 lb, float3 size, float3 vel_start)
{
    float spacing = particle_radius * 2.0f;
    float jitter  = particle_radius * 0.01f;
    float invmass = 1.0f / pow(particle_radius / 0.3, 3);
    MyIntVec3 bottomFluidDim = { static_cast<int>(size.x / spacing),
                                 static_cast<int>(size.y / spacing),
                                 static_cast<int>(size.z / spacing) };

    for (int z = 0; z < bottomFluidDim.z; ++z)
    {
        for (int y = 0; y < bottomFluidDim.y; ++y)
        {
            for (int x = 0; x < bottomFluidDim.x; ++x)
            {
                m_host_pos.push_back(spacing * x + particle_radius + lb.x + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * y + particle_radius + lb.y + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * z + particle_radius + lb.z + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(invmass);

                m_host_phase.push_back(static_cast<float>(GranularPhase::GRANULAR));

                m_host_vel.push_back(vel_start.x);
                m_host_vel.push_back(vel_start.y);
                m_host_vel.push_back(vel_start.z);
                m_host_vel.push_back(0.0f);
            }
        }
    }
    m_num_particles = static_cast<unsigned int>(m_host_pos.size() / 4);
}

void GranularParticleComponentIntegration::addInstance(float particle_radius, std::vector<float3>& init_pos)
{
    float invmass   = 1.0f / pow(particle_radius / 0.3, 3);
    int num_particles = static_cast<unsigned int>(init_pos.size());
    

    for (unsigned int i = 0; i < num_particles; i++)
    {
        m_host_pos.push_back(init_pos[i].x);
        m_host_pos.push_back(init_pos[i].y);
        m_host_pos.push_back(init_pos[i].z);
        m_host_pos.push_back(invmass);

        m_host_phase.push_back(static_cast<float>(GranularPhase::GRANULAR));

        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
    }
    m_num_particles = static_cast<unsigned int>(m_host_pos.size() / 4);

}

void GranularParticleComponentIntegration::initialize()
{
    freeMemory();
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    unsigned int memSize = sizeof(float) * 4 * m_num_particles;

    // allocation
    cudaMalloc(( void** )&m_device_pos, memSize);
    getLastCudaErrorGranular("allocation1: pos");
    cudaMalloc(( void** )&m_device_vel, memSize);
    cudaMalloc(( void** )&m_device_phase, sizeof(float) * m_num_particles);
    getLastCudaErrorGranular("allocation");

    cudaMalloc(( void** )&m_collision_particle_id, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_move_direction, sizeof(float) * 3 * m_num_particles);
    cudaMalloc(( void** )&m_move_distance, sizeof(float) * m_num_particles);
    cudaMalloc(( void** )&m_collision_force, sizeof(float) * 3 * m_num_particles);


    m_bInitialized = true;

    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(float) * m_num_particles, cudaMemcpyHostToDevice);
    getLastCudaErrorGranular("Memcpy");


    //test
    auto a0 = m_host_pos;
    auto b0 = m_device_pos;



    int ccc = 0;

    //cudaDeviceSynchronize();
}
void GranularParticleComponentIntegration::freeMemory()
{
    cudaFree(m_device_pos);
    getLastCudaErrorGranular("m_device_pos");
    cudaFree(m_device_phase);
    getLastCudaErrorGranular("m_device_phase");
    cudaFree(m_device_vel);
    getLastCudaErrorGranular("m_device_vel");
    cudaFree(m_collision_particle_id);
    cudaFree(m_move_direction);
    cudaFree(m_move_distance);
    cudaFree(m_collision_force);
    getLastCudaErrorGranular("m_collision_force");
}

GranularParticleSolverIntegration::GranularParticleSolverIntegration()
    : m_is_init(false), m_granular_particle_component(nullptr), m_cur_time(0), m_host_pos(nullptr), m_collision_particle_id(nullptr), m_device_cell_end(nullptr), m_device_cell_start(nullptr), m_device_delta_pos(nullptr), m_device_grid_particle_hash(nullptr), m_device_predicted_pos(nullptr)
{
    m_params = new GranularSimulateParams();
    // particles and grid.
    uint3 gridSize = make_uint3(128, 128, 128);
    float radius   = 0.3f;

    m_num_grid_cells            = gridSize.x * gridSize.y * gridSize.z;
    m_params->m_num_grid_cells  = m_num_grid_cells;
    m_params->m_grid_size       = gridSize;
    m_params->m_particle_radius = radius;
    m_params->m_num_particles   = 0;
    m_params->m_num_grid_cells  = gridSize.x * gridSize.y * gridSize.z;

    // iteration number.
    m_params->m_max_iter_nums = 3;

    // smooth kernel radius.
    m_params->m_sph_radius         = static_cast<float>(4.0 * radius);
    m_params->m_sph_radius_squared = m_params->m_sph_radius * m_params->m_sph_radius;

    // lagrange multiplier eps.
    m_params->m_lambda_eps = 1000.0f;

    // world boundary
    m_params->m_world_box_corner1 = make_float3(-30, -20, -15);
    m_params->m_world_box_corner2 = make_float3(30, 20, 15);

    // reset density.
    m_params->m_rest_density     = static_cast<float>(1.0f / (8.0f * powf(radius, 3.0f)));
    m_params->m_inv_rest_density = static_cast<float>(1.0f / m_params->m_rest_density);

    // sph kernel function coff.
    m_params->m_poly6_coff      = static_cast<float>(315.0f / (64.0f * M_PI * powf(m_params->m_sph_radius, 9.0)));
    m_params->m_spiky_grad_coff = static_cast<float>(-45.0f / (M_PI * powf(m_params->m_sph_radius, 6.0)));
    m_params->m_one_div_wPoly6  = static_cast<float>(1.0f / (m_params->m_poly6_coff * pow(m_params->m_sph_radius_squared - pow(0.1 * m_params->m_sph_radius, 2.0), 3.0)));

    // friction coff.
    m_params->m_stiffness               = 0.5f;
    m_params->m_static_fricton_coeff    = 0.8f;
    m_params->m_dynamic_fricton_coeff   = 0.5f;
    m_params->m_stack_height_coeff      = -2.f;
    m_params->m_static_frict_threshold  = static_cast<float>(m_params->m_static_fricton_coeff * m_params->m_particle_radius * 2.f);
    m_params->m_dynamic_frict_threshold = static_cast<float>(m_params->m_dynamic_fricton_coeff * m_params->m_particle_radius * 2.f);
    m_params->m_sleep_threshold         = 0.02f;
    // grid cells.
    float cellSize           = m_params->m_sph_radius;
    m_params->m_cell_size    = make_float3(cellSize, cellSize, cellSize);
    m_params->m_gravity      = make_float3(0.0f, -9.8f, 0.0f);
    m_params->m_world_origin = { -42.0f, -22.0f, -22.0f };
    // height field.
    m_unit_height  = 1.0f;
    m_height_x_num = static_cast<int>((m_params->m_world_box_corner2.x - m_params->m_world_box_corner1.x) / m_unit_height);
    m_height_z_num = static_cast<int>((m_params->m_world_box_corner2.z - m_params->m_world_box_corner1.z) / m_unit_height);
    m_host_height.insert(m_host_height.end(), m_height_x_num * m_height_z_num, m_params->m_world_box_corner1.y);
}

GranularParticleSolverIntegration::~GranularParticleSolverIntegration()
{
    if (m_is_init)
        _finalize();
}

bool GranularParticleSolverIntegration::initialize()
{
    if (m_is_init)
        return true;
    if (m_granular_particle_component == nullptr)
    {
        std::cout << "ERROR: Must set granular particle object first.\n";
        return false;
    }
    if (m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/ == false)
    {
        std::cout << "ERROR: granular particle object has no granular component.\n";
        return false;
    }
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
    {
        std::cout << "ERROR: no granular component.\n";
        return false;
    }
    mallocParticleMemory(m_params->m_num_particles);

    m_is_init = true;
    std::cout << "granular solver initialized successfully.\n";
    return true;
}

bool GranularParticleSolverIntegration::isInitialized() const
{
    return m_is_init;
}

bool GranularParticleSolverIntegration::reset()
{
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_granular_particle_component   = nullptr;
    m_cur_time            = 0.0;
    if (m_granular_particle_component != nullptr)
    {
        m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/->reset();
    }
    return true;
}

bool GranularParticleSolverIntegration::step()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return false;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;

    if (granular_component == nullptr)
        return false;

    setParameters(m_params);

    // advect
    {
        granularAdvection(
            ( float4* )granular_component->m_device_pos,
            ( float4* )granular_component->m_device_vel,
            ( float4* )m_device_predicted_pos,
            ( float3* )granular_component->m_collision_force,
            granular_component->m_device_phase,
            m_device_height,
            m_unit_height,
            m_height_x_num,
            m_height_z_num,
            static_cast<float>(m_config.m_dt),
            granular_component->m_num_particles);

        getLastCudaErrorGranular("Advection");
    }
    // collision handling
    {
        if (granular_component->m_num_collisions != 0)
            solverColisionConstrainGranular(
                ( float4* )granular_component->m_device_pos,
                ( float4* )m_device_predicted_pos,
                ( float3* )granular_component->m_move_direction,
                granular_component->m_move_distance,
                granular_component->m_device_phase,
                granular_component->m_collision_particle_id,
                granular_component->m_num_collisions);
    }

    // find neighbours
    {
        // calculate grid Hash.
        computeHashGranular(
            m_device_grid_particle_hash,
            m_device_predicted_pos,
            granular_component->m_num_particles);
        getLastCudaErrorGranular("computeHashGranular00");

        // sort particles based on hash value.
        sortParticlesGranular(
            m_device_grid_particle_hash,
            granular_component->m_num_particles,
            granular_component->m_device_pos,
            granular_component->m_device_vel,
            m_device_predicted_pos,
            granular_component->m_device_phase);
        getLastCudaErrorGranular("sortParticlesGranular");

        // find start index and end index of each cell.
        findCellRangeGranular(
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            granular_component->m_num_particles,
            m_num_grid_cells);
        getLastCudaErrorGranular("findCellRangeGranular");
    }

    // constraint
    {
        unsigned int iter = 0;
        while (iter < m_params->m_max_iter_nums)
        {
            solveDistanceConstrainGranluar(
                ( float4* )(granular_component->m_device_pos),
                ( float4* )granular_component->m_device_vel,
                ( float3* )m_device_delta_pos,
                ( float4* )m_device_predicted_pos,
                granular_component->m_device_phase,
                m_device_cell_start,
                m_device_cell_end,
                m_device_grid_particle_hash,
                granular_component->m_num_particles,
                m_num_grid_cells);
            ++iter;
        }
    }

    // update velocity and position
    {
        updateVelAndPosGranular(
            ( float4* )granular_component->m_device_pos,
            ( float4* )granular_component->m_device_vel,
            granular_component->m_device_phase,
            static_cast<float>(m_config.m_dt),
            granular_component->m_num_particles,
            ( float4* )m_device_predicted_pos);
    }

    return true;
}

bool GranularParticleSolverIntegration::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_granular_particle_component)
    {
        return false;
    }
    // Update till termination

    int step_id = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        // std::cout << "    step " << step_id << ": " << m_cur_time << " -> " << m_cur_time + dt << "\n";
        m_cur_time += dt;
        m_config.m_dt = dt;
        step();
        ++step_id;
        if (m_config.m_write2ply)
            writeToPly(step_id);
    }
    return true;
}

//bool GranularParticleSolverIntegration::isApplicable(const Object* object) const
//{
//    if (!object)
//        return false;
//
//    return object->hasComponent<GranularParticleComponentIntegration>();
//}

//bool GranularParticleSolverIntegration::attachObject(Object* object)
//{
//    if (!object)
//        return false;
//
//    if (!this->isApplicable(object))
//    {
//        std::cout << "error: object is not applicable.\n";
//        return false;
//    }
//
//    if (object->hasComponent<GranularParticleComponentIntegration>())
//    {
//        // std::cout << "object attached as granular particle system.\n";
//        m_granular_particle_component                   = object;
//        GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
//        m_params->m_num_particles             = granular_component->m_num_particles;
//        m_host_pos                            = new float[m_params->m_num_particles * 4];
//    }
//    initialize();
//    return true;
//}

//bool GranularParticleSolverIntegration::detachObject(Object* object)
//{
//    if (!object)
//        return false;
//
//    if (m_granular_particle_component == object)
//    {
//        m_granular_particle_component = nullptr;
//        return true;
//    }
//    std::cout << "    error: object is not attached to the solver.\n";
//    return false;
//}

void GranularParticleSolverIntegration::clearAttachment()
{
    m_granular_particle_component = nullptr;
}

void GranularParticleSolverIntegration::config(SolverConfig& config)
{
    config.m_static_friction          = std::clamp(config.m_dynamic_friction, 0.f, 1.f);
    m_params->m_dynamic_fricton_coeff = config.m_dynamic_friction;
    config.m_static_friction          = std::clamp(config.m_static_friction, 0.f, 1.f);
    m_params->m_static_fricton_coeff  = config.m_static_friction;
    m_params->m_gravity               = make_float3(0.f, config.m_gravity, 0.f);
    if (config.m_stiffness <= 0.f)
        config.m_stiffness = 0.1f;
    m_params->m_stiffness       = config.m_stiffness;
    m_params->m_sleep_threshold = config.m_sleep_threshold;
    if (config.m_solver_iteration < 0)
        config.m_solver_iteration = 1;
    m_params->m_max_iter_nums = config.m_solver_iteration;
    m_config                  = config;
}

bool GranularParticleSolverIntegration::setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz)
{
    if (!m_is_init)
        return false;
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return false;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
        return false;

    m_params->m_world_box_corner1 = make_float3(lx, ly, lz);
    m_params->m_world_box_corner2 = make_float3(ux, uy, uz);

    return true;
}

void GranularParticleSolverIntegration::getWorldBoundary(float& lx, float& ly, float& lz, float& ux, float& uy, float& uz)
{   
    lx = m_params->m_world_box_corner1.x;
    ly = m_params->m_world_box_corner1.y;
    lz = m_params->m_world_box_corner1.z;
    ux = m_params->m_world_box_corner2.x;
    uy = m_params->m_world_box_corner2.y;
    uz = m_params->m_world_box_corner2.z;
}

void GranularParticleSolverIntegration::setStiffness(float& stiffness)
{
    if (stiffness > 1.f)
        stiffness = 1.f;
    else if (stiffness <= 0)
        stiffness = 0.1;
    m_params->m_stiffness = stiffness;
}

void GranularParticleSolverIntegration::getStiffness(float& stiffness)
{
    stiffness = m_params->m_stiffness;
}

void GranularParticleSolverIntegration::setSolverIteration(unsigned int& iteration)
{
    if (iteration <= 0)
        iteration = 1;
    m_params->m_max_iter_nums = iteration;
}

void GranularParticleSolverIntegration::getSolverIteration(unsigned int& iteration)
{
    iteration = m_params->m_max_iter_nums;
}

void GranularParticleSolverIntegration::setGravity(const float& gravity)
{
    m_params->m_gravity = make_float3(0, gravity, 0);
}

void GranularParticleSolverIntegration::setGravity(const float& x, const float& y, const float& z)
{
    m_params->m_gravity = make_float3(x, y, z);
}

void GranularParticleSolverIntegration::getGravity(float& x, float& y, float& z)
{
    x = m_params->m_gravity.x;
    y = m_params->m_gravity.y;
    z = m_params->m_gravity.z;
}

void GranularParticleSolverIntegration::setSleepThreshold(float threshold)
{
    if (threshold < 0.f)
        threshold = 0.f;
    m_params->m_sleep_threshold = threshold;
}

void GranularParticleSolverIntegration::getSleepThreshold(float& threshold)
{
    threshold = m_params->m_sleep_threshold;
}

void GranularParticleSolverIntegration::setWorldOrigin(const float& x, const float& y, const float& z)
{
    m_params->m_world_origin = make_float3(x, y, z);
}

void GranularParticleSolverIntegration::getWorldOrigin(float& x, float& y, float& z)
{
    x = m_params->m_world_origin.x;
    y = m_params->m_world_origin.y;
    z = m_params->m_world_origin.z;
}

void GranularParticleSolverIntegration::setHeightField(const std::vector<float>& height, const float unit_height, const int height_x_num, const int height_z_num)
{
    m_unit_height   = unit_height;
    m_height_x_num  = height_x_num;
    m_height_z_num  = height_z_num;
    m_host_height.clear();
    m_host_height.insert(m_host_height.end(), height.begin(), height.end());
    cudaFree(m_device_height);
    getLastCudaErrorGranular("m_device_height");
    cudaMalloc(( void** )&m_device_height, sizeof(float) * m_height_x_num * m_height_z_num);    
    cudaMemcpy(m_device_height, m_host_height.data(), sizeof(float) * m_height_x_num * m_height_z_num, cudaMemcpyHostToDevice);
}

void GranularParticleSolverIntegration::setStaticFrictionCoeff(float& staticFriction)
{
    m_params->m_static_fricton_coeff = std::clamp(staticFriction, 0.f, 1.f);
}

void GranularParticleSolverIntegration::getStaticFrictionCoeff(float& staticFriction)
{
    staticFriction = m_params->m_static_fricton_coeff;
}

void GranularParticleSolverIntegration::setDynamicFrictionCoeff(float& dynamicFriction)
{
    m_params->m_dynamic_fricton_coeff = std::clamp(dynamicFriction, 0.f, 1.f);
}

void GranularParticleSolverIntegration::getDynamicFrictionCoeff(float& dynamicFriction)
{
    dynamicFriction = m_params->m_dynamic_fricton_coeff;
}

void GranularParticleSolverIntegration::_finalize()
{
    if (m_host_pos != nullptr)
        delete m_host_pos;
    if (m_granular_particle_component != nullptr)
    {
        if (m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
            m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/->freeMemory();
    }
    freeParticleMemory(m_params->m_num_particles);
    if (m_params != nullptr)
        delete m_params;
}

bool GranularParticleSolverIntegration::setParticlePosition(const std::vector<float>& position)
{
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return false;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
        return false;

    // free particle component and solver data on device.
    if (granular_component->m_bInitialized)
    {
        granular_component->m_bInitialized = false;
        freeParticleMemory(position.size() / 4);
    }
    m_num_particles                     = static_cast<unsigned int>(position.size() / 4);
    granular_component->m_num_particles = m_num_particles;

    // malloc particle component data on device
    size_t mem_size = granular_component->m_num_particles * 4 * sizeof(float);
    granular_component->initialize();
    cudaMemcpy(granular_component->m_device_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemset(granular_component->m_device_phase, 1, mem_size);
    cudaMemset(granular_component->m_device_vel, 0, mem_size);
    granular_component->m_bInitialized = true;

    // malloc solver data on device
    mallocParticleMemory(m_num_particles);
    return true;
}

bool GranularParticleSolverIntegration::setParticleVelocity(const std::vector<float>& velocity)
{
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return false;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
        return false;

    if (static_cast<unsigned int>(velocity.size() / 4) != granular_component->m_num_particles)
        return false;

    cudaMemcpy(granular_component->m_device_pos, velocity.data(), velocity.size() * sizeof(float), cudaMemcpyHostToDevice);

    granular_component->m_bInitialized = true;
    return true;
}

bool GranularParticleSolverIntegration::setParticlePhase(const std::vector<float>& particlePhase)
{
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
    {
        GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
        if (granular_component != nullptr)
        {
            if (granular_component->m_bInitialized)
            {
                if (particlePhase.size() / 4 == granular_component->m_num_particles)
                {
                    cudaMemcpy(granular_component->m_device_phase, particlePhase.data(), particlePhase.size() * sizeof(float), cudaMemcpyHostToDevice);
                    return true;
                }
            }
        }
    }
    return false;
}

bool GranularParticleSolverIntegration::setParticleCollisionForcePtr(float* collisionForce)
{
    if (m_granular_particle_component != nullptr)
    {
        if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        {
            GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
            if (granular_component != nullptr)
            {
                granular_component->m_collision_force = collisionForce;
                return true;
            }
                
        }
    }
    return false;
}

float* GranularParticleSolverIntegration::getParticlePositionPtr()
{
    if (m_granular_particle_component != nullptr)
    {
        if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        {
            GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
            if (granular_component != nullptr)
                return granular_component->m_device_pos;
        }
    }
    return nullptr;
}

float* GranularParticleSolverIntegration::getParticleVelocityPtr()
{
    if (m_granular_particle_component != nullptr)
    {
        if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        {
            GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
            if (granular_component != nullptr)
                return granular_component->m_device_vel;
        }
    }
    return nullptr;
}

float* GranularParticleSolverIntegration::getParticlePhasePtr()
{
    if (m_granular_particle_component != nullptr)
    {
        if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        {
            GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
            if (granular_component != nullptr)
                return granular_component->m_device_phase;
        }
    }
    return nullptr;
}

float* GranularParticleSolverIntegration::getParticleCollisionForcePtr()
{
    if (m_granular_particle_component != nullptr)
    {
        if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        {
            GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
            if (granular_component != nullptr)
                return granular_component->m_collision_force;
        }
    }
    return nullptr;
}

void GranularParticleSolverIntegration::setParticleRadius(const float& particleRadius)
{
    if (!m_is_init)
        return;
    if (particleRadius <= 0)
        return;
    m_params->m_particle_radius = particleRadius;
    // smooth kernel radius.
    m_params->m_sph_radius         = static_cast<float>(4.0 * particleRadius);
    m_params->m_sph_radius_squared = m_params->m_sph_radius * m_params->m_sph_radius;
    // reset density.
    m_params->m_rest_density     = static_cast<float>(1.0f / (8.0f * powf(particleRadius, 3.0f)));
    m_params->m_inv_rest_density = static_cast<float>(1.0f / m_params->m_rest_density);

    // sph kernel function coff.
    m_params->m_poly6_coff      = static_cast<float>(315.0f / (64.0f * M_PI * powf(m_params->m_sph_radius, 9.0)));
    m_params->m_spiky_grad_coff = static_cast<float>(-45.0f / (M_PI * powf(m_params->m_sph_radius, 6.0)));
    m_params->m_one_div_wPoly6  = static_cast<float>(1.0f / (m_params->m_poly6_coff * pow(m_params->m_sph_radius_squared - pow(0.1 * m_params->m_sph_radius, 2.0), 3.0)));

    float cellSize        = m_params->m_sph_radius;
    m_params->m_cell_size = make_float3(cellSize, cellSize, cellSize);
}

void GranularParticleSolverIntegration::getParticleRadius(float& particleRadius)
{
    if (!m_is_init)
        particleRadius = 0;
    particleRadius = m_params->m_particle_radius;
}

// TODO(YuanmuXu: xyuan1517@gmail.com)
void GranularParticleSolverIntegration::handleCollision(unsigned int* collision_particle_id, float* moveDirection, float* moveDistance, unsigned int collision_num)
{
    if (collision_num == 0)
        return;
    if (collision_particle_id == nullptr)
        return;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
        return;
    solverColisionConstrainGranular(
        ( float4* )granular_component->m_device_pos,
        ( float4* )m_device_predicted_pos,
        ( float3* )moveDirection,
        moveDistance,
        granular_component->m_device_phase,
        collision_particle_id,
        collision_num);
}

void GranularParticleSolverIntegration::writeToPly(const int& step_id)
{
 //   if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
 //       return;
 //   GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
 //   if (granular_component == nullptr)
 //       return;

 //   if (step_id == 1)
 //       lastExecutionTime = std::chrono::system_clock::now();
 //   auto                          currentTime  = std::chrono::system_clock::now();
 //   std::chrono::duration<double> timeDiff     = currentTime - lastExecutionTime;
 //   auto                          microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timeDiff);
 //   long                          usPart       = microseconds.count() % 1000000;
	//// output framerate
	//if (step_id == 1)
	//	std::cout << step_id << ": "
	//	<< "\tframerate: " << 60.0 << " fps\t" << m_cur_time - m_config.m_dt << " -> " << m_cur_time << " s" << std::endl;
	//else
	//	std::cout << step_id << ": "
	//	<< "\tframerate: " << 1.0 / timeDiff.count() << " fps\t" << m_cur_time - m_config.m_dt << " -> " << m_cur_time << " s" << std::endl;
	//lastExecutionTime = currentTime;
 //   
 //   cudaMemcpy(m_host_pos, granular_component->m_device_pos, sizeof(float4) * granular_component->m_num_particles, cudaMemcpyDeviceToHost);
 //   //  write to ply file
 //   std::string file_dir = "./ply";
 //   if (std::filesystem::exists(file_dir))
 //   {
 //       std::string filename = "./ply/granular_" + std::to_string(step_id) + ".ply";
 //       std::cout << "write to ply: " << filename << std::endl;
 //       // write pos ti ply file
 //       std::ofstream outfile(filename);
 //       outfile << "ply\n";
 //       outfile << "format ascii 1.0\n";
 //       outfile << "element vertex " << granular_component->m_num_particles << "\n";
 //       outfile << "property float x\n";
 //       outfile << "property float y\n";
 //       outfile << "property float z\n";
 //       outfile << "property float w\n";
 //       outfile << "property uchar red\n";
 //       outfile << "property uchar green\n";
 //       outfile << "property uchar blue\n";
 //       outfile << "end_header\n";
 //       for (unsigned int i = 0; i < granular_component->m_num_particles * 4; i += 4)
 //       {
 //           outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " " << m_host_pos[i + 3] << " 255 255 255\n";
 //       }
 //       outfile.close();
  //  }
}

void GranularParticleSolverIntegration::freeParticleMemory(const int& numParticles)
{
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return;
    GranularParticleComponentIntegration* granular_component = m_granular_particle_component/*->getComponent<GranularParticleComponentIntegration>()*/;
    if (granular_component == nullptr)
        return;
    granular_component->~GranularParticleComponentIntegration();

    cudaFree(m_device_delta_pos);
    getLastCudaErrorGranular("m_deviceDeltaPos");
    cudaFree(m_device_predicted_pos);
    getLastCudaErrorGranular("m_device_predicted_pos");
    cudaFree(m_device_grid_particle_hash);
    getLastCudaErrorGranular("m_device_grid_particle_hash");
    cudaFree(m_device_cell_start);
    getLastCudaErrorGranular("m_device_cell_start");
    cudaFree(m_device_cell_end);
    getLastCudaErrorGranular("m_device_cell_end");
    cudaFree(m_device_height);
    getLastCudaErrorGranular("m_device_height");

    if (m_host_pos != nullptr)
        delete m_host_pos;
}

void GranularParticleSolverIntegration::mallocParticleMemory(const int& numParticles)
{
    if (!m_granular_particle_component/*->hasComponent<GranularParticleComponentIntegration>()*/)
        return;
    GranularParticleComponentIntegration* granular_component = new GranularParticleComponentIntegration;
    granular_component                                       = m_granular_particle_component /*->getComponent<GranularParticleComponentIntegration>()*/;

    if (granular_component == nullptr)
        return;

    size_t memSize = numParticles * sizeof(float) * 4;
    //granular_component->initialize();
    m_granular_particle_component->initialize();
    cudaMalloc(( void** )&m_device_predicted_pos, memSize);
    cudaMalloc(( void** )&m_device_delta_pos, sizeof(float) * 3 * numParticles);
    cudaMalloc(( void** )&m_device_grid_particle_hash, numParticles * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_start, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_end, m_num_grid_cells * sizeof(unsigned int));    
    cudaMalloc(( void** )&m_device_height, sizeof(float) * m_height_x_num * m_height_z_num);
    cudaMemcpy(m_device_height, m_host_height.data(), sizeof(float) * m_height_x_num * m_height_z_num, cudaMemcpyHostToDevice);


    if (m_host_pos == nullptr)
        m_host_pos = new float[numParticles * 4];
}

void GranularParticleSolverIntegration::setInitialize()
{
    if (m_granular_particle_component)
    {
        //m_granular_particle                           = object;
        GranularParticleComponentIntegration* granular_component = m_granular_particle_component;
        m_params->m_num_particles                                = granular_component->m_num_particles;
        m_host_pos                                               = new float[m_params->m_num_particles * 4];
    }
    initialize();
}





}  // namespace Physika