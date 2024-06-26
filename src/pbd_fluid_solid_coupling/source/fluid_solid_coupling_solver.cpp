/**
 * @file fluid_solid_coupling_solver.cpp
 *
 * @author Yuhang Xu (mr.xuyh@qq.com)
 * @date 2023-08-17
 * @brief This file implements a fluid solid particle solver, primarily intended for dry sand particles.
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The fluid solid particle solver is a numerical tool for predicting the behavior of fluid solid systems,
 * specifically dry sand particles. It operates by taking the initial positions of the particles as input,
 * and outputs their positions at the subsequent time step.
 *
 * This solver is part of the Physika framework and is built to work with other components within the framework.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, and vector_functions.
 * It also includes "fluid solid_particle_solver.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To use this solver, you need to provide the initial positions of the particles(By defining the CouplingParticleComponent).
 * The solver then calculates the positions of the particles at the next time step, which can be retrieved for further use or visualization.
 *
 * The solver can handle errors during computation. Specifically, it includes a method 'getLastCudaErrorCoupling' to
 * retrieve the last CUDA error message.
 *
 * @section WARNING
 *
 * Ensure that the CUDA environment is properly set up, as the solver relies on it for computation.
 *
 */

#include "fluid_solid_coupling_solver.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <numeric>
#include <vector_functions.h>

#include "framework/object.hpp"
#include "fluid_solid_coupling_params.hpp"
#include "helper_math.hpp"


namespace Physika {
/**
 * @brief  : get last cuda error
 * @param[in]  : errorMessage
 */
extern void getLastCudaErrorCoupling(const char* errorMessage);

/**
 * @brief  : set simulation parameters (copy the params from CPU to GPU)
 *  This function will be called before each simulation step to ensure that
 *  the modified simulation parameters can take effect immediately
 *
 * @param[in]  : new simulation parameters
 */
extern void setParameters(FluidSolidCouplingParams* hostParams);

/**
 * @brief  : compute hash value for each particle
 *
 * @param[in]  : gridParticleHash  the grid hash pointer
 * @param[in]  : pos               pointer of the particle position array
 * @param[in]  : numParticles
 */
extern void computeHashCoupling(
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
extern void sortCouplingParticles(
    unsigned int* deviceGridParticleHash,
    unsigned int  numParticles,
    float*        devicePos,
    float*        deviceVel,
    float*        device_radius_pos,
    float*        devicePredictedPos,
    int*          phase);

/**
 * @brief  : reorder the particle data based on the sorted hash value
 *
 * @param[in]  : cellStart  		     pointer of the cell start array
 * @param[in]  : cellEnd    		     pointer of the cell end array
 * @param[in]  : gridParticleHash   	 pointer of the hash array
 * @param[in]  : numParticles         number of particles
 * @param[in]  : numCell              number of grid cells
 */
extern void findCellRangeCoupling(
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
extern void particleAdvection(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    int*         phase,
    float        deltaTime,
    unsigned int numParticles);


/**
 * @brief : solver the constrains for each particles
 *
 * @param[in] : phase        pointer of the particle phase array
 * @param[in/out] : position 	  pointer of the particle position array
 * @param[in/out] : velocity 	  pointer of the particle velocity array
 * @param[in] : predictedPos pointer of the particle predicted position array
 * @param[in] : cellStart
 * @param[in] : cellEnd
 * @param[in] : gridParticleHash
 * @param[in] : numParticles number of particles
 * @param[in] : numCells     number of grid cells
 */
void solveDensityConstrainCoupling(
    int*          phase,
    float4*       postion,
    float4*       velocity,
    float3*       deltaPos,
    float4*       predictedPos,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

/**
 * @brief : solver the constrains for each particles
 *
 * @param[in] : phase        pointer of the particle phase array
 * @param[in/out] : position 	  pointer of the particle position array
 * @param[in/out] : velocity 	  pointer of the particle velocity array
 * @param[in] : predicted_pos pointer of the particle predicted position array
 * @param[in] : radius_pos      pointer of the particle radius pos array
 * @param[in] : rest_cm     centor of the solid
 * @param[in] : gridParticleHash
 * @param[in] : numParticles number of particles
 * @param[in] : numCells     number of grid cells
 */
extern void solveShapeMatchingConstrain(
    int*         phase,
    float4*      position,
    float4*      predicted_pos,
    float3*      delta_pos,
    float3*      radius_pos,
    float3       rest_cm,
    const float  stiffness,
    const bool   allow_stretch,
    float4*      velocity,
    float        deltaTime,
    unsigned int num_particles);

/**
 * @brief : solver the collision constrains for particles which are in contact with the collision object
 *
 * @param[in/out] : position 	  pointer of the particle position array
 * @param[in/out] : predictedPos pointer of the particle predicted position array
 * @param[in]   : phase        pointer of the particle phase array
 * @param[in]   : collision_particle_id article id which are in contact with the collision object
 * @param[in]   : numCollisionParticles the number of particles which are in contact with the collision object
 */
extern void solverColisionConstrainCoupling(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    int*          particlePhase,
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

void CouplingParticleComponent::reset()
{
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_radius_pos, m_radius_pos.data(), sizeof(float) * 3 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(int) * m_num_particles, cudaMemcpyHostToDevice);
}

//void CouplingParticleComponent::addInstance(
//    CouplingParticlePhase mat, 
//    std::string shape, 
//    float3 vel_start, 
//    float3 lb, 
//    float3 size,
//    float  particle_radius,
//    float  particle_inmass)
//{
//    std::vector<float3> pos;
//    if (shape == "cube")
//        pos = generate_cube(lb, size, particle_radius);
//    else if (shape == "box")
//        pos = generate_box(lb, size, particle_radius);
//    else if (shape == "plane-x")
//        pos = generate_plane_X(lb, size, particle_radius);
//    else if (shape == "plane-z")
//        pos = generate_plane_Z(lb, size, particle_radius);
//
//    auto num = pos.size();
//
//    // add pos and vel
//    for (auto partile_pos : pos)
//    {
//        m_host_pos.push_back(partile_pos.x);
//        m_host_pos.push_back(partile_pos.y);
//        m_host_pos.push_back(partile_pos.z);
//        m_host_pos.push_back(particle_inmass);
//
//        m_host_vel.push_back(vel_start.x);
//        m_host_vel.push_back(vel_start.y);
//        m_host_vel.push_back(vel_start.z);
//        m_host_vel.push_back(0.f);
//    }
//
//    // add material
//    auto _mat = std::vector<int>(num, static_cast<int>(mat));
//    m_host_phase.insert(m_host_phase.end(), _mat.begin(), _mat.end());
//    if (mat == CouplingParticlePhase::FLUID)
//        m_num_fluid += pos.size();
//    if (mat == CouplingParticlePhase::SOLID)
//    {
//        m_num_solid += pos.size();
//        // 计算粒子矢径, 用于保存刚体的初始形状
//        float3 rest_centroid = { 0.f };
//        rest_centroid        = std::accumulate(pos.begin(), pos.end(), rest_centroid);
//        rest_centroid /= m_num_solid;
//        for (auto partile_pos : pos)
//        {
//            float3 partocle_radius = partile_pos - rest_centroid;
//            m_radius_pos.push_back(partocle_radius.x);
//            m_radius_pos.push_back(partocle_radius.y);
//            m_radius_pos.push_back(partocle_radius.z);
//        }
//    }
//    else
//    {
//        m_radius_pos.insert(m_radius_pos.end(), num * 3, 0.f);
//    }
//
//    m_num_particles = m_host_pos.size() / 4;
//}

void CouplingParticleComponent::addInstance(
    CouplingParticlePhase mat,
    ParticleModelConfig   config,
    float3                vel_start,
    float                 particle_inmass) 
{
    auto cube = ModelHelper::create3DParticleModel(config);
    this->addInstance(
        mat,
        cube,
        vel_start,
        particle_inmass);
}

void CouplingParticleComponent::addInstance(
    CouplingParticlePhase mat,
    std::vector<float3>   init_pos,
    float3                vel_start,
    float                 particle_inmass)
{
    auto num = init_pos.size();

    // add pos and vel
    for (auto partile_pos : init_pos)
    {
        m_host_pos.push_back(partile_pos.x);
        m_host_pos.push_back(partile_pos.y);
        m_host_pos.push_back(partile_pos.z);
        m_host_pos.push_back(particle_inmass);

        m_host_vel.push_back(vel_start.x);
        m_host_vel.push_back(vel_start.y);
        m_host_vel.push_back(vel_start.z);
        m_host_vel.push_back(0.f);
    }

    // add material
    auto _mat = std::vector<int>(num, static_cast<int>(mat));
    m_host_phase.insert(m_host_phase.end(), _mat.begin(), _mat.end());
    if (mat == CouplingParticlePhase::FLUID)
        m_num_fluid += init_pos.size();
    if (mat == CouplingParticlePhase::SOLID)
    {
        m_num_solid += init_pos.size();
        // 计算粒子矢径, 用于保存刚体的初始形状
        float3 rest_centroid = { 0.f };
        rest_centroid        = std::accumulate(init_pos.begin(), init_pos.end(), rest_centroid);
        rest_centroid /= m_num_solid;
        for (auto partile_pos : init_pos)
        {
            float3 partocle_radius = partile_pos - rest_centroid;
            m_radius_pos.push_back(partocle_radius.x);
            m_radius_pos.push_back(partocle_radius.y);
            m_radius_pos.push_back(partocle_radius.z);
        }
    }
    else
    {
        m_radius_pos.insert(m_radius_pos.end(), num * 3, 0.f);
    }

    m_num_particles = m_host_pos.size() / 4;
}

void CouplingParticleComponent::initialize(int numParticles)
{
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    unsigned int memSize = sizeof(float) * 4 * m_num_particles;

    // allocation
    cudaMalloc(( void** )&m_device_pos, memSize);
    getLastCudaErrorCoupling("allocation1: pos");
    cudaMalloc(( void** )&m_device_vel, memSize);
    cudaMalloc(( void** )&m_device_phase, sizeof(int) * m_num_particles);
    cudaMalloc(( void** )&m_device_radius_pos, sizeof(float) * 3 * m_num_particles);
    cudaMalloc(( void** )&m_device_collision_force, sizeof(float) * 3 * m_num_particles);
    getLastCudaErrorCoupling("allocation");

    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_radius_pos, m_radius_pos.data(), sizeof(float) * 3 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(int) * m_num_particles, cudaMemcpyHostToDevice);

    m_bInitialized = true;
}
void CouplingParticleComponent::freeMemory()
{
    cudaFree(m_device_pos);
    getLastCudaErrorCoupling("m_device_pos");
    cudaFree(m_device_phase);
    getLastCudaErrorCoupling("m_device_phase");
    cudaFree(m_device_vel);
    getLastCudaErrorCoupling("m_device_vel");
    cudaFree(m_device_radius_pos);
    getLastCudaErrorCoupling("m_device_radius_pos");
    cudaFree(m_device_collision_force);
    getLastCudaErrorCoupling("m_device_collision_force");
}
CouplingParticleComponent::~CouplingParticleComponent()
{
    // free gpu memory
    if (m_bInitialized)
    {
        m_bInitialized = false;
    }
}

ParticleFluidSolidCouplingSolver::ParticleFluidSolidCouplingSolver()
    : m_is_init(false), m_particle(nullptr), m_cur_time(0), m_host_pos(nullptr), m_collision_particle_id(nullptr), m_device_cell_end(nullptr), m_device_cell_start(nullptr), m_device_delta_pos(nullptr), m_device_grid_particle_hash(nullptr), m_device_predicted_pos(nullptr)
{
    m_params = new FluidSolidCouplingParams();
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
    m_params->m_world_box_corner1 = make_float3(-40, -20, -20);
    m_params->m_world_box_corner2 = make_float3(40, 20, 20);

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
}

ParticleFluidSolidCouplingSolver::~ParticleFluidSolidCouplingSolver()
{
    if (m_is_init)
        _finalize();
}

bool ParticleFluidSolidCouplingSolver::initialize()
{
    if (m_is_init)
        return true;
    if (m_particle == nullptr)
    {
        std::cout << "ERROR: Must set fluid solid particle object first.\n";
        return false;
    }
    if (m_particle->hasComponent<CouplingParticleComponent>() == false)
    {
        std::cout << "ERROR: fluid solid particle object has no fluid solid component.\n";
        return false;
    }
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
    {
        std::cout << "ERROR: no fluid solid component.\n";
        return false;
    }
    mallocParticleMemory(m_params->m_num_particles);

    m_is_init = true;
    std::cout << "fluid solid solver initialized successfully.\n";
    return true;
}

bool ParticleFluidSolidCouplingSolver::isInitialized() const
{
    return m_is_init;
}

bool ParticleFluidSolidCouplingSolver::reset()
{
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_particle   = nullptr;
    m_cur_time            = 0.0;
    if (m_particle != nullptr)
    {
        m_particle->getComponent<CouplingParticleComponent>()->reset();
    }
    return true;
}

bool ParticleFluidSolidCouplingSolver::step()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return false;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();

    if (component == nullptr)
        return false;

    setParameters(m_params);

    // advect
    {
        particleAdvection(
            ( float4* )component->m_device_pos,
            ( float4* )component->m_device_vel,
            ( float4* )m_device_predicted_pos,
            ( float3* )component->m_device_collision_force,
            component->m_device_phase,
            static_cast<float>(m_config.m_dt),
            component->m_num_particles);

        getLastCudaErrorCoupling("Advection");
    }

    // find neighbours
    {
        // calculate grid Hash.
        computeHashCoupling(
            m_device_grid_particle_hash,
            m_device_predicted_pos,
            component->m_num_particles);
        getLastCudaErrorCoupling("computeHashCoupling");

        // sort particles based on hash value.
        sortCouplingParticles(
            m_device_grid_particle_hash,
            component->m_num_particles,
            component->m_device_pos,
            component->m_device_vel,
            component->m_device_radius_pos,
            m_device_predicted_pos,
            component->m_device_phase);
        getLastCudaErrorCoupling("sortParticles");

        // find start index and end index of each cell.
        findCellRangeCoupling(
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            component->m_num_particles,
            m_num_grid_cells);
        getLastCudaErrorCoupling("findCellRange");
    }

    {
        unsigned int iter = 0;
        while (iter < m_params->m_max_iter_nums)
        {
            solveDensityConstrainCoupling(
                component->m_device_phase,
                ( float4* )(component->m_device_pos),
                ( float4* )(component->m_device_vel),
                ( float3* )m_device_delta_pos,
                ( float4* )m_device_predicted_pos,
                m_device_cell_start,
                m_device_cell_end,
                m_device_grid_particle_hash,
                component->m_num_particles,
                m_num_grid_cells);
            ++iter;
        }
    }

    // shape matching Constain
    {
        solveShapeMatchingConstrain(
            component->m_device_phase,
            ( float4* )(component->m_device_pos),
            ( float4* )m_device_predicted_pos,
            ( float3* )m_device_delta_pos,
            ( float3* )component->m_device_radius_pos,
            component->m_rest_cm,
            0.7,
            false,
            ( float4* )component->m_device_vel,
            static_cast<float>(m_config.m_dt),
            component->m_num_particles);
    }

    return true;
}

bool ParticleFluidSolidCouplingSolver::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_particle)
    {
        return false;
    }
    // Update till termination

    int step_id = 0;
    auto run_start_time  = std::chrono::steady_clock::now();
    auto step_begin_time = std::chrono::steady_clock::now();
    auto step_end_time   = std::chrono::steady_clock::now();
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        m_cur_time += dt;
        m_config.m_dt = dt;
        step();
        ++step_id;
        step_end_time            = std::chrono::steady_clock::now();
        double duration_second   = std::chrono::duration<double>(step_end_time - step_begin_time).count();
        double step_begin_second = std::chrono::duration<double>(step_begin_time - run_start_time).count();
        double step_end_second   = std::chrono::duration<double>(step_end_time - run_start_time).count();
        std::cout << "\nstep_id: " << step_id << ", time: " << step_begin_second << " -> " << step_end_second
                  << "\nframe/sec: " << (1.0 / duration_second);
        step_begin_time = step_end_time;
        //std::cout << step_id << std::endl;
        //writeToPly(step_id);
    }
    return true;
}

bool ParticleFluidSolidCouplingSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<CouplingParticleComponent>();
}

bool ParticleFluidSolidCouplingSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<CouplingParticleComponent>())
    {
        //std::cout << "object attached as fluid solid particle system.\n";
        m_particle                   = object;
        CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
        m_params->m_num_particles             = component->m_num_particles;
        m_host_pos                            = new float[m_params->m_num_particles * 4];
    }
    initialize();
    return true;
}

bool ParticleFluidSolidCouplingSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_particle == object)
    {
        m_particle = nullptr;
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void ParticleFluidSolidCouplingSolver::clearAttachment()
{
    m_particle = nullptr;
}

void ParticleFluidSolidCouplingSolver::config(SolverConfig& config)
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

void ParticleFluidSolidCouplingSolver::getWorldBoundary(float& lx, float& ly, float& lz, float& ux, float& uy, float& uz)
{
    lx = m_params->m_world_box_corner1.x;
    ly = m_params->m_world_box_corner1.y;
    lz = m_params->m_world_box_corner1.z;
    ux = m_params->m_world_box_corner2.x;
    uy = m_params->m_world_box_corner2.y;
    uz = m_params->m_world_box_corner2.z;
}

bool ParticleFluidSolidCouplingSolver::setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz)
{
    if (!m_is_init)
        return false;
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return false;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return false;

    m_params->m_world_box_corner1 = make_float3(lx, ly, lz);
    m_params->m_world_box_corner2 = make_float3(ux, uy, uz);

    return true;
}

void ParticleFluidSolidCouplingSolver::getStiffness(float& stiffness)
{
    stiffness = m_params->m_stiffness;
}

void ParticleFluidSolidCouplingSolver::setStiffness(float& stiffness)
{
    if (stiffness > 1.f)
        stiffness = 1.f;
    else if (stiffness <= 0)
        stiffness = 0.1;
    m_params->m_stiffness = stiffness;
}

void ParticleFluidSolidCouplingSolver::getSolverIteration(unsigned int& iteration)
{
    iteration = m_params->m_max_iter_nums;
}

void ParticleFluidSolidCouplingSolver::setSolverIteration(unsigned int& iteration)
{
    if (iteration <= 0)
        iteration = 1;
    m_params->m_max_iter_nums = iteration;
}

void ParticleFluidSolidCouplingSolver::getGravity(float& x, float& y, float& z)
{
    x = m_params->m_gravity.x;
    y = m_params->m_gravity.y;
    z = m_params->m_gravity.z;
}

void ParticleFluidSolidCouplingSolver::setGravity(const float& gravity)
{
    m_params->m_gravity = make_float3(0, gravity, 0);
}

void ParticleFluidSolidCouplingSolver::setGravity(const float& x, const float& y, const float& z)
{
    m_params->m_gravity = make_float3(x, y, z);
}

void ParticleFluidSolidCouplingSolver::getSleepThreshold(float& threshold)
{
    threshold = m_params->m_sleep_threshold;
}

void ParticleFluidSolidCouplingSolver::setSleepThreshold(float& threshold)
{
    if (threshold < 0.f)
        threshold = 0.f;
    m_params->m_sleep_threshold = threshold;
}

void ParticleFluidSolidCouplingSolver::getWorldOrigin(float& x, float& y, float& z)
{
    x = m_params->m_world_origin.x;
    y = m_params->m_world_origin.y;
    z = m_params->m_world_origin.z;
}

void ParticleFluidSolidCouplingSolver::setWorldOrigin(const float& x, const float& y, const float& z)
{
    m_params->m_world_origin = make_float3(x, y, z);
}

void ParticleFluidSolidCouplingSolver::_finalize()
{
    if (m_host_pos != nullptr)
        delete m_host_pos;
    if (m_particle != nullptr)
    {
        if (m_particle->hasComponent<CouplingParticleComponent>())
            m_particle->getComponent<CouplingParticleComponent>()->freeMemory();
    }
    freeParticleMemory(m_params->m_num_particles);
    if (m_params != nullptr)
        delete m_params;
}

bool ParticleFluidSolidCouplingSolver::setParticlePosition(const std::vector<float>& position)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return false;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return false;

    // free particle component and solver data on device.
    if (component->m_bInitialized)
    {
        component->m_bInitialized = false;
        freeParticleMemory(position.size() / 4);
    }
    m_num_particles                     = static_cast<unsigned int>(position.size() / 4);
    component->m_num_particles = m_num_particles;

    // malloc particle component data on device
    size_t mem_size = component->m_num_particles * 4 * sizeof(float);
    component->initialize(m_num_particles);
    cudaMemcpy(component->m_device_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemset(component->m_device_phase, 1, mem_size);
    cudaMemset(component->m_device_vel, 0, mem_size);
    component->m_bInitialized = true;

    // malloc solver data on device
    mallocParticleMemory(m_num_particles);
    return true;
}

bool ParticleFluidSolidCouplingSolver::setParticleVelocity(const std::vector<float>& velocity)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return false;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return false;

    if (static_cast<unsigned int>(velocity.size() / 4) != component->m_num_particles)
        return false;

    cudaMemcpy(component->m_device_pos, velocity.data(), velocity.size() * sizeof(float), cudaMemcpyHostToDevice);

    component->m_bInitialized = true;
    return true;
}

bool ParticleFluidSolidCouplingSolver::setParticlePhase(const std::vector<int>& particlePhase)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
    {
        CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
        if (component != nullptr)
        {
            if (component->m_bInitialized)
            {
                if (particlePhase.size() == component->m_num_particles)
                {
                    cudaMemcpy(component->m_device_phase, particlePhase.data(), particlePhase.size() * sizeof(int), cudaMemcpyHostToDevice);
                    return true;
                }
            }
        }
    }
    return false;
}

float* ParticleFluidSolidCouplingSolver::getParticlePositionPtr()
{
    if (m_particle != nullptr)
    {
        if (!m_particle->hasComponent<CouplingParticleComponent>())
        {
            CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
            if (component != nullptr)
                return component->m_device_pos;
        }
    }
    return nullptr;
}

float* ParticleFluidSolidCouplingSolver::getParticleVelocityPtr()
{
    if (m_particle != nullptr)
    {
        if (!m_particle->hasComponent<CouplingParticleComponent>())
        {
            CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
            if (component != nullptr)
                return component->m_device_vel;
        }
    }
    return nullptr;
}

int* ParticleFluidSolidCouplingSolver::getParticlePhasePtr()
{
    if (m_particle != nullptr)
    {
        if (!m_particle->hasComponent<CouplingParticleComponent>())
        {
            CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
            if (component != nullptr)
                return component->m_device_phase;
        }
    }
    return nullptr;
}

float* ParticleFluidSolidCouplingSolver::getParticleCollisionForcePtr()
{
    if (m_particle != nullptr)
    {
        if (!m_particle->hasComponent<CouplingParticleComponent>())
        {
            CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
            if (component != nullptr)
                return component->m_device_collision_force;
        }
    }
    return nullptr;
}

void ParticleFluidSolidCouplingSolver::setParticleCollisionForcePtr(float* device_collision_force)
{
    if (m_particle != nullptr)
    {
        if (!m_particle->hasComponent<CouplingParticleComponent>())
        {
            CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
            if (component != nullptr)
                component->m_device_collision_force = device_collision_force;
        }
    }
    return;
}

void ParticleFluidSolidCouplingSolver::getParticleRadius(float& particleRadius)
{
    if (!m_is_init)
        particleRadius = 0;
    particleRadius = m_params->m_particle_radius;
}

void ParticleFluidSolidCouplingSolver::setParticleRadius(const float& particleRadius)
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

// TODO(YuanmuXu: xyuan1517@gmail.com)
void ParticleFluidSolidCouplingSolver::handleCollision(unsigned int* collision_particle_id, float* moveDirection, float* moveDistance, unsigned int collision_num)
{
    if (collision_num == 0)
        return;
    if (collision_particle_id == nullptr)
        return;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return;
    solverColisionConstrainCoupling(
        ( float4* )component->m_device_pos,
        ( float4* )m_device_predicted_pos,
        ( float3* )moveDirection,
        moveDistance,
        component->m_device_phase,
        collision_particle_id,
        collision_num);
}

void ParticleFluidSolidCouplingSolver::getStaticFrictionCoeff(float& staticFriction)
{
    staticFriction = m_params->m_static_fricton_coeff;
}

void ParticleFluidSolidCouplingSolver::setStaticFrictionCoeff(float& staticFriction)
{
    if (staticFriction < 0.f)
        staticFriction = 0.f;
    if (staticFriction > 1.f)
        staticFriction = 1.f;
    m_params->m_static_fricton_coeff = staticFriction;
}

void ParticleFluidSolidCouplingSolver::getDynamicFrictionCoeff(float& dynamicFriction)
{
    dynamicFriction = m_params->m_dynamic_fricton_coeff;
}

void ParticleFluidSolidCouplingSolver::setDynamicFrictionCoeff(float& dynamicFriction)
{
    if (dynamicFriction < 0.f)
        dynamicFriction = 0.f;
    if (dynamicFriction > 1.f)
        dynamicFriction = 1.f;
    m_params->m_dynamic_fricton_coeff = dynamicFriction;
}

void ParticleFluidSolidCouplingSolver::writeToPly(const int& step_id)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return;

	std::string file_dir = "./ply";
	if (!std::filesystem::exists(file_dir))
		return;

    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return;
    cudaMemcpy(m_host_pos, component->m_device_pos, sizeof(float4) * component->m_num_particles, cudaMemcpyDeviceToHost);
    int* phase = new int[component->m_num_particles];
    cudaMemcpy(phase, component->m_device_phase, sizeof(int) * component->m_num_particles, cudaMemcpyDeviceToHost);
    
    int phase_sum = 0;
    for (int i = 0; i < component->m_num_particles; ++i)
    {
        phase_sum += phase[i] - 1;
    }
    //  write to ply file
    //file_dir = file_dir + "/fluid-solid_volume_600_invmass-1.25_v1";
    //file_dir = file_dir + "/fluid-solid_volume_600_invmass-1.0_v1";
    //file_dir = file_dir + "/fluid-solid_volume_900_invmass-1.5_v1";
    //file_dir = file_dir + "/fluid-solid_volume_1000_invmass-1.5_v1";
    //file_dir = file_dir + "/fluid-solid_volume_27_invmass-1.5_v1";
    //file_dir = file_dir + "/fluid-solid_volume_27_invmass-0.1_v1";
    //file_dir = file_dir + "/fluid-solid_volume_1000_invmass-0.1_v1";
    //file_dir = file_dir + "/fluid-solid_invmass-0.02_v1";
    //file_dir = file_dir + "/fluid-solid_invmass-10.0_v1";
    //file_dir = file_dir + "/fluid-solid_invmass-0.1_v1";
    std::string filename = file_dir + "/frame_" + std::to_string(step_id) + ".ply";
    //std::string filename = file_dir + "/solid_" + std::to_string(step_id) + ".ply";
    std::cout << "write to ply: " << filename << std::endl;
    // write pos ti ply file
    std::ofstream outfile(filename);
    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << component->m_num_particles << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property float w\n";
    outfile << "property uchar red\n";
    outfile << "property uchar green\n";
    outfile << "property uchar blue\n";
    outfile << "end_header\n";
    float exp = 1e-6;
    for (unsigned int i = 0, j = 0; i < component->m_num_particles * 4; i += 4, j += 1)
    {
        outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " " << m_host_pos[i + 3];
    
        if (phase[j] == static_cast<int>(CouplingParticlePhase::FLUID))
        //if (fabs(phase[j] - float(CouplingParticlePhase::FLUID)) < exp)
        {
            outfile << " 255 255 255\n";
        }
        else
        {
            outfile << " 255 0 0\n";
        }
    }
    outfile.close();
}

void ParticleFluidSolidCouplingSolver::freeParticleMemory(const int& numParticles)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return;
    component->~CouplingParticleComponent();

    cudaFree(m_device_delta_pos);
    getLastCudaErrorCoupling("m_deviceDeltaPos");
    cudaFree(m_device_predicted_pos);
    getLastCudaErrorCoupling("m_device_predicted_pos");
    cudaFree(m_device_grid_particle_hash);
    getLastCudaErrorCoupling("m_device_grid_particle_hash");
    cudaFree(m_device_cell_start);
    getLastCudaErrorCoupling("m_device_cell_start");
    cudaFree(m_device_cell_end);
    getLastCudaErrorCoupling("m_device_cell_end");

    if (m_host_pos != nullptr)
        delete m_host_pos;
}

void ParticleFluidSolidCouplingSolver::mallocParticleMemory(const int& numParticles)
{
    if (!m_particle->hasComponent<CouplingParticleComponent>())
        return;
    CouplingParticleComponent* component = m_particle->getComponent<CouplingParticleComponent>();
    if (component == nullptr)
        return;

    size_t memSize = numParticles * sizeof(float) * 4;
    component->initialize(numParticles);
    cudaMalloc(( void** )&m_device_predicted_pos, memSize);
    cudaMalloc(( void** )&m_device_delta_pos, sizeof(float) * 3 * numParticles);
    cudaMalloc(( void** )&m_device_grid_particle_hash, numParticles * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_start, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_end, m_num_grid_cells * sizeof(unsigned int));

    if (m_host_pos == nullptr)
        m_host_pos = new float[numParticles * 4];
}

}  // namespace Physika