/**
 * @file fluid_particle_solver.cpp
 *
 * @author Yuege Xiong (candybear0714@163.com)
 * @date 2023-11-23
 * @brief This file implements a fluid particle solver, primarily intended for dry sand particles.
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The fluid particle solver is a numerical tool for predicting the behavior of fluid systems,
 * specifically dry sand particles. It operates by taking the initial positions of the particles as input,
 * and outputs their positions at the subsequent time step.
 *
 * This solver is part of the Physika framework and is built to work with other components within the framework.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, and vector_functions.
 * It also includes "fluid_particle_solver.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To use this solver, you need to provide the initial positions of the particles(By defining the FluidComponent).
 * The solver then calculates the positions of the particles at the next time step, which can be retrieved for further use or visualization.
 *
 * The solver can handle errors during computation. Specifically, it includes a method 'getLastCudaError' to
 * retrieve the last CUDA error message.
 *
 * @section WARNING
 *
 * Ensure that the CUDA environment is properly set up, as the solver relies on it for computation.
 *
 */


#include "fluid_particle_solver.hpp"
#include "fluid_params.hpp"
#include "framework/object.hpp"

#include <iostream>
#include <string>
#include <fstream>

#include <vector_functions.h>


namespace Physika {
/**
 * @brief  : get last cuda error
 * @param[in]  : errorMessage
 */
extern void getLastCudaError(const char* errorMessage);

/**
 * @brief  : set simulation parameters (copy the params from CPU to GPU)
 *  This function will be called before each simulation step to ensure that
 *  the modified simulation parameters can take effect immediately
 *
 * @param[in]  : new simulation parameters
 */
extern void setParameters(PBDFluidSimulateParams* hostParams);

/**
 * @brief  : compute hash value for each particle
 *
 * @param[in]  : gridParticleHash  the grid hash pointer
 * @param[in]  : pos               pointer of the particle position array
 * @param[in]  : numParticles
 */
extern void computeHash(
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
extern void sortParticles(
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
extern void findCellRange(
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCell);

extern void solveContactConstrain(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        deltaTime,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

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
extern void fluidAdvection(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       phase,
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
extern void updateVelAndPos(
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
extern void solveDistanceConstrain(
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
extern void solveDensityConstrain(
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
extern void solverColisionConstrain(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles);

extern void add_surface_tension(
    float4*       velocity,
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    float         deltaTime,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

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

void PBDFluidComponent::reset()
{
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(float) * m_num_particles, cudaMemcpyHostToDevice);
}
/*
FluidComponent::FluidComponent(float radius)
    : m_device_pos(nullptr), m_device_vel(nullptr), m_bInitialized(false)
{
    float spacing = radius * 2.0f;
    float jitter  = radius * 0.01f;

    srand(1973);
    unsigned int numParticles = 0;
    m_num_collisions          = 0;
    // fluid Deambreak
    //  bottom fluid.
    MyFloatVec3 bottomFluidSize = { 20.0f, 20.0f, 20.0f };
    MyIntVec3   bottomFluidDim  = { static_cast<int>(bottomFluidSize.x / spacing),
                                    static_cast<int>(bottomFluidSize.y / spacing),
                                    static_cast<int>(bottomFluidSize.z / spacing) };

    for (int z = 0; z < bottomFluidDim.z; ++z)
    {
        for (int y = 0; y < bottomFluidDim.y; ++y)
        {
            for (int x = 0; x < bottomFluidDim.x; ++x)
            {
                m_host_pos.push_back(spacing * x + radius - 0.5f * 80.0f + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * y + radius - 0.5f * 40.0f + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * z + radius - 0.5f * 40.0f + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(1.0f);
                m_host_phase.push_back(static_cast<float>(FluidPhase::FLUID));

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
            }
        }
    }

    numParticles = static_cast<unsigned int>(m_host_pos.size() / 4);
    initialize(numParticles);
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice);
}*/


PBDFluidComponent::PBDFluidComponent(
    float radius,
    float solid_cube_x,
    float solid_cube_y,
    float solid_cube_z)
    : m_device_pos(nullptr), m_device_vel(nullptr), m_bInitialized(false)
{
    unsigned int       numParticles = 0;
    float              spacing      = radius * 2.0f;
    float              jitter       = radius * 0.01f;
    
    srand(1973);

    // scene of surface tension£¨hydrophilic & hydrophobic£©
    // bottom bound phase = 3
    MyFloatVec3 bottomSolidSize = { solid_cube_x, solid_cube_y, solid_cube_z };
    MyIntVec3   bottomSolidDim  = { static_cast<int>(bottomSolidSize.x / spacing),
                                    static_cast<int>(bottomSolidSize.y / spacing),
                                    static_cast<int>(bottomSolidSize.z / spacing) };

    for (int z = 0; z < bottomSolidDim.z; ++z)
    {
        for (int y = 0; y < bottomSolidDim.y; ++y)
        {
            for (int x = 0; x < bottomSolidDim.x; ++x)
            {

                m_host_pos.push_back(spacing * x + radius - 0.5f * bottomSolidSize.x + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * y + radius - 0.5f * bottomSolidSize.y + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(spacing * z + radius - 0.5f * bottomSolidSize.z + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(1.f);
                m_host_phase.push_back(static_cast<float>(PBDFluidPhase::ELASTIC));

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
            }
        }
    }

    // waterdrop phase = 0
    float spherRadius = 6.0f;
    float boxLength   = 2.0f * spherRadius;
    int   boxDim      = boxLength / spacing;
    for (int z = 0; z < boxDim; ++z)
    {
        for (int y = 0; y < boxDim; ++y)
        {
            for (int x = 0; x < boxDim; ++x)
            {
                float dx = x * spacing - 0.5f * boxLength;
                float dy = y * spacing - 0.5f * boxLength;
                float dz = z * spacing - 0.5f * boxLength;
                float l  = sqrtf(dx * dx + dy * dy + dz * dz);
                if (l > spherRadius)
                    continue;

                m_host_pos.push_back(dx + radius + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(dy + radius + (frand() * 2.0f - 1.0f) * jitter + 9.f);
                m_host_pos.push_back(dz + radius + (frand() * 2.0f - 1.0f) * jitter);
                m_host_pos.push_back(1.f);
                m_host_phase.push_back(static_cast<float>(PBDFluidPhase::FLUID));

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(-2.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
            }
        }
    }

    numParticles = static_cast<unsigned int>(m_host_pos.size() / 4);
    initialize(numParticles);
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_phase, m_host_phase.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice);
}

void PBDFluidComponent::addParticlePos(std::vector<float3>& pos, int phase = static_cast<int>(PBDFluidPhase::FLUID))
{

    unsigned int numParticles = 0;

    for (int i = 0; i < static_cast<unsigned int>(pos.size()); ++i)
    {
        m_host_pos.push_back(pos[i].x);
        m_host_pos.push_back(pos[i].y);
        m_host_pos.push_back(pos[i].z);
        m_host_pos.push_back(1.f);

        m_host_phase.push_back(static_cast<float>(PBDFluidPhase::FLUID));

        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
    }

    numParticles = static_cast<unsigned int>(m_host_pos.size() / 4);
}

void PBDFluidComponent::initialize(int numParticles)
{
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    m_num_particles      = numParticles;
    //m_particle_radius    = 0.4f;
    unsigned int memSize = sizeof(float) * 4 * m_num_particles;

    // allocation
    cudaMalloc(( void** )&m_device_pos, memSize);
    getLastCudaError("allocation1: pos");
    cudaMalloc(( void** )&m_device_vel, memSize);
    cudaMalloc(( void** )&m_device_phase, sizeof(float) * m_num_particles);
    cudaMalloc(( void** )&m_external_force, sizeof(float) * 3 * m_num_particles);
    getLastCudaError("allocation");

    m_bInitialized = true;
}
void PBDFluidComponent::freeMemory()
{
    cudaFree(m_device_pos);
    getLastCudaError("m_device_pos");
    cudaFree(m_device_phase);
    getLastCudaError("m_device_phase");
    cudaFree(m_device_vel);
    getLastCudaError("m_device_vel");
    cudaFree(m_external_force);
    getLastCudaError("m_external_force");
    
}
PBDFluidComponent::~PBDFluidComponent()
{
    // free gpu memory
    if (m_bInitialized)
    {
        m_bInitialized = false;
    }
}

PBDFluidParticleSolver::PBDFluidParticleSolver()
    : m_is_init(false), m_fluid_particle(nullptr), m_cur_time(0), m_host_pos(nullptr), m_collision_particle_id(nullptr), m_device_cell_end(nullptr), m_device_cell_start(nullptr), m_device_delta_pos(nullptr), m_device_grid_particle_hash(nullptr), m_device_predicted_pos(nullptr)
{
    m_params = new PBDFluidSimulateParams();
    // particles and grid.
    uint3 gridSize = make_uint3(128, 128, 128);
    float radius   = 0.4f;  // 0.3f

    m_num_grid_cells            = gridSize.x * gridSize.y * gridSize.z;
    m_params->m_num_grid_cells  = m_num_grid_cells;
    m_params->m_grid_size       = gridSize;
    m_params->m_particle_radius = radius;
    m_params->m_num_particles   = 0;
    m_params->m_num_grid_cells  = gridSize.x * gridSize.y * gridSize.z;

    // iteration number.
    m_params->m_max_iter_nums = 5;
      
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

    // fluid surface tension
    m_params->m_sf_coeff       = 1.5f;
    m_params->m_adhesion_coeff = 0.8f; 
    m_params->m_scaling_factor = 3.5f;

    // grid cells.
    float cellSize           = m_params->m_sph_radius;
    m_params->m_cell_size    = make_float3(cellSize, cellSize, cellSize);
    m_params->m_gravity      = make_float3(0.0f, -9.8f, 0.0f);
    m_params->m_world_origin = { -42.0f, -22.0f, -22.0f };
}

PBDFluidParticleSolver::~PBDFluidParticleSolver()
{
    if (m_is_init)
        _finalize();
}

bool PBDFluidParticleSolver::initialize()
{
    if (m_is_init)
        return true;
    if (m_fluid_particle == nullptr)
    {
        std::cout << "ERROR: Must set fluid particle object first.\n";
        return false;
    }
    if (m_fluid_particle->hasComponent<PBDFluidComponent>() == false)
    {
        std::cout << "ERROR: fluid particle object has no fluid component.\n";
        return false;
    }
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
    {
        std::cout << "ERROR: no fluid component.\n";
        return false;
    }
    mallocParticleMemory(m_params->m_num_particles);

    m_is_init = true;
    std::cout << "fluid solver initialized successfully.\n";
    return true;
}

bool PBDFluidParticleSolver::isInitialized() const
{
    return m_is_init;
}

bool PBDFluidParticleSolver::reset()
{
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_fluid_particle      = nullptr;
    m_cur_time            = 0.0;
    if (m_fluid_particle != nullptr)
    {
        m_fluid_particle->getComponent<PBDFluidComponent>()->reset();
    }
    return true;
}

bool PBDFluidParticleSolver::step()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return false;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();

    if (fluid_component == nullptr)
        return false;

    setParameters(m_params);

    // advect
    {
        fluidAdvection(
            ( float4* )fluid_component->m_device_pos,
            ( float4* )fluid_component->m_device_vel,
            ( float4* )m_device_predicted_pos,
            fluid_component->m_device_phase,
            static_cast<float>(m_config.m_dt),
            fluid_component->m_num_particles);

        getLastCudaError("Advection");
    }

    // solid contact
   solveContactConstrain(
        ( float4* )(fluid_component->m_device_pos),
        ( float4* )fluid_component->m_device_vel,
        ( float4* )m_device_predicted_pos,
        fluid_component->m_device_phase,
        static_cast<float>(m_config.m_dt),
        m_device_cell_start,
        m_device_cell_end,
        m_device_grid_particle_hash,
        fluid_component->m_num_particles,
        m_num_grid_cells);
    getLastCudaError("solveContactConstrain");

    // find neighbours
    {
        // calculate grid Hash.
        computeHash(
            m_device_grid_particle_hash,
            m_device_predicted_pos,
            fluid_component->m_num_particles);
        getLastCudaError("computeHash");

        // sort particles based on hash value.
        sortParticles(
            m_device_grid_particle_hash,
            fluid_component->m_num_particles,
            fluid_component->m_device_pos,
            fluid_component->m_device_vel,
            m_device_predicted_pos,
            fluid_component->m_device_phase);
        getLastCudaError("sortParticles");

        // find start index and end index of each cell.
        findCellRange(
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            fluid_component->m_num_particles,
            m_num_grid_cells);
        getLastCudaError("findCellRange");
    }
   
    // constraint
    {
        unsigned int iter = 0;
        while (iter < m_params->m_max_iter_nums)
        {
            solveDensityConstrain(
                ( float4* )(fluid_component->m_device_pos),
                ( float4* )fluid_component->m_device_vel,
                ( float3* )m_device_delta_pos,
                ( float4* )m_device_predicted_pos,
                fluid_component->m_device_phase,
                m_device_cell_start,
                m_device_cell_end,
                m_device_grid_particle_hash,
                fluid_component->m_num_particles,
                m_num_grid_cells);
            ++iter;
        }
    }

    // update velocity and position
    {
        updateVelAndPos(
            ( float4* )fluid_component->m_device_pos,
            ( float4* )fluid_component->m_device_vel,
            fluid_component->m_device_phase,
            static_cast<float>(m_config.m_dt),
            fluid_component->m_num_particles,
            ( float4* )m_device_predicted_pos);
    }


    add_surface_tension(
        ( float4* )fluid_component->m_device_vel,
        ( float4* )m_device_predicted_pos,
        ( float3* )m_device_delta_pos,
        fluid_component->m_device_phase,
        static_cast<float>(m_config.m_dt),
        m_device_cell_start,
        m_device_cell_end,
        m_device_grid_particle_hash,
        fluid_component->m_num_particles,
        m_num_grid_cells);


    return true;
}

bool PBDFluidParticleSolver::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_fluid_particle)
    {
        return false;
    }
    // Update till termination
    int step_id = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        std::cout << "    step: " << step_id << " step_start_time: " << m_cur_time << " step_start_time: " << m_cur_time + dt << "\n";
        std::cout << "    frame/sec: " << 1/dt << "\n";
        
        m_cur_time += dt;
        m_config.m_dt = dt;
        step();
        ++step_id;
        //writeToPly(step_id);
    }
    return true;
}

bool PBDFluidParticleSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<PBDFluidComponent>();
}

bool PBDFluidParticleSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<PBDFluidComponent>())
    {
        std::cout << "object attached as fluid particle system.\n";
        m_fluid_particle                = object;
        PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
        m_params->m_num_particles       = fluid_component->m_num_particles;
        m_host_pos                      = new float[m_params->m_num_particles * 4];
    }
    initialize();
    return true;
}

bool PBDFluidParticleSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_fluid_particle == object)
    {
        m_fluid_particle = nullptr;
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void PBDFluidParticleSolver::clearAttachment()
{
    m_fluid_particle = nullptr;
}

void PBDFluidParticleSolver::config(SolverConfig& config)
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

bool PBDFluidParticleSolver::setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz)
{
    std::cout << "set world boundary over.\n";
    if (!m_is_init)
        return false;
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return false;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return false;

    m_params->m_world_box_corner1 = make_float3(lx, ly, lz);
    m_params->m_world_box_corner2 = make_float3(ux, uy, uz);

    return true;
}

void PBDFluidParticleSolver::setStiffness(float& stiffness)
{
    if (stiffness > 1.f)
        stiffness = 1.f;
    else if (stiffness <= 0)
        stiffness = 0.1;
    m_params->m_stiffness = stiffness;
}

void PBDFluidParticleSolver::setSolverIteration(unsigned int& iteration)
{
    if (iteration <= 0)
        iteration = 1;
    m_params->m_max_iter_nums = iteration;
}

void PBDFluidParticleSolver::setGravity(const float& gravity)
{
    m_params->m_gravity = make_float3(0, gravity, 0);
}

void PBDFluidParticleSolver::setGravity(const float& x, const float& y, const float& z)
{
    m_params->m_gravity = make_float3(x, y, z);
}

void PBDFluidParticleSolver::setSleepThreshold(float& threshold)
{
    if (threshold < 0.f)
        threshold = 0.f;
    m_params->m_sleep_threshold = threshold;
}

void PBDFluidParticleSolver::setWorldOrigin(const float& x, const float& y, const float& z)
{
    m_params->m_world_origin = make_float3(x, y, z);
}

void PBDFluidParticleSolver::setSFCoeff(const float& sfcoeff)
{
    m_params->m_sf_coeff = sfcoeff;
}

void PBDFluidParticleSolver::setAdhesionCoeff(const float& AdhesionCoeff)
{
    m_params->m_adhesion_coeff = AdhesionCoeff;
}

void PBDFluidParticleSolver::_finalize()
{
    if (m_host_pos != nullptr)
        delete m_host_pos;
    if (m_fluid_particle != nullptr)
    {
        if (m_fluid_particle->hasComponent<PBDFluidComponent>())
            m_fluid_particle->getComponent<PBDFluidComponent>()->freeMemory();
    }
    freeParticleMemory(m_params->m_num_particles);
    if (m_params != nullptr)
        delete m_params;
}

bool PBDFluidParticleSolver::setParticlePosition(const std::vector<float>& position)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return false;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return false;

    // free particle component and solver data on device.
    if (fluid_component->m_bInitialized)
    {
        fluid_component->m_bInitialized = false;
        freeParticleMemory(position.size() / 4);
    }
    m_num_particles                  = static_cast<unsigned int>(position.size() / 4);
    fluid_component->m_num_particles = m_num_particles;

    // malloc particle component data on device
    size_t mem_size = fluid_component->m_num_particles * 4 * sizeof(float);
    fluid_component->initialize(m_num_particles);
    cudaMemcpy(fluid_component->m_device_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemset(fluid_component->m_device_phase, 0, mem_size);
    cudaMemset(fluid_component->m_device_vel, 0, mem_size);
    fluid_component->m_bInitialized = true;

    // malloc solver data on device
    mallocParticleMemory(m_num_particles);
    return true;
}

bool PBDFluidParticleSolver::setParticleVelocity(const std::vector<float>& velocity)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return false;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return false;

    if (static_cast<unsigned int>(velocity.size() / 4) != fluid_component->m_num_particles)
        return false;

    cudaMemcpy(fluid_component->m_device_pos, velocity.data(), velocity.size() * sizeof(float), cudaMemcpyHostToDevice);

    fluid_component->m_bInitialized = true;
    return true;
}

bool PBDFluidParticleSolver::setParticlePhase(const std::vector<float>& particlePhase)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
    {
        PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
        if (fluid_component != nullptr)
        {
            if (fluid_component->m_bInitialized)
            {
                if (particlePhase.size() / 4 == fluid_component->m_num_particles)
                {
                    cudaMemcpy(fluid_component->m_device_phase, particlePhase.data(), particlePhase.size() * sizeof(float), cudaMemcpyHostToDevice);
                    return true;
                }
            }
        }
    }
    return false;
}

bool PBDFluidParticleSolver::setParticleExternalForce(float* external_force)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
    {
        PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
        if (fluid_component != nullptr)
        {
            fluid_component->m_external_force = external_force;
            return true;
        }
    }
    return false;
}

float* PBDFluidParticleSolver::getParticlePositionPtr()
{
    if (m_fluid_particle != nullptr)
    {
        if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        {
            PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
            if (fluid_component != nullptr)
                return fluid_component->m_device_pos;
        }
    }
    return nullptr;
}

float* PBDFluidParticleSolver::getParticleVelocityPtr()
{
    if (m_fluid_particle != nullptr)
    {
        if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        {
            PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
            if (fluid_component != nullptr)
                return fluid_component->m_device_vel;
        }
    }
    return nullptr;
}

float* PBDFluidParticleSolver::getParticlePhasePtr()
{
    if (m_fluid_particle != nullptr)
    {
        if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        {
            PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
            if (fluid_component != nullptr)
                return fluid_component->m_device_phase;
        }
    }
    return nullptr;
}

float* PBDFluidParticleSolver::getParticleExternalForcePtr()
{
    if (m_fluid_particle != nullptr)
    {
        if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        {
            PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
            if (fluid_component != nullptr)
                return fluid_component->m_external_force;
        }
    }
    return nullptr;
}

void PBDFluidParticleSolver::getParticleRadius(float& particleRadius)
{
    if (!m_is_init)
        particleRadius = 0;
    particleRadius = m_params->m_particle_radius;
}

void PBDFluidParticleSolver::setParticleRadius(const float& particleRadius)
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
void PBDFluidParticleSolver::handleCollision(unsigned int* collision_particle_id, float* moveDirection, float* moveDistance, unsigned int collision_num)
{
    if (collision_num == 0)
        return;
    if (collision_particle_id == nullptr)
        return;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return;
    solverColisionConstrain(
        ( float4* )fluid_component->m_device_pos,
        ( float4* )m_device_predicted_pos,
        ( float3* )moveDirection,
        moveDistance,
        fluid_component->m_device_phase,
        collision_particle_id,
        collision_num);
}

void PBDFluidParticleSolver::setStaticFrictionCoeff(float& staticFriction)
{
    if (staticFriction < 0.f)
        staticFriction = 0.f;
    if (staticFriction > 1.f)
        staticFriction = 1.f;
    m_params->m_static_fricton_coeff = staticFriction;
}

void PBDFluidParticleSolver::setDynamicFrictionCoeff(float& dynamicFriction)
{
    if (dynamicFriction < 0.f)
        dynamicFriction = 0.f;
    if (dynamicFriction > 1.f)
        dynamicFriction = 1.f;
    m_params->m_dynamic_fricton_coeff = dynamicFriction;
}

void PBDFluidParticleSolver::writeToPly(const int& step_id)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return;
    cudaMemcpy(m_host_pos, fluid_component->m_device_pos, sizeof(float4) * fluid_component->m_num_particles, cudaMemcpyDeviceToHost);
    //  write to ply file
    std::string filename = "./ply/fluid_" + std::to_string(step_id) + ".ply";
    std::cout << "write to ply: " << filename << "fluid_particle_solver.cpp " << std::endl;
    // write pos ti ply file
    std::ofstream outfile(filename);
    // cout << "mark" << endl;
    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << fluid_component->m_num_particles << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property uchar red\n";
    outfile << "property uchar green\n";
    outfile << "property uchar blue\n";
    outfile << "end_header\n";
    for (unsigned int i = 0; i < fluid_component->m_num_particles * 4; i += 4)
    {
        outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " 255 255 255\n";
    }
    outfile.close();
}

void PBDFluidParticleSolver::freeParticleMemory(const int& numParticles)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return;
    fluid_component->~PBDFluidComponent();

    cudaFree(m_device_delta_pos);
    getLastCudaError("m_deviceDeltaPos");
    cudaFree(m_device_predicted_pos);
    getLastCudaError("m_device_predicted_pos");
    cudaFree(m_device_grid_particle_hash);
    getLastCudaError("m_device_grid_particle_hash");
    cudaFree(m_device_cell_start);
    getLastCudaError("m_device_cell_start");
    cudaFree(m_device_cell_end);
    getLastCudaError("m_device_cell_end");

    if (m_host_pos != nullptr)
        delete m_host_pos;
}

void PBDFluidParticleSolver::mallocParticleMemory(const int& numParticles)
{
    if (!m_fluid_particle->hasComponent<PBDFluidComponent>())
        return;
    PBDFluidComponent* fluid_component = m_fluid_particle->getComponent<PBDFluidComponent>();
    if (fluid_component == nullptr)
        return;

    size_t memSize = numParticles * sizeof(float) * 4;
    fluid_component->initialize(numParticles);
    cudaMalloc(( void** )&m_device_predicted_pos, memSize);
    cudaMalloc(( void** )&m_device_delta_pos, sizeof(float) * 3 * numParticles);
    cudaMalloc(( void** )&m_device_grid_particle_hash, numParticles * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_start, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_end, m_num_grid_cells * sizeof(unsigned int));

    if (m_host_pos == nullptr)
        m_host_pos = new float[numParticles * 4];
}

}  // namespace Physika