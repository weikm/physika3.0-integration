/**
 * @file solid_liquid_phase_change_solver.cpp
 *
 * @author Ruolan Li (3230137958@qq.com)
 * @date 2023-11-17
 * @brief This file implements a solver for simulating solid-liquid phase change processes, designed to model
 *        the transition between solid and liquid states of a material under varying conditions.
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The solid-liquid phase change solver is a computational tool used for predicting the behavior of materials
 * undergoing phase transitions between solid and liquid phases. It takes the initial properties and conditions
 * of the material as input and calculates the evolution of its phase state over time, providing insights into
 * the material's phase change dynamics.
 *
 * This solver is an integral part of the Physika framework and is designed to seamlessly integrate with other
 * components within the framework, facilitating comprehensive material modeling and simulation capabilities.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, and vector_functions.
 * It also includes "solid_liquid_phase_change_solver.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To utilize this solver, it requires the specification of initial material properties and environmental conditions,
 * which are encapsulated within the SolidLiquidComponent. The solver then computes the evolution of the material's
 * phase state, providing access to the resulting properties for further analysis or visualization.
 *
 * The solver incorporates error handling mechanisms, including a method 'getLastCudaError' for retrieving the
 * last CUDA error message in case of computational errors.
 *
 * @section WARNING
 *
 * It is essential to ensure that the CUDA environment is correctly configured and operational, as the solver relies on
 * CUDA for efficient computation of phase change processes.
 *
 */
#include "solid_liquid_phase_change_solver.hpp"

#include <iostream>
#include <string>
#include <fstream>

#include "solid_liquid_phase_change_params.hpp"
#include <vector_functions.h>
#include "framework/object.hpp"

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
extern void setParameters(SolidLiquidPhaseChangeParams* hostParams);

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
 * @brief  Sort all particles based on the hash value
 *
 * @param[in]  : deviceGridParticleHash   Pointer to the grid hash array
 * @param[in]  : numParticles             Number of particles
 * @param[in/out]  : devicePos               Pointer to the particle position array
 * @param[in/out]  : deviceTem               Pointer to the particle temperature array
 * @param[in/out]  : deviceVel               Pointer to the particle velocity array
 * @param[in/out]  : deviceType              Pointer to the particle type array (e.g., solid, liquid, boundary)
 * @param[in/out]  : deviceInitId2           Pointer to the array storing original particle indices after sorting
 * @param[in/out]  : devicePredictedPos      Pointer to the predicted position array for the particles
 */
extern void sortParticlesAll(
	unsigned int *deviceGridParticleHash,
	unsigned int numParticles,
	float *devicePos,
	float* deviceTem,
	float *deviceVel,
	int* deviceType,
	int* deviceInitId2,
	float *devicePredictedPos);

/**
 * @brief  Sort solid particles based on the hash value
 *
 * @param[in]  : m_deviceGridParticleHash_solid  Pointer to the grid hash array for solid particles
 * @param[in]  : numParticles                     Number of solid particles
 * @param[in/out]  : deviceInitPos                   Pointer to the initial position array of solid particles
 * @param[in/out]  : deviceInitId1                   Pointer to the array containing original indices of particles in their initial positions
 */
extern void sortParticlesForSolid(
	unsigned int* m_deviceGridParticleHash_solid,
	unsigned int numParticles,
	float* deviceInitPos,
	int* deviceInitId1);

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
extern void setInitIdToNow(
	int* deviceInitId1,
	int* deviceInitId2,
	int* deviceInitId2Now,
	int* deviceInitId2Rest,
	unsigned int numParticles);


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

/**
 * @brief : advect particles
 *
 * @param[in] : type 	  pointer of the particle type array
 * @param[in] : position 	  pointer of the particle position array
 * @param[in] : velocity 	  pointer of the particle velocity array
 * @param[in] : predictedPos pointer of the particle predicted position array
 * @param[in] : phase        pointer of the particle phase array
 * @param[in] : deltaTime    time step
 * @param[in] : numParticles number of particles
 */
extern void fluidAdvection(
	int *type,
	float4 *position,
    float* temperature,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles,
    float3* external_force);

/**
 * @brief Apply density constraint to fluid and solid particles
 *
 * @param[in] type             Pointer to the array containing particle types
 * @param[in] position         Pointer to the array containing particle positions
 * @param[in] velocity         Pointer to the array containing particle velocities
 * @param[in] temperature      Pointer to the array containing particle temperatures
 * @param[in/out] deltaPos         Pointer to the array containing positional deltas
 * @param[in/out] predictedPos     Pointer to the array containing predicted particle positions
 * @param[in]    cellStart         Pointer to the array containing cell start indices
 * @param[in]    cellEnd           Pointer to the array containing cell end indices
 * @param[in]    gridParticleHash  Pointer to the array containing grid particle hashes
 * @param[in]    numParticles      Number of particles
 * @param[in]    numCells          Number of cells in the grid
 */
extern void densityConstraint(
	int* type,
	float4 *position,
	float4 *velocity,
    float*  temperature,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells);

/**
 * @brief Update temperature of particles
 *
 * @param[in]    type              Pointer to the array containing particle types
 * @param[in] position          Pointer to the array containing particle positions
 * @param[in] temperature       Pointer to the array containing particle temperatures
 * @param[in/out] deltaTem          Pointer to the array containing temperature deltas
 * @param[in/out] latentTem         Pointer to the array containing latent heat values
 * @param[in]    cellStart          Pointer to the array containing cell start indices
 * @param[in]    cellEnd            Pointer to the array containing cell end indices
 * @param[in]    gridParticleHash   Pointer to the array containing grid particle hashes
 * @param[in]    numParticles       Number of particles
 * @param[in]    numCells           Number of cells in the grid
 */
extern void updateTemperature(
	int* type,
	float4* position,
	float* temperature,
	float* deltaTem,
	float* latentTem,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells);

/**
 * @brief Perform solid-fluid phase change
 *
 * @param[in/out]    type                Pointer to the array containing particle types
 * @param[in]    initId              Pointer to the array containing initial particle indices
 * @param[in]    initId2Rest         Pointer to the array containing remaining original particle indices after sorting
 * @param[in/out]    initPos             Pointer to the array containing initial particle positions
 * @param[in] position           Pointer to the array containing particle positions
 * @param[in] temperature        Pointer to the array containing particle temperatures
 * @param[in]    cellStart           Pointer to the array containing cell start indices
 * @param[in]    cellEnd             Pointer to the array containing cell end indices
 * @param[in]    gridParticleHash    Pointer to the array containing grid particle hashes
 * @param[in]    numParticles        Number of particles
 */
extern void solidFluidPhaseChange(
	int* type,
	int* initId,
	int* initId2Rest,
	float4* initPos,
	float4* position,
	float* temperature,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles);

/**
 * @brief Update velocities and positions of particles
 *
 * @param[in]    type               Pointer to the array containing particle types
 * @param[in/out] position          Pointer to the array containing particle positions
 * @param[in/out] velocity          Pointer to the array containing particle velocities
 * @param[in/out] predictedPos      Pointer to the array containing predicted particle positions
 * @param[in]    deltaTime          Time step size for the update
 * @param[in]    numParticles       Number of particles
 */
extern void updateVelAndPos(
	int* type,
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles);

/**
 * @brief Apply XSPH viscosity to particles
 *
 * @param[in]    type              Pointer to the array containing particle types
 * @param[in/out] velocity         Pointer to the array containing particle velocities
 * @param[in/out] position         Pointer to the array containing particle positions
 * @param[in]    cellStart         Pointer to the array containing cell start indices
 * @param[in]    cellEnd           Pointer to the array containing cell end indices
 * @param[in]    gridParticleHash  Pointer to the array containing grid particle hashes
 * @param[in]    numParticles      Number of particles
 */
extern void applyXSPHViscosity(
	int* type,
	float4 *velocity,
	float4 *position,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles);

/**
 * @brief Solve distance constraints between particles
 *
 * @param[in]    type              Pointer to the array containing particle types
 * @param[in]    initId            Pointer to the array containing initial particle indices
 * @param[in]    initId2Now        Pointer to the array containing current particle indices after sorting
 * @param[in/out] predictedPos     Pointer to the array containing predicted particle positions
 * @param[in]    initPos           Pointer to the array containing initial particle positions
 * @param[in/out] deltaPos         Pointer to the array containing positional deltas
 * @param[in]    cellStart         Pointer to the array containing cell start indices
 * @param[in]    cellEnd           Pointer to the array containing cell end indices
 * @param[in]    gridParticleHash  Pointer to the array containing grid particle hashes
 * @param[in]    numParticles      Number of particles
 */
void solveDistanceConstrain(
	int* type,
	int* initId,
	int* initId2Now,
	float4* predictedPos,
	float4* initPos,
	float3* deltaPos,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles);

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

void SolidLiquidPhaseChangeComponent::reset()
{
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_pos, m_host_pos.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_type, m_host_type.data(), sizeof(int) * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_tem, m_host_tem.data(), sizeof(float) * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id1, m_host_init_id.data(), sizeof(int) * m_num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id2, m_host_init_id.data(), sizeof(int) * m_num_particles, cudaMemcpyHostToDevice);
}

SolidLiquidPhaseChangeComponent::SolidLiquidPhaseChangeComponent(void* v_config)
    : m_device_pos(nullptr), m_device_vel(nullptr), m_device_tem(nullptr), m_device_type(nullptr), m_device_init_id1(nullptr), m_device_init_id2(nullptr), m_bInitialized(false)
{
    SolidLiquidPhaseChangeParams* config = static_cast<SolidLiquidPhaseChangeParams*>(v_config);

    float radius  = 0.3;
    float spacing = radius * 2.0f;
    float jitter  = radius * 0.01f;

    int num_iter = 0;

    srand(1973);
    unsigned int numParticles = 0;
    // fluid phase
    MyFloatVec3 fluidSize = { config->m_fluid_size.x, config->m_fluid_size.y, config->m_fluid_size.z};
    MyIntVec3   fluidDim  = { static_cast<int>(fluidSize.x / spacing),
                              static_cast<int>(fluidSize.y / spacing),
                              static_cast<int>(fluidSize.z / spacing) };

    for (int z = 0; z < fluidDim.z; ++z)
    {
        for (int y = 0; y < fluidDim.y; ++y)
        {
            for (int x = 0; x < fluidDim.x; ++x)
            {
                m_host_type.push_back(2);  // 0 represent boundary, 1 represent solid, 2 represent fluid

                m_host_pos.push_back(spacing * x + radius - 0.5f * config->m_fluid_locate.x);
                m_host_pos.push_back(spacing * y + radius - 0.5f * config->m_fluid_locate.y);
                m_host_pos.push_back(spacing * z + radius - 0.5f * config->m_fluid_locate.z);
                m_host_pos.push_back(1.0f);

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);

                m_host_tem.push_back(config->m_fluid_tem);

                m_host_init_id.push_back(num_iter);
                num_iter++;
            }
        }
    }

    // boundary
    MyFloatVec3 boundarySize = { config->m_boundary_size.x, config->m_boundary_size.y, config->m_boundary_size.z };
     MyIntVec3   boundaryDim  = { static_cast<int>(boundarySize.x / spacing),
                                  static_cast<int>(boundarySize.y / spacing),
                                  static_cast<int>(boundarySize.z / spacing) };

     for (int z = 0; z < boundaryDim.z; ++z)
    {
         for (int y = 0; y < boundaryDim.y; ++y)
         {
             for (int x = 0; x < boundaryDim.x; ++x)
             {
                 m_host_type.push_back(0); 

                m_host_pos.push_back(spacing * x + radius - 0.5f * config->m_boundary_locate.x);
                m_host_pos.push_back(spacing * y + radius - 0.5f * config->m_boundary_locate.y);
                m_host_pos.push_back(spacing * z + radius - 0.5f * config->m_boundary_locate.z);
                m_host_pos.push_back(1.0f);

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);

                m_host_tem.push_back(config->m_boundary_tem);

                m_host_init_id.push_back(num_iter);
                num_iter++;
            }
        }
    }

    // solid phase
    MyFloatVec3 solidSize = { config->m_solid_size.x, config->m_solid_size.y, config->m_solid_size.z };
    MyIntVec3   solidDim  = { static_cast<int>(solidSize.x / spacing),
                              static_cast<int>(solidSize.y / spacing),
                              static_cast<int>(solidSize.z / spacing) };

    for (int z = 0; z < solidDim.z; ++z)
    {
        for (int y = 0; y < solidDim.y; ++y)
        {
            for (int x = 0; x < solidDim.x; ++x)
            {
                m_host_type.push_back(1); 

                m_host_pos.push_back(spacing * x + radius - 0.5f * config->m_solid_locate.x);
                m_host_pos.push_back(spacing * y + radius - 0.5f * config->m_solid_locate.y);
                m_host_pos.push_back(spacing * z + radius - 0.5f * config->m_solid_locate.z);
                m_host_pos.push_back(1.0f);

                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);
                m_host_vel.push_back(0.0f);

                m_host_tem.push_back(config->m_solid_tem);

                m_host_init_id.push_back(num_iter);
                num_iter++;
            }
        }
    }

    numParticles = static_cast<unsigned int>(m_host_pos.size() / 4);
    initialize(numParticles);
    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_type, m_host_type.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_tem, m_host_tem.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id1, m_host_init_id.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id2, m_host_init_id.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);
}

SolidLiquidPhaseChangeComponent::SolidLiquidPhaseChangeComponent()
    : m_bInitialized(false)
{

}

void SolidLiquidPhaseChangeComponent::addInstance(float particle_radius, std::vector<float3>& init_pos, int type = static_cast<int>(SLPCPhase::FLUID), float tem = 50.f)
{
    float invmass       = 1.0f / pow(particle_radius / 0.3, 3);
    int   num_particles = static_cast<unsigned int>(init_pos.size());
    int   num_iter      = static_cast<unsigned int>(m_host_pos.size() / 4);

    for (unsigned int i = 0; i < num_particles; i++)
    {
        m_host_pos.push_back(init_pos[i].x);
        m_host_pos.push_back(init_pos[i].y);
        m_host_pos.push_back(init_pos[i].z);
        m_host_pos.push_back(invmass);

        m_host_type.push_back(static_cast<int>(type));

        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);
        m_host_vel.push_back(0.0f);

        m_host_tem.push_back(tem);
        m_host_init_id.push_back(num_iter);
        num_iter++;
    }
    m_num_particles = static_cast<unsigned int>(m_host_pos.size() / 4);
}

void SolidLiquidPhaseChangeComponent::initialize(int numParticles)
{
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    m_num_particles      = numParticles;
    unsigned int memSize = sizeof(float) * 4 * m_num_particles;

    // allocation
    cudaMalloc(( void** )&m_device_pos, memSize);
    cudaMalloc(( void** )&m_device_init_pos, memSize);
    getLastCudaError("allocation1: pos");
    cudaMalloc(( void** )&m_device_vel, memSize);
    cudaMalloc(( void** )&m_device_type, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_device_init_id_2_rest, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_device_init_id_2_now, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_device_init_id1, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_device_init_id2, sizeof(unsigned int) * m_num_particles);
    cudaMalloc(( void** )&m_device_tem, sizeof(float) * m_num_particles);
    cudaMalloc(( void** )&m_external_force, sizeof(float) *3* m_num_particles);

    getLastCudaError("allocation");    

    cudaMemcpy(m_device_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_pos, m_host_pos.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_vel.data(), sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_type, m_host_type.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_tem, m_host_tem.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id1, m_host_init_id.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_init_id2, m_host_init_id.data(), sizeof(int) * numParticles, cudaMemcpyHostToDevice);

    m_bInitialized = true;
}
void SolidLiquidPhaseChangeComponent::freeMemory()
{
    cudaFree(m_device_pos);
    getLastCudaError("m_device_pos");
    cudaFree(m_device_init_pos);
    getLastCudaError("m_device_init_pos");
    cudaFree(m_device_vel);
    getLastCudaError("m_device_vel");
    cudaFree(m_device_type);
    getLastCudaError("m_device_type");
    cudaFree(m_device_init_id_2_rest);
    getLastCudaError("m_device_init_id_2_rest");
    cudaFree(m_device_init_id_2_now);
    getLastCudaError("m_device_init_id_2_now");
    cudaFree(m_device_init_id1);
    getLastCudaError("m_device_init_id1");
    cudaFree(m_device_init_id2);
    getLastCudaError("m_device_init_id2");
    cudaFree(m_device_tem);
    getLastCudaError("m_device_tem");
    cudaFree(m_external_force);
    getLastCudaError("m_external_force");
}
SolidLiquidPhaseChangeComponent::~SolidLiquidPhaseChangeComponent()
{
    // free gpu memory
    if (m_bInitialized)
    {
        m_bInitialized = false;
    }
}

SolidLiquidPhaseChangeSolver::SolidLiquidPhaseChangeSolver()
    : m_is_init(false), m_solid_fluid_phase_change(nullptr), m_cur_time(0), m_host_pos(nullptr), m_host_tem(nullptr), m_host_type(nullptr), m_device_cell_end(nullptr), m_device_cell_start(nullptr), m_device_delta_pos(nullptr), m_device_grid_particle_hash(nullptr), m_device_predicted_pos(nullptr)
{
    m_params = new SolidLiquidPhaseChangeParams();
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

SolidLiquidPhaseChangeSolver::~SolidLiquidPhaseChangeSolver()
{
    if (m_is_init)
        _finalize();
}

bool SolidLiquidPhaseChangeSolver::initialize()
{
    if (m_is_init)
        return true;
    if (m_solid_fluid_phase_change == nullptr)
    {
        std::cout << "ERROR: Must set solid fluid phase change particle object first.\n";
        return false;
    }
    if (m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>() == false)
    {
        std::cout << "ERROR: solid_fluid_phase_change object has no SolidLiquidPhaseChange component.\n";
        return false;
    }
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_liquid_phase_change_component == nullptr)
    {
        std::cout << "ERROR: no SolidLiquidPhaseChange component.\n";
        return false;
    }
    mallocParticleMemory(m_params->m_num_particles);

    m_is_init = true;
    std::cout << "SolidLiquidPhaseChange solver initialized successfully.\n";
    return true;
}

bool SolidLiquidPhaseChangeSolver::isInitialized() const
{
    return m_is_init;
}

bool SolidLiquidPhaseChangeSolver::reset()
{
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_solid_fluid_phase_change   = nullptr;
    m_cur_time            = 0.0;
    if (m_solid_fluid_phase_change != nullptr)
    {
        m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>()->reset();
    }
    return true;
}


bool SolidLiquidPhaseChangeSolver::step()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return false;
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();

    if (solid_liquid_phase_change_component == nullptr)
        return false;

    setParameters(m_params);


    // advect
    {
        fluidAdvection(
            solid_liquid_phase_change_component->m_device_type,
            ( float4* )solid_liquid_phase_change_component->m_device_pos,
            solid_liquid_phase_change_component->m_device_tem,
            ( float4* )solid_liquid_phase_change_component->m_device_vel,
            ( float4* )m_device_predicted_pos,
            static_cast<float>(m_config.m_dt),
            solid_liquid_phase_change_component->m_num_particles,
            solid_liquid_phase_change_component->m_external_force);

        getLastCudaError("Advection");
    }

    // find neighbours
    {
        // calculate grid Hash.
        computeHash(
            m_device_grid_particle_hash,
            m_device_predicted_pos,
            solid_liquid_phase_change_component->m_num_particles);
        getLastCudaError("computeHash");

        // sort all particles based on hash value.
        sortParticlesAll(
            m_device_grid_particle_hash,
            solid_liquid_phase_change_component->m_num_particles,
            solid_liquid_phase_change_component->m_device_pos,
            solid_liquid_phase_change_component->m_device_tem,
            solid_liquid_phase_change_component->m_device_vel,            
            solid_liquid_phase_change_component->m_device_type,
            solid_liquid_phase_change_component->m_device_init_id2,
            m_device_predicted_pos);
        getLastCudaError("sortParticles");

        // find start index and end index of each cell.
        findCellRange(
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            solid_liquid_phase_change_component->m_num_particles,
            m_num_grid_cells);
        getLastCudaError("findCellRange");
    }

    //neighbor search for solid
		{
			computeHash(
				m_device_grid_particle_hash_solid,
				solid_liquid_phase_change_component->m_device_init_pos,
				solid_liquid_phase_change_component->m_num_particles
			);
			getLastCudaError("computeHash");


			// sort particles based on hash value of rest position.
			sortParticlesForSolid(
				m_device_grid_particle_hash_solid,
				solid_liquid_phase_change_component->m_num_particles,
				solid_liquid_phase_change_component->m_device_init_pos,
				solid_liquid_phase_change_component->m_device_init_id1
			);
			getLastCudaError("sortParticles_solid");

			// find start index and end index of each cell.
			findCellRange(
				m_device_cell_start_solid,
				m_device_cell_end_solid,
				m_device_grid_particle_hash_solid,
				solid_liquid_phase_change_component->m_num_particles,
				m_num_grid_cells);
			getLastCudaError("findCellRange_solid");
		}
        //make sure the sorted particle based two position array can correspond 
		setInitIdToNow(
			solid_liquid_phase_change_component->m_device_init_id1,
			solid_liquid_phase_change_component->m_device_init_id2,
			solid_liquid_phase_change_component->m_device_init_id_2_now,
			solid_liquid_phase_change_component->m_device_init_id_2_rest,
			solid_liquid_phase_change_component->m_num_particles
		);


    // density constraint.
    {
        unsigned int iter = 0;
        while (iter < m_params->m_max_iter_nums)
        {
            densityConstraint(
                solid_liquid_phase_change_component->m_device_type,
                ( float4* )(solid_liquid_phase_change_component->m_device_pos),
                ( float4* )solid_liquid_phase_change_component->m_device_vel,
                solid_liquid_phase_change_component->m_device_tem,	
                ( float3* )m_device_delta_pos,
                ( float4* )m_device_predicted_pos,
                m_device_cell_start,
                m_device_cell_end,
                m_device_grid_particle_hash,
                solid_liquid_phase_change_component->m_num_particles,
                m_num_grid_cells);
            ++iter;
        }
    }
    // distance constraint.
    {
        solveDistanceConstrain(
        solid_liquid_phase_change_component->m_device_type,
        solid_liquid_phase_change_component->m_device_init_id1,
        solid_liquid_phase_change_component->m_device_init_id_2_now,
        (float4*)m_device_predicted_pos,
        (float4*)solid_liquid_phase_change_component->m_device_init_pos,
        (float3*)m_device_delta_pos,			
        m_device_cell_start_solid,
        m_device_cell_end_solid,
        m_device_grid_particle_hash_solid,
        solid_liquid_phase_change_component->m_num_particles);
    }

    // update velocity and position
    {
        updateVelAndPos(
            solid_liquid_phase_change_component->m_device_type,
            ( float4* )solid_liquid_phase_change_component->m_device_pos,
            ( float4* )solid_liquid_phase_change_component->m_device_vel,
            ( float4* )m_device_predicted_pos,
            static_cast<float>(m_config.m_dt),
            solid_liquid_phase_change_component->m_num_particles
            );
    }

    // update Temperature.
    {        
        updateTemperature(
            solid_liquid_phase_change_component->m_device_type,
            ( float4* )solid_liquid_phase_change_component->m_device_pos,            
            solid_liquid_phase_change_component->m_device_tem,		
            m_device_delta_tem,
            m_device_latent_tem,
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            solid_liquid_phase_change_component->m_num_particles,
            m_num_grid_cells);
        
    }

    //solid-fluid phase change
    {
        solidFluidPhaseChange(
            solid_liquid_phase_change_component->m_device_type,
            solid_liquid_phase_change_component->m_device_init_id2,
            solid_liquid_phase_change_component->m_device_init_id_2_rest,
           ( float4* )solid_liquid_phase_change_component->m_device_init_pos,
            ( float4* )solid_liquid_phase_change_component->m_device_pos,
            solid_liquid_phase_change_component->m_device_tem,
            m_device_cell_start,
            m_device_cell_end,
            m_device_grid_particle_hash,
            solid_liquid_phase_change_component->m_num_particles);
    }

    return true;
}

bool SolidLiquidPhaseChangeSolver::run()
{
    if (!m_is_init)
    {
        return false;
    }
    if (!m_solid_fluid_phase_change)
    {
        return false;
    }
    // Update till termination

    float sim_run_time = 0.0;
    int   step_id      = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        
        m_cur_time += dt;
        m_config.m_dt = dt;

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
        
        if(m_params->m_write_ply)
            writeToPly(step_id);
        if (m_params->m_write_statistics)
            writeToStatistics(step_id, elapsedTime);
    }
    std::cout << "Run Time: " << sim_run_time << " ms" << std::endl;
    return true;
}

bool SolidLiquidPhaseChangeSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<SolidLiquidPhaseChangeComponent>();
}

bool SolidLiquidPhaseChangeSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<SolidLiquidPhaseChangeComponent>())
    {
        std::cout << "object attached as solid fluid phase change system.\n";
        m_solid_fluid_phase_change                   = object;
        SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
        m_params->m_num_particles             = solid_liquid_phase_change_component->m_num_particles;
        m_host_pos                            = new float[m_params->m_num_particles * 4];
    }
    initialize();
    return true;
}

bool SolidLiquidPhaseChangeSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_solid_fluid_phase_change == object)
    {
        m_solid_fluid_phase_change = nullptr;
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void SolidLiquidPhaseChangeSolver::clearAttachment()
{
    m_solid_fluid_phase_change = nullptr;
}

void SolidLiquidPhaseChangeSolver::config(SolverConfig& config)
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
    m_params->m_is_convect      = config.m_is_convect;
    m_params->m_world_size      = config.m_world_size;
    m_params->m_fluid_size      = config.m_fluid_size;
    m_params->m_fluid_locate    = config.m_fluid_locate;
    m_params->m_fluid_tem       = config.m_fluid_tem;
    m_params->m_solid_size      = config.m_solid_size;
    m_params->m_solid_locate    = config.m_solid_locate;
    m_params->m_solid_tem       = config.m_solid_tem;
    m_params->m_boundary_size   = config.m_boundary_size;
    m_params->m_boundary_locate = config.m_boundary_locate;
    m_params->m_boundary_tem    = config.m_boundary_tem;
    m_params->m_write_ply        = config.m_write_ply;
    m_params->m_write_statistics = config.m_write_statistics;
    m_params->m_radiate          = config.m_radiate;
    m_params->m_melt_tem         = 20.f;
    m_params->m_solidify_tem     = 20.f;
}

void SolidLiquidPhaseChangeSolver::_finalize()
{
    if (m_host_pos != nullptr)
        delete m_host_pos;
    if (m_solid_fluid_phase_change != nullptr)
    {
        if (m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
            m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>()->freeMemory();
    }
    freeParticleMemory(m_params->m_num_particles);
    if (m_params != nullptr)
        delete m_params;
}

bool SolidLiquidPhaseChangeSolver::setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz)
{
    if (!m_is_init)
        return false;
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return false;
    SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_fluid_phase_change_component == nullptr)
        return false;

    m_params->m_world_box_corner1 = make_float3(lx, ly, lz);
    m_params->m_world_box_corner2 = make_float3(ux, uy, uz);

    return true;
}

void SolidLiquidPhaseChangeSolver::getWorldBoundary(float& lx, float& ly, float& lz, float& ux, float& uy, float& uz)
{
    lx = m_params->m_world_box_corner1.x;
    ly = m_params->m_world_box_corner1.y;
    lz = m_params->m_world_box_corner1.z;
    ux = m_params->m_world_box_corner2.x;
    uy = m_params->m_world_box_corner2.y;
    uz = m_params->m_world_box_corner2.z;
}


bool SolidLiquidPhaseChangeSolver::setParticlePosition(const std::vector<float>& position)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return false;
    SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_fluid_phase_change_component == nullptr)
        return false;

    // free particle component and solver data on device.
    if (solid_fluid_phase_change_component->m_bInitialized)
    {
        solid_fluid_phase_change_component->m_bInitialized = false;
        freeParticleMemory(position.size() / 4);
    }
    m_num_particles                  = static_cast<unsigned int>(position.size() / 4);
    solid_fluid_phase_change_component->m_num_particles = m_num_particles;

    // malloc particle component data on device
    size_t mem_size = solid_fluid_phase_change_component->m_num_particles * 4 * sizeof(float);
    solid_fluid_phase_change_component->initialize(m_num_particles);
    cudaMemcpy(solid_fluid_phase_change_component->m_device_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemset(solid_fluid_phase_change_component->m_device_type, 0, solid_fluid_phase_change_component->m_num_particles *  sizeof(int));
    cudaMemset(solid_fluid_phase_change_component->m_device_vel, 0, mem_size);
    solid_fluid_phase_change_component->m_bInitialized = true;

    // malloc solver data on device
    mallocParticleMemory(m_num_particles);
    return true;
}

bool SolidLiquidPhaseChangeSolver::setParticleVelocity(const std::vector<float>& velocity)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return false;
    SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_fluid_phase_change_component == nullptr)
        return false;

    if (static_cast<unsigned int>(velocity.size() / 4) != solid_fluid_phase_change_component->m_num_particles)
        return false;

    cudaMemcpy(solid_fluid_phase_change_component->m_device_pos, velocity.data(), velocity.size() * sizeof(float), cudaMemcpyHostToDevice);

    solid_fluid_phase_change_component->m_bInitialized = true;
    return true;
}

bool SolidLiquidPhaseChangeSolver::setParticlePhase(const std::vector<int>& particlePhase)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
    {
        SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
        if (solid_fluid_phase_change_component != nullptr)
        {
            if (solid_fluid_phase_change_component->m_bInitialized)
            {
                if (particlePhase.size() / 4 == solid_fluid_phase_change_component->m_num_particles)
                {
                    cudaMemcpy(solid_fluid_phase_change_component->m_device_type, particlePhase.data(), particlePhase.size() * sizeof(int), cudaMemcpyHostToDevice);
                    return true;
                }
            }
        }
    }
    return false;
}

bool SolidLiquidPhaseChangeSolver::setParticleExternalForce(float3* external_force)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
    {
        SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
        if (solid_fluid_phase_change_component != nullptr)
        {
            solid_fluid_phase_change_component->m_external_force = external_force;
            return true;
        }
    }
    return false;
}

float* SolidLiquidPhaseChangeSolver::getParticlePositionPtr()
{
    if (m_solid_fluid_phase_change != nullptr)
    {
        if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        {
            SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
            if (solid_fluid_phase_change_component != nullptr)
                return solid_fluid_phase_change_component->m_device_pos;
        }
    }
    return nullptr;
}

float* SolidLiquidPhaseChangeSolver::getParticleVelocityPtr()
{
    if (m_solid_fluid_phase_change != nullptr)
    {
        if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        {
            SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
            if (solid_fluid_phase_change_component != nullptr)
                return solid_fluid_phase_change_component->m_device_vel;
        }
    }
    return nullptr;
}

int* SolidLiquidPhaseChangeSolver::getParticlePhasePtr()
{
    if (m_solid_fluid_phase_change != nullptr)
    {
        if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        {
            SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
            if (solid_fluid_phase_change_component != nullptr)
                return solid_fluid_phase_change_component->m_device_type;
        }
    }
    return nullptr;
}

float3* SolidLiquidPhaseChangeSolver::getParticleExternalForcePtr()
{
    if (m_solid_fluid_phase_change != nullptr)
    {
        if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        {
            SolidLiquidPhaseChangeComponent* solid_fluid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
            if (solid_fluid_phase_change_component != nullptr)
                return solid_fluid_phase_change_component->m_external_force;
        }
    }
    return nullptr;
}

void SolidLiquidPhaseChangeSolver::getParticleRadius(float& particleRadius)
{
    if (!m_is_init)
        particleRadius = 0;
    particleRadius = m_params->m_particle_radius;
}


void SolidLiquidPhaseChangeSolver::setParam(const std::string& paramName, void* value)
{
    if (paramName == "m_is_convect")
    {
        m_params->m_is_convect = *static_cast<bool*>(value);
    }
    else if (paramName == "m_world_size")
    {
        m_params->m_world_size = *static_cast<float3*>(value);
    }
    else if (paramName == "m_radiate")
    {
        m_params->m_radiate = *static_cast<bool*>(value);
    }
    else if (paramName == "m_melt_tem")
    {
        m_params->m_melt_tem = *static_cast<float*>(value);
    }
    else if (paramName == "m_solidify_tem")
    {
        m_params->m_solidify_tem = *static_cast<float*>(value);
    }
    else
    {
        printf("set error: Parameter %s not found.\n", paramName.c_str());
    }
}

void* SolidLiquidPhaseChangeSolver::getParam(const std::string& paramName)
{
    if (paramName == "m_is_convect")
    {
        return &m_params->m_is_convect;
    }
    else if (paramName == "m_world_size")
    {
        return &m_params->m_world_size;
    }
    else if (paramName == "m_radiate")
    {
        return &m_params->m_radiate;
    }
    else if (paramName == "m_melt_tem")
    {
        return &m_params->m_melt_tem;
    }
    else if (paramName == "m_solidify_tem")
    {
        return &m_params->m_solidify_tem;
    }
    else
    {
        printf("get error: Parameter %s not found.\n", paramName.c_str());
        return nullptr;
    }
}


void SolidLiquidPhaseChangeSolver::writeToPly(const int& step_id)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return;
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_liquid_phase_change_component == nullptr)
        return;
    cudaMemcpy(m_host_pos, solid_liquid_phase_change_component->m_device_pos, sizeof(float4) * solid_liquid_phase_change_component->m_num_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_host_tem, solid_liquid_phase_change_component->m_device_tem, sizeof(float) * solid_liquid_phase_change_component->m_num_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_host_type, solid_liquid_phase_change_component->m_device_type, sizeof(int) * solid_liquid_phase_change_component->m_num_particles, cudaMemcpyDeviceToHost);
    //  write to ply file
    std::string filename = "./ply/solid_liquid_phase_change_" + std::to_string(step_id) + ".ply";
    std::cout << "write to ply: " << filename << std::endl;
    // write pos ti ply file
    std::ofstream outfile(filename);
    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << solid_liquid_phase_change_component->m_num_particles << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property float tem\n";
    outfile << "property int type\n";
    outfile << "end_header\n";
    for (unsigned int i = 0; i < solid_liquid_phase_change_component->m_num_particles * 4; i += 4)
    {
        outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " " << m_host_tem[i / 4] << " " << m_host_type[i / 4] << "\n";
    }
    outfile.close();
}

void SolidLiquidPhaseChangeSolver::writeToStatistics(const int& step_id, const float& frame_time)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return;
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_liquid_phase_change_component == nullptr)
        return;
    //  write to ply file
    std::string filename = "./Statistics.txt";
    // write pos ti ply file
    if (step_id == 1)
    {
        std::ofstream outfile(filename);
        outfile << "frame_num: " << step_id << " time: " << frame_time << " ms\n";
        outfile.close();
    }        
    else
    {
        std::ofstream outfile(filename, std::ios_base::app); 
        outfile << "frame_num: " << step_id << " time: " << frame_time << " ms\n";
        outfile.close();
    }
}

void SolidLiquidPhaseChangeSolver::freeParticleMemory(const int& numParticles)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return;
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_liquid_phase_change_component == nullptr)
        return;
    solid_liquid_phase_change_component->~SolidLiquidPhaseChangeComponent();

    cudaFree(m_device_predicted_pos);
    getLastCudaError("m_device_predicted_pos");
    cudaFree(m_device_delta_pos);
    getLastCudaError("m_device_delta_pos");
    cudaFree(m_device_delta_tem);
    getLastCudaError("m_device_delta_tem");
    cudaFree(m_device_latent_tem);
    getLastCudaError("m_device_latent_tem");
    cudaFree(m_device_grid_particle_hash);
    getLastCudaError("m_device_grid_particle_hash");
    cudaFree(m_device_cell_start);
    getLastCudaError("m_device_cell_start");
    cudaFree(m_device_cell_end);
    getLastCudaError("m_device_cell_end");
    cudaFree(m_device_grid_particle_hash_solid);
    getLastCudaError("m_device_grid_particle_hash_solid");
    cudaFree(m_device_cell_start_solid);
    getLastCudaError("m_device_cell_start_solid");
    cudaFree(m_device_cell_end_solid);
    getLastCudaError("m_device_cell_end_solid");

    if (m_host_pos != nullptr)
        delete m_host_pos;
    if (m_host_tem != nullptr)
        delete m_host_tem;
    if (m_host_type != nullptr)
        delete m_host_type;
}

void SolidLiquidPhaseChangeSolver::mallocParticleMemory(const int& numParticles)
{
    if (!m_solid_fluid_phase_change->hasComponent<SolidLiquidPhaseChangeComponent>())
        return;
    SolidLiquidPhaseChangeComponent* solid_liquid_phase_change_component = m_solid_fluid_phase_change->getComponent<SolidLiquidPhaseChangeComponent>();
    if (solid_liquid_phase_change_component == nullptr)
        return;

    size_t memSize = numParticles * sizeof(float) * 4;
    solid_liquid_phase_change_component->initialize(numParticles);
    cudaMalloc(( void** )&m_device_predicted_pos, memSize);
    
    cudaMalloc(( void** )&m_device_delta_pos, sizeof(float) * 3 * numParticles);
    cudaMalloc(( void** )&m_device_delta_tem, sizeof(float) * numParticles);
    cudaMalloc(( void** )&m_device_latent_tem, sizeof(float) * numParticles);
    cudaMalloc(( void** )&m_device_grid_particle_hash, numParticles * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_start, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_end, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_grid_particle_hash_solid, numParticles * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_start_solid, m_num_grid_cells * sizeof(unsigned int));
    cudaMalloc(( void** )&m_device_cell_end_solid, m_num_grid_cells * sizeof(unsigned int));

    if (m_host_pos == nullptr)
        m_host_pos = new float[numParticles * 4];
    if (m_host_tem == nullptr)
        m_host_tem = new float[numParticles];
    if (m_host_type == nullptr)
        m_host_type = new int[numParticles];
}

}  // namespace Physika