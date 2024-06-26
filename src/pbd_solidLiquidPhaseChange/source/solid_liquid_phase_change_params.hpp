/**
 * @file       : solid_liquid_phase_change_params.hpp
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @description: This file declares the parameters required for simulating solid-liquid phase change in a granular system,
 *               embodied in the struct 'SolidLiquidPhaseChangeParams'. These parameters encompass various physical and
 *               computational aspects of the phase change simulation, such as initial particle indices, positions,
 *               temperatures, cell start and end indices, grid particle hashes, and the number of particles involved.
 *
 *               The parameters in this struct are used to govern the behavior of the solid-liquid phase change process and
 *               are typically calculated on the CPU before being utilized in the GPU-based simulation code. They provide
 *               essential information for handling the phase change phenomena within the granular system.
 *
 * @version    : 1.0
 * @note       : The parameters are intended to be used within the Physika namespace.
 * @dependencies: This file may depend on external libraries or CUDA data types for the data structures used in the
 *               SolidLiquidPhaseChangeParams struct.
 */

#ifndef SOLID_LIQUID_PHASE_CHANGE_PARAMS_HPP
#define SOLID_LIQUID_PHASE_CHANGE_PARAMS_HPP

#include <vector_types.h>

namespace Physika {
/**
 * @brief the SolidLiquidPhaseChange system parameters
 *
 * Most parameters are not allowed to be modified from the outside.
 * Some parameters are calculated on CPU and copied to GPU to speed up the simulation (eg. m_poly6Coff, m_spikyGradCoff, m_sphRadius, m_sphRadiusSquared).
 *
 * Some parameters related to visual performance can be set through the set...() function inside the GranularSolver class.
 *
 * Users are not allowed to directly define this structure and modify it. This is not an interface data.
 *
 */
struct SolidLiquidPhaseChangeParams
{
    uint3        m_grid_size;                // The size of the spatial hash grid used for particle lookup
    float3       m_gravity;                  // The acceleration due to gravity in the simulation
    float3       m_cell_size;                // The size of each cell in the spatial hash grid
    float3       m_world_origin;             // The origin of the simulation world
    float3       m_world_box_corner1;        // The coordinates of the lower corner of the world bounding box
    float3       m_world_box_corner2;        // The coordinates of the upper corner of the world bounding box
    float        m_poly6_coff;               // Coefficient in the poly6 kernel function used in SPH simulations
    float        m_spiky_grad_coff;          // Coefficient in the gradient of the spiky kernel function used in SPH simulations
    float        m_sph_radius;               // The interaction radius for the SPH simulations
    float        m_sph_radius_squared;       // Square of the interaction radius for the SPH simulations
    float        m_lambda_eps;               // Small constant used in density constraint computation for numerical stability
    float        m_rest_density;             // The rest density of the particle material in the simulation
    float        m_inv_rest_density;         // Inverse of the rest density, used for computations to avoid division
    float        m_particle_radius;          // The radius of each particle in the simulation
    float        m_one_div_wPoly6;           // Pre-computed division result for performance optimization
    float        m_stiffness;                // Stiffness constant for the pressure forces in the SPH simulation
    float        m_static_fricton_coeff;     // Coefficient of static friction, used in friction constraint
    float        m_dynamic_fricton_coeff;    // Coefficient of dynamic friction, used in friction constraint
    float        m_stack_height_coeff;       // Stacking height coefficient, used to limit stacking height of particles
    float        m_static_frict_threshold;   // Static friction threshold
    float        m_dynamic_frict_threshold;  // Dynamic friction threshold
    float        m_sleep_threshold;          // Velocity threshold below which particles are considered as 'sleeping'
    unsigned int m_max_iter_nums;            // Maximum number of iterations for the constraint solver
    unsigned int m_num_grid_cells;           // Total number of grid cells in the hash grid
    unsigned int m_num_particles;            // Total number of particles in the simulation

    bool   m_is_convect;              // Whether to enable convection in the simulation
    float3 m_world_size;        // Size of the simulation world
    float3 m_fluid_size;       // The size of the fluid region in the simulation
    float3 m_fluid_locate;     // The location of the fluid region in the simulation
    float  m_fluid_tem;        // The temperature of the fluid region in the simulation
    float3 m_solid_size;       // The size of the solid region in the simulation
    float3 m_solid_locate;     // The location of the solid region in the simulation
    float  m_solid_tem;        // The temperature of the solid region in the simulation
    float3 m_boundary_size;    // The size of the boundary region in the simulation
    float3 m_boundary_locate;  // The location of the boundary region in the simulation
    float  m_boundary_tem;     // The temperature of the boundary region in the simulation
    bool   m_write_ply;        // Whether to write ply file
    bool   m_write_statistics; // Whether to write statistics txt file
    bool   m_radiate;           // Whether to enable radiate
    float  m_melt_tem;          //melting point
    float  m_solidify_tem;        //solidifying point
};

}  // namespace Physika

#endif  // !SOLID_LIQUID_PHASE_CHANGE_PARAMS_HPP
