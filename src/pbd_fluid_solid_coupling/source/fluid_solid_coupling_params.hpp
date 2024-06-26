/**
 * @file       : fluid_solid_coupling_params.hpp
 * @author     : Yuhang Xu (mr.xuyh@qq.com)
 * @date       : 2023-08-17
 * @description: This file declares the parameters required for the simulation of fluid solid coupling systems, embodied in the
 *               struct 'SimulateParams'. These parameters encompass various physical and computational aspects of the
 *               simulation, such as hash grid size, gravitational effect, boundaries of the world, coefficients for
 *               simulation computations, and details regarding particles like their density and radius.
 *
 *               Most of the parameters in this struct are intended to be immutable once initialized, with a few
 *               exceptions that can be modified through the 'set...' functions of the 'FluidSolidSolver' class.
 *
 *               The parameters are used extensively throughout the GPU-based fluid solid coupling simulation code, serving as
 *               fundamental building blocks of the simulation. They are typically calculated on the CPU and then
 *               transferred to the GPU to accelerate the simulation.
 *
 * @version    : 1.0
 * @note       : The parameters are used in the Physika namespace.
 * @dependencies: This file depends on vector_types.h for the CUDA data types used in the SimulateParams struct.
 */

#ifndef Fluid_Solid_PARAMS_HPP
#define Fluid_Solid_PARAMS_HPP

#include <vector_types.h>

namespace Physika {
/**
 * @brief the fluid solid coupling system parameters
 *
 * Most parameters are not allowed to be modified from the outside.
 * Some parameters are calculated on CPU and copied to GPU to speed up the simulation (eg. m_poly6Coff, m_spikyGradCoff, m_sphRadius, m_sphRadiusSquared).
 *
 * Some parameters related to visual performance can be set through the set...() function inside the FluidSolidSolver class.
 *
 * Users are not allowed to directly define this structure and modify it. This is not an interface data.
 *
 */
struct FluidSolidCouplingParams
{
    uint3        m_grid_size;                // The size of the spatial hash grid used for particle lookup
    float3       m_gravity;                  // The acceleration due to gravity in the simulation
    float3       m_cell_size;                // The size of each cell in the spatial hash grid
    float3       m_world_origin;             // The origin of the simulation world
    float3       m_world_box_corner1;        // The coordinates of the lower corner of the world bounding box
    float3       m_world_box_corner2;        // The coordinates of the upper corner of the world bounding box
    float        m_damp;                     // Coefficient for damping the normal velocity to reduce particle oscillations
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
};

}  // namespace Physika

#endif  // !Fluid_Solid_PARAMS_HPP
