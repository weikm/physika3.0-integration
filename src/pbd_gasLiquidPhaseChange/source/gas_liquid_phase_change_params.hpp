/**
 * @author     : Wang Qianwei (729596003@qq.com)
 * @date       : 2023-11-07
 * @description: Gas-Liquid phase change Simulation parameters
 * @version    : 1.0
 *
 * @file       snow_simulation_params.h
 * @brief      Defines data structures and parameters for snow simulations
 *
 * This file contains the main parameters.
 *
 * @note       : It contains the definition of particles and a series of simulation parameters,
 *               These parameters allow users to customize the simulation process, such as time steps, material properties, etc.
 *
 * @remark     : The performance and accuracy of this module are crucial to the entire physical simulation process. It is recommended to modify
 *               Exercise caution and ensure all changes are fully tested.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace Physika {
// parameters for solver
namespace GLPhaseChange_Example {
enum class Example_type
{
    Convection,
    Bubble,
    Condensation
};
};
struct GasLiquidPhaseChangeParams
{
    float3                              space_size{ make_float3(1.1f, 1.0f, 1.1f) };                                                                                                                                      
    float                               sph_radius{ 0.01f };                                                                                                                // Size of particle
    float                               sph_spacing{ 0.02f };                                                                                                               // Size of particle layout spacing
    float                               sph_smoothing_radius{ 2.0f * sph_spacing };                                                                                         // Smooth length of sph Kernels
    float                               sph_cell_size{ 1.01f * sph_smoothing_radius };                                                                                      // Cell size for neighbor searching
    int3                                cell_size = make_int3(ceil(space_size.x / sph_cell_size), ceil(space_size.y / sph_cell_size), ceil(space_size.z / sph_cell_size));  // cells size
    const float                         rest_rho{ 1.0f };                                                                                                                   // rest density of liquid
    const float                         rest_rho_boundary{ 1.4f * rest_rho };                                                                                               // rest density of boundary
    float                               rest_mass{ 76.596750762082e-6f };                                                                                                   // rest mass of liquid
    const float                         sph_stiff{ 10.0f };                                                                                                                 // stiffness constant for solver
    const float3                        sph_g{ make_float3(0.0f, -9.8f, 0.0f) };                                                                                            // gravity
    const float                         sph_visc{ 5e-4f };                                                                                                                  // viscosity for liquid
    const float                         sph_surface_tension_intensity{ 0.0001f };                                                                                           // surface tension
    const float                         sph_air_pressure{ 0.0001f };                                                                                                        // reference air pressure
    const float                         m_latenHeat{2500.0f};                                                                                                               // latent heat for gas-liquid phase-change
    const float                         m_boilingPoint{100.0f};                                                                                                               // boiling point temperature for fluid

    float                               dt{ 0.003 };                                                                                                                        // time step
    float3                              m_world_size;                                                                                                                       // Size of the simulation world
    float                               m_fluid_tem{ 95.0f };                                                                                                               // Temperature of the fluid for initializing
    float                               m_heaten_tem{ 140.0f };                                                                                                            // Temperature of the heaten particles for initializing. 2500 is recommended.
    GLPhaseChange_Example::Example_type m_test_type{ GLPhaseChange_Example::Example_type::Bubble };
    
    /**
     * @brief set particle and seaching grid sampling
     *
     * @param[in] particle sampling radius
     */
    void setParticleSize(float radius);

    /**
     * @brief set particle and seaching grid sampling
     *
     * @param[in] world space size
     */
    void setSpaceSize(float3 size);

};
void GasLiquidPhaseChangeParams::setParticleSize(float radius)
{
    sph_radius = radius;
    sph_spacing = 2.0f * radius;
    sph_smoothing_radius = 2.0f * sph_spacing;
    sph_cell_size        = 1.01f * sph_smoothing_radius;
    cell_size            = make_int3(ceil(space_size.x / sph_cell_size), ceil(space_size.y / sph_cell_size), ceil(space_size.z / sph_cell_size));
    rest_mass            = sph_radius * sph_radius * sph_radius * 80.0f * rest_rho;
}
void GasLiquidPhaseChangeParams::setSpaceSize(float3 size)
{
    space_size = size;
    cell_size            = make_int3(ceil(space_size.x / sph_cell_size), ceil(space_size.y / sph_cell_size), ceil(space_size.z / sph_cell_size));
}

}  // namespace Physika