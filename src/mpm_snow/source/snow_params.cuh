/**
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-04-11
 * @description: Snow Simulation parameters
 * @version    : 1.0
 *
 * @file       snow_simulation_params.h
 * @brief      Defines data structures and parameters for snow simulations
 *
 * This file contains the main data structures used during snow simulation, including the definition of particles and grid cells,
 * and parameters used to control the simulation process. These data structures and parameters are at the heart of physical simulations and are used to
 * Describe and control the physical behavior of snow, such as elasticity, plastic deformation, collision handling, etc.
 *
 * @dependencies: This file relies on the CUDA runtime library for GPU-accelerated data processing and on
 *               "mat3.hpp" to provide support for 3x3 matrices, which is useful for handling physical transformations and
 *               Mechanical calculations are crucial.
 *
 * @note       : This file is part of a physics simulation project, specifically for the Particle-Grid Method (PIC)
 *               Snow simulation. It contains the definition of particles and grid cells, as well as a series of simulation parameters,
 *               These parameters allow users to customize the simulation process, such as time steps, material properties, etc.
 *
 * @remark     : The performance and accuracy of this module are crucial to the entire physical simulation process. It is recommended to modify
 *               Exercise caution and ensure all changes are fully tested.
 */
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "mpm_snow/source/include/mat3.hpp"

namespace Physika {

// This is the data structure for a single particle(PIC Particles)
struct Point
{
    float3 m_position;  // position, data layout: x, y, z
    float3 m_velocity;  // velocity, data layout: x, y, z
    float  m_mass;      // mass
    float  m_volume;    // volume
    mat3   fe;          // elastic deformation gradient
    mat3   fp;          // plastic deformation gradient

    Point(float x = 0, float y = 0, float z = 0, float vx = 0, float vy = 0, float vz = 0 )
        : m_position(make_float3(x, y, z)), m_velocity(make_float3(vx, vy, vz)), m_mass(1.0), m_volume(0), fe(mat3(1.0)), fp(mat3(1.0))
    {
    }
};

// This is the data structure for a single grid (PIC Grid)
struct Grid
{
    float  mass;           // grid mass. (used for transfer mass from particle to grid)
    float3 velocity;       // grid velocity
    float3 velocity_star;  // grid velocity star (new velocity).
    float3 force;          // grid force

    Grid()
        : mass(0.0), velocity(make_float3(0, 0, 0)), velocity_star(make_float3(0, 0, 0)), force(make_float3(0, 0, 0))
    {
    }
};

// parameters for snow simualtion solver
struct SolverParam
{
    float alpha;      // FLIP aplha 0.99 FLIP
    float dt;         // time step
    float young;      // Young's modulus
    float poisson;    // Poisson's ratio
    float hardening;  // hardening coefficient

    float lambda;  // Lame's first parameter
    float mu;      // Lame's second parameter

    float compression;  // compression limit for snow
    float stretch;      // stretch limit for snow

    // If we have 100 x 100 x100 cells, and the world box is 0 - 1 m. so we have cellSize = 1.f / 100.f; gridSize = (100, 100, 100)
    float cellSize;  // the length per cell
    int3  gridSize;  // the index of the grid size

    float3 gravity;
    // from boxCorner1 to boxCorner2  for example: (0,0,0) - (1,1,1)
    float3 boxCorner1;  // the world box corner 1
    float3 boxCorner2;  // the world box corner 2

    float3 particleCorner1;
    float3 particleCorner2;

    // particle parameters
    float p_vol;   // particle volume
    float p_mass;  // particle mass

    bool  stickyWalls;    // Sticky behavior performance switch
    float frictionCoeff;  // On the premise of allowing viscosity, the friction effect coefficient with the contact surface
};

}  // namespace Physika