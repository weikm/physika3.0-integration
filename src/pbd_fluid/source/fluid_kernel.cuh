/**
 * @file       : fluid_kernel.hpp
 * @author     : Yuege Xiong (candybear0714@163.com)
 * @date       : 2023-11-22
 * @description: This file contains the declaration of CUDA functions for the fluid system. The main functionality
 *               of this file revolves around the GPU accelerated physics simulation of fluid systems. It includes
 *               definitions for setting up and managing simulation parameters, performing calculations related to the
 *               fluid system like hashing and sorting of particles, particle advection, updating velocities and
 *               positions, handling surface tension and adhesion force, and other related operations.
 *
 *               This file interfaces with the 'pbd_fluid' module and specifically uses the fluid_params.hpp for
 *               the parameters required in the simulation. The CUDA kernels defined in this file are expected to be
 *               called by the host-side (CPU) code.
 *
 * @dependencies: This file depends on CUDA runtime headers and the fluid_params.hpp from the 'pbd_fluid' module.
 *
 * @version    : 1.0
 */

#ifndef __FLUID_KERNEL__

#define __FLUID_KERNEL__

#include <vector_types.h>

namespace Physika {
// forward declaration
struct PBDFluidSimulateParams;
/**
 * @brief setup :the fluid system parameters
 * @param[in] params :the fluid system parameters
 */
__host__ void setSimulationParams(PBDFluidSimulateParams* hostParam);

/**
 * @brief calculate the hash value of each particle
 *
 * @param[out] gridParticleHash :the hash value of each particle
 * @param[in] pos :the position of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void calcParticlesHashKernel(
    unsigned int* gridParticleHash,
    float4*       pos,
    unsigned int  numParticles);

/**
 * @brief sort the particles based on their hash value
 *
 * @param[out] gridParticleHash the hash value of each particle
 * @param[out] gridParticleIndex the index of each particle
 * @param[in] numParticles the number of particles
 */
__global__ void findCellRangeKernel(
    unsigned int* cellStart,         // output: cell start index
    unsigned int* cellEnd,           // output: cell end index
    unsigned int* gridParticleHash,  // input: sorted grid hashes
    unsigned int  numParticles);

/**
 * @brief advect the particles
 *
 * @param[in] position :the position of each particle
 * @param[in] velocity :the velocity of each particle
 * @param[out] predictedPos :the predicted position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] deltaTime :the time step
 * @param[in] numParticles :the number of particles
 */
__global__ void fluidAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        deltaTime,
    unsigned int numParticles);

__global__ void contactSDF(
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

// calculate density and lagrange multiplier.
__global__ void calcLagrangeMultiplier(
    float4*       predictedPos,
    float4*       velocity,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

__global__ void calcDeltaPosition(
    float4*       predictedPos,
    float4*       velocity,
    float3*       deltaPos,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles);

/**
 * @brief add the delta position to the predicted position
 *
 * @param[in/out] predictedPos :the predicted position of each particle
 * @param[in] deltaPos :the delta position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void addDeltaPosition(
    float4*      predictedPos,
    float3*      deltaPos,
    float*       particlePhase,
    unsigned int numParticles);

/**
 * @brief solver the distance constrain and friction constrain for each particle
 *
 * @param[in] predictedPos :the predicted position of each particle
 * @param[in/out] deltaPos :the delta position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] cellStart :the start index of each cell
 * @param[in] cellEnd :the end index of each cell
 * @param[in] gridParticleHash :the hash value of each particle
 * @param[in] numParticles :the number of particles
 * @param[in] numCells :the number of cells
 */
__global__ void distanceConstrain(
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

/**
 * @brief update the velocity and position of each particle
 *
 * @param[in/out] position : the position of each particle
 * @param[in/out] velocity :the velocity of each particle
 * @param[in] predictedPos :the predicted position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] invDeltaTime :the inverse of delta time
 * @param[in] numParticles :the number of particles
 */
__global__ void updateVelocityAndPosition(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        invDeltaTime,
    unsigned int numParticles);

/**
 * @brief handle collisions between particles and the container
 *
 * @param[in/out] position : the position of each particle
 * @param[in/out] velocity : the velocity of each particle
 * @param[in] predictedPos : the predicted position of each particle
 * @param[in] moveDirection : the move direction of each particle. (This is given by the collision system mesh_point_sdf)
 * @param[in] moveDistance : the move distance of each particle.   (This is given by the collision system mesh_point_sdf)
 * @param[in] particlePhase: the phase of each particle
 * @param[in] collision_particle_id: the id of particles which collide with other objects
 * @param[in] numCollisionParticles: the number of particles which collide with other objects
 */
__global__ void solverCollisionConstrain(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles);

__global__ void add_surfacetension(
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

__global__ void add_adhesionforce(
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

}  // namespace Physika

#endif  // !__FLUID_KERNEL__
