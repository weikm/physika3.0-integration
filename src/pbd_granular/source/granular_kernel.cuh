/**
 * @file       : granular_kernel.hpp
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-07-17
 * @description: This file contains the declaration of CUDA functions for the granular system. The main functionality
 *               of this file revolves around the GPU accelerated physics simulation of granular systems. It includes
 *               definitions for setting up and managing simulation parameters, performing calculations related to the
 *               granular system like hashing and sorting of particles, particle advection, updating velocities and
 *               positions, handling inter-particle and particle-container collisions, and other related operations.
 *
 *               This file interfaces with the 'pbd_granular' module and specifically uses the granular_params.hpp for
 *               the parameters required in the simulation. The CUDA kernels defined in this file are expected to be
 *               called by the host-side (CPU) code.
 *
 * @dependencies: This file depends on CUDA runtime headers and the granular_params.hpp from the 'pbd_granular' module.
 *
 * @version    : 1.0
 */

#ifndef __GRANULAR_KERNEL__

#define __GRANULAR_KERNEL__

#include <vector_types.h>

namespace Physika {
// forward declaration
struct GranularSimulateParams;
/**
 * @brief setup :the granular system parameters
 * @param[in] params :the granular system parameters
 */
__host__ void setSimulationParams(GranularSimulateParams* hostParam);

/**
 * @brief calculate the hash value of each particle
 *
 * @param[out] gridParticleHash :the hash value of each particle
 * @param[in] pos :the position of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void calcParticlesHashKernelGranular(
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
__global__ void findCellRangeKernelGranular(
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
 * @param[in] collisionForce :the force for each particle from collision
 * @param[in] particlePhase :the phase of each particle
 * @param[in] deltaTime :the time step
 * @param[in] numParticles :the number of particles
 */
__global__ void granularAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    float*       particlePhase,
    float*       height,
    float        unit_height,
    int          height_x_num,
    int          height_z_num,
    float        deltaTime,
    unsigned int numParticles);

/**
 * @brief add the delta position to the predicted position
 *
 * @param[in/out] predictedPos :the predicted position of each particle
 * @param[in] deltaPos :the delta position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void addDeltaPositionGranular(
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
__global__ void distanceConstrainGranluar(
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
__global__ void updateVelocityAndPositionGranular(
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
__global__ void solverCollisionConstrainGranular(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles);

}  // namespace Physika

#endif  // !__GRANULAR_KERNEL__
