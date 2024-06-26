/**
 * @file       : fluid_solid_coupling_kernel.cuh
 * @author     : Yuhang Xu (mr.xuyh@qq.com)
 * @date       : 2023-08-17
 * @description: This file contains the declaration of CUDA functions for the fluid solid coupling system. The main functionality
 *               of this file revolves around the GPU accelerated physics simulation of fluid solid coupling systems. It includes
 *               definitions for setting up and managing simulation parameters, performing calculations related to the
 *               fluid solid coupling system like hashing and sorting of particles, particle advection, updating velocities and
 *               positions, handling inter-particle and particle-container collisions, and other related operations.
 *
 *               This file interfaces with the 'fluid_solid_coupling' module and specifically uses the fluid_solid_coupling_params.hpp for
 *               the parameters required in the simulation. The CUDA kernels defined in this file are expected to be
 *               called by the host-side (CPU) code.
 *
 * @dependencies: This file depends on CUDA runtime headers and the fluid_solid_coupling_params.hpp from the 'fluid_solid_coupling' module.
 *
 * @version    : 1.0
 */

#ifndef __FLUID_SOLID_COUPLING_KERNEL__

#define __FLUID_SOLID_COUPLING_KERNEL__

#include <vector_types.h>

#include "include/mat3.cuh"

namespace Physika {
// forward declaration
struct FluidSolidCouplingParams;
/**
 * @brief setup :the fluid solid coupling system parameters
 * @param[in] params :the fluid solid coupling system parameters
 */
__host__ void setSimulationParams(FluidSolidCouplingParams* hostParam);

/**
 * @brief calculate the hash value of each particle
 *
 * @param[out] gridParticleHash :the hash value of each particle
 * @param[in] pos :the position of each particle
 * @param[in] numParticles :the number of particles
 */
__global__ void calcCouplingParticlesHashKernel(
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
__global__ void findCellRangeCouplingKernel(
    unsigned int* cellStart,         // output: cell start index
    unsigned int* cellEnd,           // output: cell end index
    unsigned int* gridParticleHash,  // input: sorted grid hashes
    unsigned int  numParticles);

/**
 * @brief advect the particles
 *
 * @param[in] position :the position of each particle
 * @param[in] velocity :the velocity of each particle
 * @param[in] collisionForce :the collision force of each particle
 * @param[out] predictedPos :the predicted position of each particle
 * @param[in] particlePhase :the phase of each particle
 * @param[in] deltaTime :the time step
 * @param[in] numParticles :the number of particles
 */
__global__ void particleAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    int*         phase,
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
__global__ void addDeltaPositionCoupling(
    float4*      postion,
    float4*      predictedPos,
    float3*      deltaPos,
    int*         phase,
    unsigned int numParticles);


__global__ void calcLagrangeMultiplierCoupling(
    int*          phase,
    float4*       predictedPos,
    float4*       velocity,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells);

__global__ void calcDeltaPositionCoupling(
    int*          phase,
    float4*       predictedPos,
    float4*       velocity,
    float3*       deltaPos,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles);

__global__ void calMassCenterMatrixR(
    int*          phase,
    float4*       predicted_pos,
    float3*       cm,
    const float3* radius_pos,
    const bool    allow_stretch,
    unsigned int  num_particles,
    mat3*         R);

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

__global__ void updateSolidVelocityAndPosition(
    int*          phase,
    float4*       position,
    float4*       predicted_pos,
    float3*       delta_pos,
    const float3* cm,
    const float3* radius_pos,
    const float   stiffness,
    float4*       velocity,
    const float   invDeltaTime,
    const mat3*   R,
    unsigned int  num_particles);

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
__global__ void solverCollisionConstrainCoupling(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    int*          particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles);

}  // namespace Physika

#endif  // !__FLUID_SOLID_COUPLING_KERNEL__
