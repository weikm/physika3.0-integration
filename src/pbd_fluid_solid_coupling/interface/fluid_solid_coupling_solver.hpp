/**
 * @file      : fluid_solid_coupling_solver.hpp
 * @author    : Yuhang Xu (mr.xuyh@qq.com)
 * @date      : 2023-08-17
 * @brief     : This file declares the fluid solid coupling solver which is designed to handle simulations of fluid solid coupling material systems.
 *              The implementation is based on the paper "Unified particle physics for real-time applications" presented at TOG 2014.
 *              More specifically, the fluid solid coupling material and friction handling described in Section 6 of the paper are the main focus of this solver.
 *
 *              This solver can be employed for simulations requiring realistic behavior of fluid solid coupling materials (like sand, gravel, etc.)
 *              The file defines the CouplingParticlePhase enum, CouplingParticleComponent struct, and the ParticleFluidSolidCouplingSolver class which inherits from the base Solver class.
 *              Various methods for managing the fluid solid coupling system such as initializing, resetting, stepping, attaching/detaching objects, etc. are also declared.
 *              A SolverConfig struct is provided for configuring simulation time parameters.
 *
 *              Use cases include, but are not limited to, game development, real-time physics simulations, CGI, etc.
 *              The simulation parameters, although tested and set to provide stable execution, can be modified according to the specific requirements of the application.
 *
 * @version   : 1.0
 */

#ifndef FLUID_SOLID_COUPLING_HPP
#define FLUID_SOLID_COUPLING_HPP

#include <stdlib.h>
#include <string>
#include <vector>
#include <vector_types.h>

#include "framework/solver.hpp"
#include "utils/interface/model_helper.hpp"

namespace Physika {
// forwar declaration
struct FluidSolidCouplingParams;

/**
 * @brief the phase of particles
 * Considering that the engine may have other multiple PBD solvers, the phase is used to distinguish the particles of different solvers.
 *
 * this solver is fluid solid coupling solver, so the particle phase in this solver is FluidSolid.
 */
enum class CouplingParticlePhase
{
    GRANULAR = 1,
    SOLID    = 2,
    FLUID    = 3
};
/**
 * @brief the fluid solid coupling component of the solver
 * When creating a fluid solid coupling component, a cube-shaped particles are generated by default,
 *
 * it can be changed by setting m_host_pos and m_host_vel.
 * When apply a new scene, you should set the m_host_pos, m_host_vel and m_host_phase at the same time.
 */
struct CouplingParticleComponent
{
    // host data(CPU data)
    /**
     * these data are used to initialize the fluid solid coupling component
     * they can be modified by the user.
     *
     * When modifying these data, you should comply with the following data layout methods
     */

    /**
     * data layout: x, y, z, mass
     * x, y, z are the position of particles in simulation
     * mass is the mass of particles in simulation
     */
    std::vector<float> m_host_pos;
    /**
     * data layout: x, y, z, 1
     * x, y, z are the velocity of particles in simulation
     * 1 is the phase of particles in simulation
     */
    std::vector<float> m_host_vel;
    std::vector<int>   m_host_phase;

    /**
     * data layout: x, y, z
     * x, y, z are the radius vector relative to center position
     */
    std::vector<float> m_radius_pos;  // for SOLID

    /**
     * the center pos of the solid
     */
    float3             m_rest_cm;

    // device(GPU) data
    /**
     * Users should not access or modify these data.
     * If you want to set a new scene for fluid solid coupling simulation, you should use the set...() methods in FluidSolidParticleSolver.
     */
    float* m_device_pos{ nullptr };
    float* m_device_vel{ nullptr };

    float* m_device_radius_pos{ nullptr };  // for SOLID

    // collision handling ��these params are changed in the collision system for particles colliding with other objects).
    float* m_device_collision_force{ nullptr };  // for advert

    // particle phase
    int* m_device_phase{ nullptr };
    bool m_bInitialized;

    unsigned int m_num_particles{ 0 };
    unsigned int m_num_solid{ 0 };
    unsigned int m_num_fluid{ 0 };

    /**
     * @brief reset the fluid solid coupling component to the initial state
     */
    void reset();

    /**
     * @brief construction function of fluid solid coupling component
     */
    CouplingParticleComponent() = default;

    /**
     * @brief add instance function of granular component
     *
     * @param[in]  : particle_radius the radius of particles in simulation
     * @param[in]  : shape particle shape
     * @param[in]  : lb              left-bottom coordinate of cube
     * @param[in]  : size            size of cube
     * @param[in]  : vel_start       initial vel of particle
     * @param[in]  : particle_inmass initial inmass of granular cube
     */
    /*void CouplingParticleComponent::addInstance(
        CouplingParticlePhase mat,
        std::string           shape,
        float3                vel_start,
        float3                lb,
        float3                size,
        float                 particle_radius,
        float                 particle_inmass
    );*/

    /**
     * @brief add instance function of granular component
     *
     * @param[in]  : mat             fluid or solid
     * @param[in]  : config          ParticleModelConfig used by ModelHelper
     * @param[in]  : vel_start       initial vel of granular cube
     * @param[in]  : particle_inmass initial inmass of particle
     */
    void CouplingParticleComponent::addInstance(
        CouplingParticlePhase mat,
        ParticleModelConfig   config,
        float3                vel_start,
        float                 particle_inmass);

    /**
     * @brief add instance function of granular component
     *
     * @param[in]  : mat             fluid or solid
     * @param[in]  : init_pos        init pos
     * @param[in]  : vel_start       initial vel of granular cube
     * @param[in]  : particle_inmass initial inmass of particle
     */
    void CouplingParticleComponent::addInstance(
        CouplingParticlePhase mat,
        std::vector<float3>   init_pos,
        float3                vel_start,
        float                 particle_inmass);

    /**
     * @brief malloc memory for GPU data
     *
     * @param[in] numParticles the number of particles in simulation
     */
    void initialize(int numParticles);

    inline float frand()
    {
        return rand() / ( float )RAND_MAX;
    }
    /**
    * @brief free the cuda memory
    * You can also actively call it, and then use the initialize() function to allocate new GPU memory.
    * 
    *  You can also actively call it, and then use the initialize() function to allocate new GPU memory.
    *  Used for new scene creation, old scenes need to be destroyed. 
    *  Or when the simulation is over, this function would be called to free the GPU memory.
    */
    void freeMemory();
    /**
     * @brief destruction function of fluid solid coupling component, free the GPU memory
     *
     * Using FluidSolidParticleSolver::set...() methods instead of this function is recommended
     */
    ~CouplingParticleComponent();
};

/**
 * FluidSolid solver. (Position based dynamics)
 * When create a FluidSolid solver, a default simulation parameter will automatically be created. Its parameters are tested and can run stably
 * But you can also modify some parameters through some function methods inside the class.
 *
 * This solver is used to simulate fluid solid coupling particles.
 * Before using this solver, FluidSolidComponent needs to be defined and successfully bound to this solver.
 * By modifying the parameters such as friction coefficient and stiffness coefficient in the solver,
 * the fine-tuning effect of particle phenomenon can be realized. The default params if used to simulate sand.
 *
 * the friction constrain and distance constrain are used to simulate the fluid solid coupling particles.
 * An efficient parallel hash-grid (GREEN, S. 2008. Cuda particles. nVidia Whitepaper.) is implemented in this solver to accelerate the particle neighbor finding.
 */
class ParticleFluidSolidCouplingSolver : public Solver
{
public:
    struct SolverConfig
    {
        double       m_dt{ 1.f / 60.f };          // Time step
        double       m_total_time{ 0.f };         // total time of simulation
        float        m_static_friction{ 1.f };    // Coefficient of static friction, used in friction constraint
        float        m_dynamic_friction{ 0.5f };  // Coefficient of dynamic friction, used in friction constraint
        float        m_stiffness{ 0.5f };         // Stiffness constant for the pressure forces in the SPH simulation
        float        m_gravity{ -9.8f };          // The acceleration due to gravity in the simulation
        float        m_sleep_threshold{ 0.02f };  // Velocity threshold below which particles are considered as 'sleeping'
        unsigned int m_solver_iteration{ 3 };     // Maximum number of iterations for the constraint solvera.
    };
    /**
     * @brief construction function of fluid solid coupling solver
     */
    ParticleFluidSolidCouplingSolver();

    /**
     * @brief destruction function of fluid solid coupling solver
     */
    ~ParticleFluidSolidCouplingSolver();

    /**
     * @brief initialize the fluid solid coupling solver to get it ready for execution.
     *
     * @return  true if initialization succeeds, otherwise return false
     *
     */
    bool initialize() override;

    /**
     * @brief get the initialization state of the solver.
     *
     * @return   true if solver has been properly initialized, otherwise return false
     */
    bool isInitialized() const override;

    /**
     * @brief reset the solver to newly constructed state
     *
     * @return    true if reset succeeds, otherwise return false
     */
    bool reset() override;

    /**
     * @brief step the solver ahead through a prescribed time step. The step size is set via other methods
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool step() override;

    /**
     * @brief run the solver till termination condition is met. Termination conditions are set via other methods
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool run() override;

    /**
     * @brief check whether the solver is applicable to given object
     *        If the object contains fluid solid coupling Components, it is defined as applicable.
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    bool isApplicable(const Object* object) const override;

    /**
     * @brief attach an object to the solver
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully attached
     *            false if error occurs, e.g., object is nullptr, object has been attached before, etc.
     */
    bool attachObject(Object* object) override;

    /**
     * @brief detach an object to the solver
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully detached
     *            false if error occurs, e.g., object is nullptr, object has not been attached before, etc.
     */
    bool detachObject(Object* object) override;

    /**
     * @brief clear all object attachments
     */
    void clearAttachment() override;

    /**
     * @brief set the simulation data and total time of simulation
     */
    void config(SolverConfig& config);

public:
    /**
     * @brief get the boundary of this simulator
     *
     * @param[out] lx  the lower boundary of the world in x direction
     * @param[out] ly  the lower boundary of the world in y direction
     * @param[out] lz  the lower boundary of the world in z direction
     * @param[out] ux  the upper boundary of the world in x direction
     * @param[out] uy  the upper boundary of the world in y direction
     * @param[out] uz  the upper boundary of the world in z direction
     *
     */
    void getWorldBoundary(float& lx, float& ly, float& lz, float& ux, float& uy, float& uz);

    /**
     * @brief set the boundary of this simulator
     *
     * @param[in] lx  the lower boundary of the world in x direction
     * @param[in] ly  the lower boundary of the world in y direction
     * @param[in] lz  the lower boundary of the world in z direction
     * @param[in] ux  the upper boundary of the world in x direction
     * @param[in] uy  the upper boundary of the world in y direction
     * @param[in] uz  the upper boundary of the world in z direction
     *
     * @return    true if the boundary is successfully set
     */
    bool setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz);

    /**
     * @brief set fluid solid coupling particles' positions
     *
     * @param[in] host data of particle position array
     * @return true if position data is successfully set
     */
    bool setParticlePosition(const std::vector<float>& position);

    /**
     * @brief set fluid solid coupling particles' velocity
     *
     * @param[in] host data of particle velocity array
     * @return true if velocity data is successfully set
     */
    bool setParticleVelocity(const std::vector<float>& velocity);

    /**
     * @brief set particle phase
     * In order to complete the coupling with other solver particles(such as the surface particles of vehicles),
     * we need to give fluid solid coupling particles and other particles different phase. So that the fluid solid coupling solver will not
     * solve other particles by mistake.
     *
     * @param[in] host data of particle phase array
     * @return true if phase data is successfully set
     */
    bool setParticlePhase(const std::vector<int>& particlePhase);

    /**
     * @brief get the device pointer of particle position
     *
     * float* particlePositionPtr = getParticlePositionPtr();
     * data layout: x, y, z, 1.0;
     *
     * @return the device pointer of particle position.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float* getParticlePositionPtr();

    /**
     * @brief get the device pointer of particle Velocity
     *
     * float* particleVelocityPtr = getParticleVelocityPtr();
     * data layout: x, y, z, 0.0;
     *
     * @return the device pointer of particle Velocity.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float* getParticleVelocityPtr();

     /**
     * @brief get the device pointer of particle Phase
     *
     * float* particlePhasePtr = getParticlePhasePtr();
     * data layout: Phase;
     *
     * @return the device pointer of particle Phase.
     *         nullptr if no valid object/component is attached to this solver.
     */
    int* getParticlePhasePtr();

    /**
     * @brief get the device pointer of particle collision force
     *
     * float* particlePhasePtr = getParticleCollisionForcePtr();
     * data layout: x, y, z;
     *
     * @return the device pointer of particle collision force.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float* getParticleCollisionForcePtr();

    /**
     * @brief set the device pointer of particle collision force
     *
     * float* particlePhasePtr = getParticleCollisionForcePtr();
     * data layout: x, y, z;
     *
     * @return the device pointer of particle collision force.
     *         nullptr if no valid object/component is attached to this solver.
     */
    void setParticleCollisionForcePtr(float* device_collision_force);

    /**
    * @brief get the fluid solid coupling particle radius. (eg: used for rendering or collision detect)
    * 
    * @param[out] radius : the radius of fluid solid coupling particle.
    * 
    */
    void getParticleRadius(float& particleRadius);

    /**
     * @brief set the fluid solid coupling particle radius.
     *
     * @param[in] radius : the radius of fluid solid coupling particle. 
     * the value of radius show be positive. 
     * If a negative or zero value is given, the value of particle_radius in simulation will not be changed.
     *
     */
    void setParticleRadius(const float& particleRadius);

    /**
     * @brief get static friction coefficient for particles
     *
     * @param[out] staticFriction : the static friction coefficient.
     * The value of the static friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getStaticFrictionCoeff(float& staticFriction);

    /**
     * @brief set static friction coefficient for particles
     *
     * @param[in] staticFriction : the static friction coefficient.
     * The value of the static friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void setStaticFrictionCoeff(float& staticFriction);

    /**
     * @brief get static friction coefficient for particles
     *
     * @param[out] dynamicFriction : the static friction coefficient
     * The value of the dynamic friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getDynamicFrictionCoeff(float& dynamicFriction);

    /**
     * @brief set static friction coefficient for particles
     *
     * @param[in] dynamicFriction : the static friction coefficient
     * The value of the dynamic friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void setDynamicFrictionCoeff(float& dynamicFriction);

    /**
     * @brief get the stitffness coefficient of position based dynamics solver
     * If the stiffness is too large, simulation will be crash.
     *
     * @param[out] stiffness : Stiffness coefficient that determines the displacement of PBD particles
     * The value of the static Stiffness coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getStiffness(float& stiffness);

    /**
     * @brief set the stitffness coefficient of position based dynamics solver
     * If the stiffness is too large, simulation will be crash.
     *
     * @param[in] stiffness : Stiffness coefficient that determines the displacement of PBD particles
     * The value of the static Stiffness coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void setStiffness(float& stiffness);

    /**
     * @brief get the number of iterations for the position based dynamics solver
     *
     * @param[out] iteration : number of iterations, it cannot be negative. It should be greater than 0. (default value: 3)
     *
     * Increasing the number of iterations can make the simulation more stable.
     * And it can increase the initial accumulation height of sand to a certain extent.
     * Lowering it will reduce the pile height of the sand somewhat.
     *
     */
    void getSolverIteration(unsigned int& iteration);

    /**
     * @brief Set the number of iterations for the position based dynamics solver
     *
     * @param[in] iteration : number of iterations, it cannot be negative. It should be greater than 0. (default value: 3)
     *
     * Increasing the number of iterations can make the simulation more stable.
     * And it can increase the initial accumulation height of sand to a certain extent.
     * Lowering it will reduce the pile height of the sand somewhat.
     *
     */
    void setSolverIteration(unsigned int& iteration);

    /**
     * @brief get the gravity of the simulation
     *
     * @param[out] x : extern force in x direction
     * @param[out] y : extern force in y direction
     * @param[out] z : extern force in z direction
     *
     * It is allowed to modify these parameter to any value as a extern force and lead to some interesting results.
     *
     */
    void getGravity(float& x, float& y, float& z);

    /**
     * @brief Set the gravity of the simulation
     *
     * @param[in] gravity : gravity in the world.(default value: -9,8)
     *
     * It is allowed to modify this parameter to any value as a extern force and lead to some interesting results.
     *
     */
    void setGravity(const float& gravity);

    /**
     * @brief Set the gravity of the simulation
     *
     * @param[in] x : extern force in x direction
     * @param[in] y : extern force in y direction
     * @param[in] z : extern force in z direction
     *
     * It is allowed to modify these parameter to any value as a extern force and lead to some interesting results.
     *
     */
    void setGravity(const float& x, const float& y, const float& z);

    /**
     * @brief Unified particle physics for real-time applications. (Section 4.5 Particle Sleeping.)
     *
     * @param[out] threshold : Sleep threshold in the world.
     *
     * Positional drift may occur when constraints are not fully satisfied at the end of a time-step.
     * We address this by freezing particles in place if their velocity has dropped below a user-defined threshold,
     *
     */
    void getSleepThreshold(float& threshold);

    /**
     * @brief Unified particle physics for real-time applications. (Section 4.5 Particle Sleeping.)
     *
     * Positional drift may occur when constraints are not fully satisfied at the end of a time-step.
     * We address this by freezing particles in place if their velocity has dropped below a user-defined threshold,
     *
     */
    void setSleepThreshold(float& threshold);

    /**
     * @brief set the world origin of the simulation
     *
     * @param[out] x: the x coordinate of the world origin
     * @param[out] y: the y coordinate of the world origin
     * @param[out] z: the z coordinate of the world origin
     *
     * For example: setWorldOrigin(0, 0, 0);
     * this worldOrigin will be used to calculate the hash value.
     *
     */
    void getWorldOrigin(float& x, float& y, float& z);

    /**
     * @brief set the world origin of the simulation
     *
     * @brief x: the x coordinate of the world origin
     * @brief y: the y coordinate of the world origin
     * @brief z: the z coordinate of the world origin
     *
     * For example: setWorldOrigin(0, 0, 0);
     * this worldOrigin will be used to calculate the hash value.
     *
     */
    void setWorldOrigin(const float& x, const float& y, const float& z);

protected:
    /**
     * @brief Handle the collision between particles and other objects
     *
     * @param[in] collision_particle_id : the id of the particle which collides with other objects
     * @param[in] moveDirection : the direction of the particle movement
     * @param[in] moveDistance :  the distance of the particle movement
     * @param[in] collision_num : the number of particles which collide with other objects
     *
     */
    void handleCollision(unsigned int* collision_particle_id, float* moveDirection, float* moveDistance, unsigned int collision_num);

    /**
     * @brief destroy the solver
     */
    void _finalize();

    /**
     * @brief write the particles data to ply file
     */
    void writeToPly(const int& step_id);

    /**
     * @brief free particle memory
     */
    void freeParticleMemory(const int& numParticles);

    /**
     * @brief malloc particle memory
     */
    void mallocParticleMemory(const int& numParticles);

private:
    Object*                   m_particle;
    unsigned int*             m_collision_particle_id;
    bool                      m_is_init;
    double                    m_cur_time;
    SolverConfig              m_config;
    FluidSolidCouplingParams* m_params;
    float*                    m_host_pos;

private:
    // GPU parameters
    // semi-lagrangian advect
    float* m_device_delta_pos;
    float* m_device_predicted_pos;
    // hash neighbour search
    unsigned int* m_device_cell_start;
    unsigned int* m_device_cell_end;
    unsigned int* m_device_grid_particle_hash;
    unsigned int  m_num_grid_cells;
    unsigned int  m_num_particles;
};

}  // namespace Physika


#endif  // PHYSIKA_FLUID_SOLID_COUPLING_H
