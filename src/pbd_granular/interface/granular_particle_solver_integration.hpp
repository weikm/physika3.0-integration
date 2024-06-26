/**
 * @file      : granular_particle_solver.hpp
 * @author    : Yuanmu Xu (xyuan1517@gmail.com)
 * @date      : 2023-07-17
 * @brief     : This file declares the granular solver which is designed to handle simulations of granular material systems.
 *              The implementation is based on the paper "Unified particle physics for real-time applications" presented at TOG 2014.
 *              More specifically, the granular material and friction handling described in Section 6 of the paper are the main focus of this solver.
 *
 *              This solver can be employed for simulations requiring realistic behavior of granular materials (like sand, gravel, etc.)
 *              The file defines the GranularPhase enum, GranularComponent struct, and the GranularParticleSolverIntegration class which inherits from the base Solver class.
 *              Various methods for managing the granular system such as initializing, resetting, stepping, attaching/detaching objects, etc. are also declared.
 *              A SolverConfig struct is provided for configuring simulation time parameters.
 *
 *              Use cases include, but are not limited to, game development, real-time physics simulations, CGI, etc.
 *              The simulation parameters, although tested and set to provide stable execution, can be modified according to the specific requirements of the application.
 *
 * @version   : 1.0
 */

/**
 * @file      : granular_particle_solver_integration.hpp
 * @author    : weikeming
 * @date      : 2023-06-25
 * @brief     : 
 *
 * @version   : 1.0
 */



#ifndef GRANULAR_PARTICLE_SOLVER_INTEGRATION_HPP
#define GRANULAR_PARTICLE_SOLVER_INTEGRATION_HPP


#include <stdlib.h>
#include <vector>

//#include "framework/solver.hpp"
#include <vector_types.h>


namespace Physika {
// forwar declaration
struct GranularSimulateParams;

/**
 * @brief the phase of particles
 * Considering that the engine may have other multiple PBD solvers, the phase is used to distinguish the particles of different solvers.
 *
 * this solver is granular solver, so the particle phase in this solver is GRANULAR.
 */
enum class GranularPhase
{
    GRANULAR = 1,
    SOLID    = 2
};
/**
 * @brief the granular component of the solver
 * When creating a granular component, a cube-shaped particles are generated by default,
 *
 * it can be changed by setting m_host_pos and m_host_vel.
 * When apply a new scene, you should set the m_host_pos, m_host_vel and m_host_phase at the same time.
 */
struct GranularParticleComponentIntegration

{

    // host data(CPU data)
    /**
     * these data are used to initialize the granular component
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
    std::vector<float> m_host_phase;


    // device(GPU) data
    /**
     * Users should not access or modify these data.
     * If you want to set a new scene for granular simulation, you should use the set...() methods in GranularParticleSolverIntegration.
     */
    float* m_device_pos = nullptr;
    float* m_device_vel = nullptr;

    // particle phase
    float* m_device_phase = nullptr;
    bool   m_bInitialized = false;
    /**
     * Used for rendering, return the vertical buffer object(VBO) of the granular particles position. (Not implemented yet)
     */
    // unsigned int m_pos_vbo;
    unsigned int m_num_particles = 0;

    // collision handling （these params are changed in the collision system for sand particles colliding with rigid bodies).
    // They cannot be modified by the GranularSolver but can be accessed. Only the collision system can change them.
    unsigned int  m_num_collisions        = 0;        // how many particles are in collision
    unsigned int* m_collision_particle_id = nullptr;  // The particle id of collision particles (according to m_device_pos).
    float*        m_move_direction  = nullptr;  // data layout: x, y, z. The move direction for partiles to avoid collision.
    float*        m_move_distance   = nullptr;  // data layout: single float. The move distance for particles to avoid collision
    float*        m_collision_force = nullptr;  // The force for partiles from collision.

    /**
     * @brief reset the granular component to the initial state
     */
    void reset();

    /**
     * @brief construction function of granular component
     *
     * @param[in] radius the radius of particles in simulation
     */
    // GranularParticleComponentIntegration();

    /**
     * @brief add instance function of granular component
     *
     * @param[in]  : radius         the radius of particles in simulation
     * @param[in]  : lb             left-bottom coordinate of granular cube
     * @param[in]  : size           size of granular cube
     * @param[in]  : vel_start      initial vel of granular cube
     */
    void addInstance(float particle_radius, float3 lb, float3 size, float3 vel_start);

    /**
     * @brief add instance function of granular component
     *
     * @param[in]  : radius         the radius of particles in simulation
     * @param[in]  : init_pos       initial pos of granular cube
     */
    void addInstance(float particle_radius, std::vector<float3>& init_pos);

    /**
     * @brief malloc memory for GPU data
     *
     * @param[in] numParticles the number of particles in simulation
     */
    void initialize();

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
     * @brief destruction function of granular component, free the GPU memory
     *
     * Using GranularParticleSolverIntegration::set...() methods instead of this function is recommended
     */
    // ~GranularParticleComponentIntegration();
};

/**
 * Granular solver. (Position based dynamics)
 * When create a Granular solver, a default simulation parameter will automatically be created. Its parameters are tested and can run stably
 * But you can also modify some parameters through some function methods inside the class.
 *
 * This solver is used to simulate granular particles.
 * Before using this solver, GranularComponent needs to be defined and successfully bound to this solver.
 * By modifying the parameters such as friction coefficient and stiffness coefficient in the solver,
 * the fine-tuning effect of particle phenomenon can be realized. The default params if used to simulate sand.
 *
 * the friction constrain and distance constrain are used to simulate the granular particles.
 * An efficient parallel hash-grid (GREEN, S. 2008. Cuda particles. nVidia Whitepaper.) is implemented in this solver to accelerate the particle neighbor finding.
 */
class GranularParticleSolverIntegration //: public Solver
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
        bool         m_write2ply{ false };          // Whether to write the particle data to ply file
    };
    /**
     * @brief construction function of granular solver
     */
    GranularParticleSolverIntegration();

    /**
     * @brief destruction function of granular solver
     */
    ~GranularParticleSolverIntegration();

    /**
     * @brief initialize the granular solver to get it ready for execution.
     *
     * @return  true if initialization succeeds, otherwise return false
     *
     */
    bool initialize() /*override*/;

    /**
     * @brief get the initialization state of the solver.
     *
     * @return   true if solver has been properly initialized, otherwise return false
     */
    bool isInitialized() const /*override*/;

    /**
     * @brief reset the solver to newly constructed state
     *
     * @return    true if reset succeeds, otherwise return false
     */
    bool reset() /*override*/;

    /**
     * @brief step the solver ahead through a prescribed time step. The step size is set via other methods
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool step() /*override*/;

    /**
     * @brief run the solver till termination condition is met. Termination conditions are set via other methods
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool run() /*override*/;

    /**
     * @brief check whether the solver is applicable to given object
     *        If the object contains granular Components, it is defined as applicable.
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    //bool isApplicable(const Object* object) const /*override*/;

    /**
     * @brief attach an object to the solver
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully attached
     *            false if error occurs, e.g., object is nullptr, object has been attached before, etc.
     */
    //bool attachObject(Object* object) /*override*/;

    /**
     * @brief detach an object to the solver
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully detached
     *            false if error occurs, e.g., object is nullptr, object has not been attached before, etc.
     */
    //bool detachObject(Object* object) override;

    /**
     * @brief clear all object attachments
     */
    void clearAttachment() /*override*/;

    /**
     * @brief set the simulation data and total time of simulation
     */
    void config(SolverConfig& config);

public:
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
     * @brief Set the Height Field
     * @param[in] height: 1d array
     * @param[in] unit_height: height field cell size
     * @param[in] height_x_num: height field x axis cell num
     * @param[in] height_z_num: height field z axis cell num
     */
    void setHeightField(const std::vector<float>& height, const float unit_height, const int height_x_num, const int height_z_num);

    /**
     * @brief set granular particles' positions
     *
     * @param[in] host data of particle position array
     * @return true if position data is successfully set
     */
    bool setParticlePosition(const std::vector<float>& position);

    /**
     * @brief set granular particles' velocity
     *
     * @param[in] host data of particle velocity array
     * @return true if velocity data is successfully set
     */
    bool setParticleVelocity(const std::vector<float>& velocity);

    /**
     * @brief set particle phase
     * In order to complete the coupling with other solver particles(such as the surface particles of vehicles),
     * we need to give granular particles and other particles different phase. So that the granular solver will not
     * solve other particles by mistake.
     *
     * @param[in] host data of particle velocity array
     * @return true if velocity data is successfully set
     */
    bool setParticlePhase(const std::vector<float>& particlePhase);

    /**
     * @brief set the device pointer of particle collision force
     *
     * @param[in] host data of particle CollisionForce array
     * data layout: x, y, z;
     *
     * @return the device pointer of particle collision force.
     *         nullptr if no valid object/component is attached to this solver.
     */
    bool setParticleCollisionForcePtr(float* collisionforce);

    /**
     * @brief get the device pointer of particle position
     *
     * float* particlePositionPtr = getParticlePositionPtr();
     * data layout: x, y, z, insMass;
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
    float* getParticlePhasePtr();

    /**
     * @brief get the device pointer of particle collision force
     *
     * float* ParticleCollisionForcePtr = getParticleCollisionForcePtr();
     * data layout: x, y, z;
     *
     * @return the device pointer of particle collision force.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float* getParticleCollisionForcePtr();

    /**
     * @brief set the granular particle radius.
     *
     * @param[in] particleRadius : the radius of granular particle.
     * the value of particleRadius show be positive.
     * If a negative or zero value is given, the value of particle_radius in simulation will not be changed.
     *
     */
    void setParticleRadius(const float& particleRadius);

    /**
     * @brief get the granular particle radius. (eg: used for rendering or collision detect)
     *
     * @param[out] particleRadius : the radius of granular particle.
     *
     */
    void getParticleRadius(float& particleRadius);

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
     * @param[out] staticFriction : the static friction coefficient.
     * The value of the static friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getStaticFrictionCoeff(float& staticFriction);

    /**
     * @brief set static friction coefficient for particles
     *
     * @param[in] dynamicFriction : the static friction coefficient
     * The value of the dynamic friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void setDynamicFrictionCoeff(float& dynamicFriction);


    /**
     * @brief get static friction coefficient for particles
     *
     * @param[out] dynamicFriction : the static friction coefficient
     * The value of the dynamic friction coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getDynamicFrictionCoeff(float& dynamicFriction);

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
     * @brief get the stitffness coefficient of position based dynamics solver
     * If the stiffness is too large, simulation will be crash.
     *
     * @param[out] stiffness : Stiffness coefficient that determines the displacement of PBD particles
     * The value of the static Stiffness coefficient should be between (0, 1].
     * Settings outside this range are invalid and and will be clamped to this range.
     */
    void getStiffness(float& stiffness);

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
     * @brief Unified particle physics for real-time applications. (Section 4.5 Particle Sleeping.)
     *
     * @param[in] threshold : Sleep threshold in the world.
     *
     * Positional drift may occur when constraints are not fully satisfied at the end of a time-step.
     * We address this by freezing particles in place if their velocity has dropped below a user-defined threshold,
     *
     */
    void setSleepThreshold(float threshold);
    
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


    void setComponent(GranularParticleComponentIntegration* granular_particle_component)
    {
        m_granular_particle_component = granular_particle_component;
    }

    //GranularSimulateParams* getParams() {
    //    return m_params;
    //}
    void setInitialize();
    


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
    //Object*                 m_granular_particle;

    GranularParticleComponentIntegration* m_granular_particle_component = new GranularParticleComponentIntegration;
    unsigned int*                         m_collision_particle_id       = nullptr;
    bool                    m_is_init;
    double                  m_cur_time;
    SolverConfig            m_config;
    GranularSimulateParams* m_params;
    float*                                m_host_pos = nullptr;

private:
    // GPU parameters
    // semi-lagrangian advect
    float* m_device_delta_pos = nullptr;
    float* m_device_predicted_pos = nullptr;
    // hash neighbour search
    unsigned int* m_device_cell_start = nullptr;
    unsigned int* m_device_cell_end   = nullptr;
    unsigned int* m_device_grid_particle_hash = nullptr;//
    unsigned int  m_num_grid_cells;
    unsigned int  m_num_particles;

    // for height field
    std::vector<float> m_host_height;
    float*             m_device_height = nullptr;
    float              m_unit_height;
    int                m_height_x_num;
    int                m_height_z_num;
};

}  // namespace Physika

#endif  // PHYSIKA_GRANULAR_SOLVER_H