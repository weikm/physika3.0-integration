/**
 * @file       : solid_liquid_phase_change_solver.hpp
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @brief      : This file declares the solid-liquid phase change solver designed to handle simulations of solid-liquid phase change phenomena
 *               . The implementation is based on 'Heat-based bidirectional phase shifting simulation using position-based dynamics'.
 * @version    : 1.0
 */


#ifndef SOLID_LIQUID_PHASE_CHANGE_SOLVER_HPP
#define SOLID_LIQUID_PHASE_CHANGE_SOLVER_HPP

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <vector_functions.h>

#include "framework/solver.hpp"

namespace Physika {
// forward declaration
struct SolidLiquidPhaseChangeParams;

enum class SLPCPhase
{
    BOUNDARY = 0,
    SOLID    = 1,
    FLUID    = 2
};

/**
 * @brief the SolidLiquidPhaseChange component of the solver
 * When creating a SolidLiquidPhaseChange component, a cube-shaped particles are generated by default,
 *
 * it can be changed by setting m_host_pos and m_host_vel.
 * When apply a new scene, you should set the m_host_pos, m_host_vel,m_host_tem,m_host_type at the same time.
 */
struct SolidLiquidPhaseChangeComponent
{
    // host data(CPU data)
    /**
     * these data are used to initialize the SolidLiquidPhaseChange component
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

    std::vector<float> m_host_tem; /**< Temperature of particles in simulation */
    std::vector<int> m_host_type; /**< Type of particles in simulation */
    std::vector<int> m_host_init_id; /**< Initial ID of particles in simulation */



    // device(GPU) data
    /**
     * Users should not access or modify these data.
     */
    float* m_device_pos;
    float* m_device_vel;

    float* m_device_tem;
    int* m_device_type;
    int* m_device_init_id1;
    int* m_device_init_id2;
    int* m_device_init_id_2_rest;
    int* m_device_init_id_2_now;
    float* m_device_init_pos;

    bool   m_bInitialized;
    unsigned int m_num_particles;

    /*scene parameters*/
    float3* m_external_force;         // External forces involved in the simulation

   
    /**
     * @brief reset the component to the initial state
     */
    void reset();

    /**
     * @brief construction function of this component
     *
     * @param[in] config the config of initial scene
     */
    SolidLiquidPhaseChangeComponent(void* config);

    /**
     * @brief construction function of this component without parameter     
     */
    SolidLiquidPhaseChangeComponent();

    /**
     * @brief add instance function of SLPC component
     *
     * @param[in]  : radius         the radius of particles in simulation
     * @param[in]  : init_pos       initial pos of granular cube
     */
    void addInstance(float particle_radius, std::vector<float3>& init_pos,int type,float tem);

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
     * @brief destruction function of  component, free the GPU memory
     *
     * Using SolidLiquidPhaseChangeParticleSolver::set...() methods instead of this function is recommended
     */
    ~SolidLiquidPhaseChangeComponent();
};

/**
 * SolidLiquidPhaseChange solver. (Position based dynamics)
 * When create a SolidLiquidPhaseChange solver, a default simulation parameter will automatically be created. Its parameters are tested and can run stably
 * But you can also modify some parameters through some function methods inside the class.
 *
 * This solver is used to simulate SolidLiquidPhaseChange particles.
 * Before using this solver, SolidLiquidPhaseChangeComponent needs to be defined and successfully bound to this solver.
 *
 * An efficient parallel hash-grid (GREEN, S. 2008. Cuda particles. nVidia Whitepaper.) is implemented in this solver to accelerate the particle neighbor finding.
 */
class SolidLiquidPhaseChangeSolver : public Solver
{
public:
    struct SolverConfig
    {
        double       m_dt{ 1.f / 60.f };          // Time step
        double       m_total_time{ 100.f };         // total time of simulation
        float        m_static_friction{ 1.f };    // Coefficient of static friction, used in friction constraint
        float        m_dynamic_friction{ 0.5f };  // Coefficient of dynamic friction, used in friction constraint
        float        m_stiffness{ 0.5f };         // Stiffness constant for the pressure forces in the SPH simulation
        float        m_gravity{ -9.8f };          // The acceleration due to gravity in the simulation
        float        m_sleep_threshold{ 0.02f };  // Velocity threshold below which particles are considered as 'sleeping'
        unsigned int m_solver_iteration{ 3 };     // Maximum number of iterations for the constraint solvera.
        bool         m_is_convect{false};              // Whether to enable convection in the simulation
        float3       m_world_size{ 50.f, 50.f, 50.f };    // Size of the simulation world
        float3       m_fluid_size{ 0.f, 0.f, 0.f };    // Size of the fluid region
        float3       m_fluid_locate{ 0.f, 0.f, 0.f };  // Location of the fluid region
        float        m_fluid_tem{ 90.0f };               // Temperature of the fluid
        float3       m_solid_size{ 0.f, 0.f, 0.f };    // Size of the solid region
        float3       m_solid_locate{ 0.f, 0.f, 0.f };     // Location of the solid region
        float        m_solid_tem{ 0.f };               // Temperature of the solid
        float3       m_boundary_size{ 0.f, 0.f, 0.f };    // Size of the boundary region
        float3       m_boundary_locate{0.f,0.f,0.f};           // Location of the boundary region
        float        m_boundary_tem{ 0.f };       // Temperature of the boundary
        bool         m_write_ply{ false };         // whether to write the ply
        bool         m_write_statistics{ false };   // whether to write statistics
        bool         m_radiate{false};         // whether to enable heat radiate

    };
    /**
     * @brief construction function of solver
     */
    SolidLiquidPhaseChangeSolver();

    /**
     * @brief destruction function of solver
     */
    ~SolidLiquidPhaseChangeSolver();

    /**
     * @brief initialize the solver to get it ready for execution.
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
     *        If the object contains Components, it is defined as applicable.
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
    SolidLiquidPhaseChangeParams* m_params;

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
     * @brief set solid-liquid phase change part particles' positions
     *
     * @param[in] host data of particle position array
     * @return true if position data is successfully set
     */
    bool setParticlePosition(const std::vector<float>& position);

    /**
     * @brief set solid-liquid phase change part particles' velocity
     *
     * @param[in] host data of particle velocity array
     * @return true if velocity data is successfully set
     */
    bool setParticleVelocity(const std::vector<float>& velocity);

    /**
     * @brief set particle phase
     * In order to complete the coupling with other solver particles(such as the surface particles of vehicles),
     * we need to give fluid particles and other particles different phase. So that the fluid solver will not
     * solve other particles by mistake.
     *
     * @param[in] host data of particle velocity array
     * @return true if velocity data is successfully set
     */
    bool setParticlePhase(const std::vector<int>& particlePhase);

    /**
     * @brief set fluid particle external force
     *
     * @return true if force data is successfully set
     */
    bool setParticleExternalForce(float3* external_force);

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
     * @brief get the device pointer of particle velocity
     *
     * float* particleVelocityPtr = getParticleVelocityPtr();
     * data layout: x, y, z, 0.0;
     *
     * @return the device pointer of particle velocity.
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
     * @brief get the device pointer of particle external force
     *
     * float* particleExternalForcePtr = getParticleExternalForcePtr();
     * data layout: x, y, z;
     *
     * @return the device pointer of particle external force.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float3* getParticleExternalForcePtr();

    /**
    * @brief get the particle radius. (eg: used for rendering )
    * 
    * @param[out] particleRadius : the radius of particle.
    * 
    */
    void getParticleRadius(float& particleRadius);

    void setParam(const std::string& paramName, void* value);

    void* getParam(const std::string& paramName);
        

protected:
    
    /**
     * @brief destroy the solver
     */
    void _finalize();

    /**
     * @brief write the particles data to ply file
     */
    void writeToPly(const int& step_id);

    void writeToStatistics(const int& step_id, const float& frame_time);

    /**
     * @brief free particle memory
     */
    void freeParticleMemory(const int& numParticles);

    /**
     * @brief malloc particle memory
     */
    void mallocParticleMemory(const int& numParticles);

private:
    Object*                 m_solid_fluid_phase_change;
    bool                    m_is_init;
    double                  m_cur_time;
    SolverConfig            m_config;    
    float*                  m_host_pos;
    float*                  m_host_tem;
    int*                    m_host_type;

private:
    // GPU parameters
    // semi-lagrangian advect
    float* m_device_delta_pos;
    float* m_device_predicted_pos;
    float* m_device_delta_tem;
    float* m_device_latent_tem;
    // hash neighbour search
    unsigned int* m_device_cell_start;
    unsigned int* m_device_cell_end;
    unsigned int* m_device_grid_particle_hash;
    unsigned int  m_num_grid_cells;
    unsigned int  m_num_particles;

    unsigned int* m_device_cell_start_solid;
	unsigned int* m_device_cell_end_solid;
	unsigned int* m_device_grid_particle_hash_solid;
};

}  // namespace Physika

#endif  // SOLID_LIQUID_PHASE_CHANGE_SOLVER_HPP