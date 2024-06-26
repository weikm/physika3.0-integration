/**
 * @file       : gas_liquid_phase_change_solver.hpp
 * @author     : Wang Qianwei (729596003@qq.com)
 * @date       : 2023-11-17
 * @brief      : This file declares the gas-liquid phase change solver designed to handle simulations of gas-liquid phase change phenomena
 *               .
 * @version    : 1.0
 */

#ifndef GAS_LIQUID_PHASE_CHANGE_SOLVER_HPP
#define GAS_LIQUID_PHASE_CHANGE_SOLVER_HPP

#include <stdlib.h>
#include <vector>

#include "pbd_gasLiquidPhaseChange/source/gl_phase_change_system.hpp"
#include "pbd_gasLiquidPhaseChange/source/thermal_pbf_solver.hpp"
#include "pbd_gasLiquidPhaseChange/source/gas_liquid_phase_change_params.hpp"
#include "framework/solver.hpp"

namespace Physika {
// forward declaration
struct GasLiquidPhaseChangeParams;

/**
 * @brief the GasLiquidPhaseChange component of the solver
 * 
 * This component consists of three types of particles: 'fluid' particles, 'boundary' particles, and 'heaten' particles. 
 * The boundary particles are adiabatic particles, and the nucleation points for vaporization/liquefaction only occur on the 'heaten' particles.
 * The information on whether a heaten particle is a nucleation point is stored in 'nucleation_sets'.
 *
 */
struct GasLiquidPhaseChangeComponent
{
    GasLiquidPhaseChangeParams* m_config;
    // host data(CPU data)
    /**
     * these data are used to initialize the GasLiquidPhaseChange component
     * they can be modified by the user.
     *
     * When modifying these data, you should comply with the following data layout methods
     */

     /**
     * data layout: x, y, z
     * x, y, z are the positions of fluid(gas/liquid) particles in simulation
     */
    std::vector<float3> m_host_pos_fluid;

    /**
     * data layout: x, y, z
     * x, y, z are the positions of heaten particles in simulation
     */
    std::vector<float3> m_host_pos_heaten;

    /**
     * data layout: x, y, z
     * x, y, z are the positions of boundary particles in simulation
     */
    std::vector<float3> m_host_pos_boundary;
    
    /**
     * data layout: x, y, z
     * x, y, z are the velocities of fluid(gas/liquid) particles in simulation
     */
    std::vector<float3> m_host_vel_fluid;

    /**
     * data: 0 or 1
     * Init heaten particle, '1': nucleation point, '0': not nucleation point
     */
    std::vector<int>    m_host_nucleation_marks;  // bound with heaten particles
    /**
     * Init heaten particles temperature
     */
    std::vector<float>  m_host_tem_heaten;
     /**
     * Init fluid particles temperature
     */
    std::vector<float> m_host_tem_fluid;
    /*
    * Fluid capacity for fluid model, set to twice of the initial fluid size.
    */
    int m_fluid_capacity;
    
    /*
    * Initialization flag(all).
    */
    bool m_bInitialized;
    
    /*
    * Initialization flag of world space
    */
    bool m_space_initialized;

    /*
    * Initialization flag of fluids particle positions
    */
    bool m_fluid_initialized;

    // The management structure of particle(fluid, boundary and heaten) data on device(GPU). 
    std::shared_ptr<BoilingSystem> m_sim_system;
    
    /**
     * num particles of fluids
     */
    unsigned int m_num_particles;

    /**
     * @brief reset the component to the initial state
     */
    void reset();

    /**
     * @brief construction function of this component
     *
     */
    GasLiquidPhaseChangeComponent();

    /**
     * @brief malloc memory for GPU data
     *
     * @param[in] numParticles the number of particles in simulation
     */
    void initialize();

    // Get particle data ptr on device(GPU)
    /**
     * Get device data pointer by these methods.
     */

    /**
     * @brief get fluid particle position ptr
     *
     * @param[out] device(CUDA) ptr of postions 
     */
    float* getFluidParticlePositionPtr();

    /**
     * @brief get fluid particle velocity ptr
     *
     * @param[out] device(CUDA) ptr of velocity 
     */
    float* getFluidParticleVelocityPtr();
     
    /**
     * @brief get fluid particle external force. Set external forces for fluid particles in this ptr.
     *
     * @param[out] device(CUDA) ptr of external force.
     */
    float* getFluidExternalForcePtr();

    /**
     * @brief get fluid particle temperature
     *
     * @param[out] device(CUDA) ptr of temperature 
     */
    float* getFluidParticleTemperaturePtr();

     /**
     * @brief get fluid type('0':gas or '1':liquid) particle temperature
     *
     * @param[out] device(CUDA) ptr of temperature 
     */
    int* getFluidParticleTypePtr();

    /**
     * @brief get fluid particle num
     *
     * @param[out] fluid particle num 
     */
    int getFluidParticleNum();


    // Set/Get parameters and fluid particles data
    /**
     * Set/Get parameters and fluid particles data by these methods.
     */

    /**
     * @brief Use this to initialize particle radius. Otherwise, particle radius is set to 0.01 by default.
     *
     * @param[in]  particle radius 
     */
    bool initializeParticleRadius(float radius);

    /**
     * @brief initialize world space boundary
     *
     * @param[in]  world boundary 0,0,0 -> space_size
     */
    bool initializeWorldBoundary(float3 space_size);

     /**
     * @brief initialize particle position
     *
     * @param[in] position vector
     */
    bool initializePartilcePosition(const std::vector<float3>& init_pos);

    /**
     * @brief initialize particle velocity
     *
     * @param[in] position vector
     */
    bool initializePartilceVelocity(const std::vector<float3>& init_vel);

    /**
    *  @brief add an instance of fluid before simulation
    * 
    *  @param lb, left button of the cube
    *  @param cube_size, size of the cube
    *  @param sampling_size, particle sampling size 
    */
    bool initializeLiquidInstance(float3 lb, float3 cube_size, float sampling_size);

    /**
     * @brief get latent heat
     *
     * @param[out] latent heat
     */
    float getLatentHeat();

    /**
     * @brief get boiling point temperature
     *
     * @param[out] boiling point temperature
     */
    float getBoilingPoint();


    /**
    * @brief free the cuda memory
    * The component will be set to not initlized.
    */
    void freeMemory();
    /**
     * @brief destruction function of  component, free the GPU memory
     *
     * Using GasLiquidPhaseChangeParticleSolver::set...() methods instead of this function is recommended
     */
    ~GasLiquidPhaseChangeComponent();
};

/**
 * GasLiquidPhaseChange solver. (Position based dynamics)
 *
 */
class GasLiquidPhaseChangeSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt{ 0.003 }; // time step
        float m_total_time{ 0.0f }; //total time of simulation
    };
    /**
     * @brief construction function of solver
     */
    GasLiquidPhaseChangeSolver();

    /**
     * @brief destruction function of solver
     */
    ~GasLiquidPhaseChangeSolver();

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
    /**
     * @brief get the device pointer of particle position
     *
     * float* particlePositionPtr = getParticlePositionPtr();
     * data layout: x, y, z;
     *
     * @return the device pointer of particle position.
     *         nullptr if no valid object/component is attached to this solver.
     */
    float* getParticlePositionPtr();

    /**
    * @brief get the particle radius. (eg: used for rendering )
    * 
    * @param[out] particleRadius : the radius of particle.
    * 
    */
    void getParticleRadius(float& particleRadius);

    /**
    * @brief Set particle external force.
    * 
    * @param[in] ex_force: external force vector on host.
    * 
    */
    bool setPartilceExternalForce(std::vector<float3>& ex_force);

private:
    Object*      m_gas_liquid_phase_change;
    bool         m_is_init;
    double       m_cur_time;
    SolverConfig m_config;

};
}  // namespace Physika
#endif  // GAS_LIQUID_PHASE_CHANGE_SOLVER_HPP