/**
 * @file      : pbf_nonnewton_solver.hpp
 * @author    : Long Shen (sl_111211@163.com)
 * @date      : 2023-10-07
 * @brief     : This file declares the solver which is designed to handle simulations of nonNewtonian fluid.
 *              The implementation is based on the paper "Position Based Fluids" presented at TOG 2013 and the paper "Smoothed particle hydrodynamics techniques for the physics based
 *              simulation of fluids and solids" presented at arXiv.
 *
 *              This solver can be employed for simulations requiring nonNewtonian-viscous behavior of particle fluid (like chocolate, paint, etc.)
 *              The file defines the SimMaterial enum, NNComponent struct, and the PBFNonNewtonSolver class which inherits from the base Solver class.
 *              Various methods for managing the particle system such as initializing, resetting, stepping, attaching/detaching objects, etc. are also declared.
 *              A SolverConfig struct is provided for configuring simulation time parameters.
 *
 *              Use cases include, but are not limited to, game development, real-time physics simulations, CGI, etc.
 *              The simulation parameters, although tested and set to provide stable execution, can be modified according to the specific requirements of the application.
 *
 * @version   : 1.0
 */

#ifndef PHYSIKA_PBF_NONNEWTON_SOLVER_HPP
#define PHYSIKA_PBF_NONNEWTON_SOLVER_HPP

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>

#include "framework/solver.hpp"
#include "pbf_nonnewton/source/enum.hpp"
#include "pbf_nonnewton/source/const_pack.hpp"
#include "utils/interface/model_helper.hpp"

namespace Physika {

struct NonNewtonFluidComponent;

class PBFNonNewtonSolver : public Solver
{
public:
    struct SolverConfig
    {
        float3 m_gravity{ 0.f, -9.8f, 0.f };
        double m_dt{ 0.f };
        double m_total_time{ 0.f };
        int    m_iter_num{ 2 };
        bool   m_use_qwUI{ false };
    };
    /**
     * @brief construction function of granular solver
     */
    PBFNonNewtonSolver();

    /**
     * @brief destruction function of granular solver
     */
    ~PBFNonNewtonSolver();

    /**
     * @brief initialize the nn solver to get it ready for execution.
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
     *        If the object contains granular Components, it is defined as applicable.
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
     * @brief set parent scene info to the solver
     *
     * @param[in] scene_lb left-bottom coordinate of the scene
     * @param[in] scene_size size of the scene
     *
     */
    void setSceneBoundary(float3 scene_lb, float3 scene_size);

    /**
     * @brief set particle radius info to the solver
     *
     * @param[in] radius particle radius
     *
     */
    void setUnifiedParticleRadius(float radius);

    void setParticlePosition(const std::vector<float3>& position);

    void setParticleVelocity(const std::vector<float3>& velocity);

    void setParticlePhase(const std::vector<float>& particlePhase);

    /**
     * @brief final operates when sim done, free device memory or other ops
     */
    void finalize();

    float3* getPosDevPtr() const;

    float3* getVelDevPtr() const;

    float* getVisDevPtr() const;

    void setExtForce(float3* device_ext_force);

private:
    /**
     * @brief neighbor search
     */
    void updateNeighbors();

    /**
     * @brief compute volume of boundary particles for fluid-rigid coupling
     */
    void computeRigidParticleVolume();

    /**
     * @brief compute external force
     */
    void computeExtForce();

    /**
     * @brief compute density of fluid particles
     */
    void computeDensity();

    /**
     * @brief compute delta_x using PBF density constraint
     */
    void computeDxFromDensityConstraint();

    /**
     * @brief advect pos from delta_x
     */
    void applyDx();

    /**
     * @brief compute cross-viscous force
     */
    void computeVisForce();

    /**
     * @brief dump info of neighbor searcher
     */
    void dumpNeighborSearchInfo();

    /**
     * @brief initialize device resource of solver (substep of initialize())
     */
    bool initDeviceSource();

    /**
     * @brief free device memory
     */
    void freeMemory();

private:
    bool         m_is_init{ false };
    SolverConfig m_config;
    Object*      m_fluid_obj{ nullptr };
    Object*      m_solid_obj{ nullptr };
    Object*      m_bound_obj{ nullptr };
    double       m_cur_time{ 0.f };
    float        m_particle_radius{ 0.f };
    bool         m_isStart{ true };

    std::vector<float3>      m_host_overall_pos;
    std::vector<float3>      m_host_overall_vel;
    std::vector<SimMaterial> m_host_overall_material;
    ConstPack                m_host_constPack{};

    ConstPack*   m_device_constPack{ nullptr };
    int3*        m_device_ns_cellOffsets{ nullptr };
    uint32_t*    m_device_ns_particleIndices{ nullptr };
    uint32_t*    m_device_ns_cellIndices{ nullptr };
    uint32_t*    m_device_ns_cellStart{ nullptr };
    uint32_t*    m_device_ns_cellEnd{ nullptr };
    uint32_t*    m_device_ns_neighborNum{ nullptr };
    uint32_t*    m_device_ns_neighbors{ nullptr };
    float3*      m_device_pos{ nullptr };
    float3*      m_device_predictPos{ nullptr };
    float3*      m_device_vel{ nullptr };
    float3*      m_device_acc{ nullptr };
    float3*      m_device_dx{ nullptr };
    float*       m_device_density{ nullptr };
    float*       m_device_volume{ nullptr };
    float*       m_device_lam{ nullptr };
    float*       m_device_vis{ nullptr };
    float*       m_device_shearRate{ nullptr };
    SimMaterial* m_device_material{ nullptr };
    float3*      m_device_ext_force{ nullptr };
    double       m_device_mem{ 0 };
};

struct NonNewtonFluidComponent
{
    void reset();  // physika requires every component type to have a reset() method

    /**
     * @brief construction function of NNComponent
     *
     * @param[in] material the simMaterial of cur object
     */
    NonNewtonFluidComponent(SimMaterial material);

    /**
     * @brief default construction function of NNComponent
     */
    NonNewtonFluidComponent() = default;

    /**
     * @brief construction function of NNComponent
     *
     * @param[in] vel_start the start velocity of the obj
     */
    void setStartVelocity(float3 vel_start);

    /**
     * @brief to check if the component is empty
     */
    bool hasParticles() const;

    /**
     * @brief set simMaterial type of the obj
     */
    void sestMaterial(SimMaterial material);

    /**
     * @brief get simMaterial type of the obj
     */
    SimMaterial getMaterial() const;

    /**
     * @brief get total particle num of the component
     */
    uint32_t getParticleNum() const;

    /**
     * @brief add particles into component
     *
     * @param[in] particles host particle set which stores the pos of particle
     */
    void addParticles(std::vector<float3> particles);

    /**
     * @brief create instance by config
     *
     * @param[in] config ParticleModelConfig used by ModelHelper
     */
    void addInstance(ParticleModelConfig config);

private:
    std::vector<float3> m_host_meta_pos;
    std::vector<float3> m_host_meta_vel;
    SimMaterial         m_host_meta_material;
    uint32_t            m_particleNum{ 0 };

public:
    std::vector<float3> m_host_cur_pos;
    std::vector<float3> m_host_cur_vel;
    std::vector<float3> m_host_ext_force;

    // neighbor_search related
};

}  // namespace Physika

#endif  // PHYSIKA_PBF_NONNEWTON_SOLVER_HPP
