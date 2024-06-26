/**
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-10
 * @description: declaration of snow solver
 * @version    : 1.0
 */
#pragma once
#include "framework/solver.hpp"

#include <stdlib.h>
#include <vector>

#include "mpm_snow/source/include/mat3.hpp"
#include "mpm_snow/source/snow_params.cuh"
#include "utils/interface/model_helper.hpp"

namespace Physika {

struct MPMSnowComponent
{
    void reset();  // physika requires every component type to have a reset() method
    // host data
    std::vector<Point> m_hParticles;  // host particle data.
    std::vector<float> m_boundary;    // boundary of the snow. Data layout: x_min, y_min, z_min, x_max, y_max, z_max

    // data entries here
    Point* m_deviceParticles;  // device particle data
    Grid*  m_deviceGrid;       // device grid data
    float* m_devicePos;        // device position data
    float* m_deviceVel;        // device velocity data
    float* m_externelForce;    // device external force data
    bool         m_bInitialized;  // whether the component is initialized
    unsigned int m_numParticles;  // number of particles
    unsigned int m_numGrids;      // number of grids(Material point method girds)
    MPMSnowComponent() = default;
    /**
     * @brief add instance function of snow component
     *
     * @param[in]  : config          ParticleModelConfig used by ModelHelper
     * @param[in]  : vel_start       initial vel of granular cube
     */
    void MPMSnowComponent::addInstance(
        ParticleModelConfig   config,
        std::vector<float3> vel_start);

    /**
     * @brief add instance function of snow component
     *
     * @param[in]  : init_pos        init pos
     * @param[in]  : vel_start       initial vel 
     */
    void MPMSnowComponent::addInstance(
        std::vector<float3> init_pos,
        std::vector<float3> vel_start);
    
    void         initialize(int numParticles);
    void         deInit();
    inline float frand()
    {
        return rand() / ( float )RAND_MAX;
    }
    ~MPMSnowComponent();
};

class MPMSnowSolver : public Solver
{
public:
    struct SolverConfig
    {
        double m_dt{ 1.0 / 60.f };      // time step
        double m_total_time{ 0.0f };    // total time of simulation
        bool   m_write2ply{ false };    // whether write the result to ply file
        bool   m_showGUI{ false };      // whether show the GUI
    };
    MPMSnowSolver(std::vector<float> world_boundary);
    ~MPMSnowSolver();

    /**
    * @brief initialize the solver
    *
    * @return    true if initialization succeeds, otherwise return false
    */
    bool initialize() override;

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
     *        If the object contains snow Components, it is defined as applicable.
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
    void config(const SolverConfig& config);

public:
    //=============== get method ========================
    /**
     * @brief Get the Youngs Modulus
     * @return the youngs modulus
     */
    float getYoungsModulus() const;

    /**
     * @brief Get the poisson ratio
     * @return the poisson ratio
     */
    float getPoissonRatio() const;

    /**
     * @brief Get the hardening coefficient
     * @return the hardening coeffcient
     */
    float getHardeningCeoff() const;

    /**
     * @brief Get the Compressio coefficient
     * @return the Compressio coeffcient
     */
    float getCompressionCoeff() const;
    
    /**
     * @brief Get the Friction coefficient
     * @return the Friction coeffcient
     */
    float getFrictionCoeff() const;

    /**
     * @brief Get the stretch coefficient
     * @return the stretch coeffcient
     */
    float getStretch() const;

    /**
     * @brief Get the Stick behavior
     * @return whether the snow particles stick to the solid boundary
     */
    bool getIfStick() const;

    /**
    * @brief Get the Particle Position
    * @param[out] pos: the particle position (this must be a gpu pointer.)
    * @param[out] numParticles: the number of particles.
    */
    bool getParticlePositionPtr(float* pos, unsigned int& numParticles);

public:
    //=============== set method ========================
    /**
     * @brief Set the Height Field
     * @param[in] height: 1d array
     * @param[in] unit_height: height field cell size
     * @param[in] height_x_num: height field x axis cell num
     * @param[in] height_z_num: height field z axis cell num
     */
    void setHeightField(const std::vector<float>& height, const float unit_height, const int height_x_num, const int height_z_num);
    /**
     * @brief Set the Height Field
     * @param[in] height: 1d array
     * @param[in] unit_height: height field cell size
     * @param[in] height_x_num: height field x axis cell num
     * @param[in] height_z_num: height field z axis cell num
     */
    void setHeightField(float*& height, const float unit_height, const int height_x_num, const int height_z_num);
    /**
     * @brief Set the Youngs Modulus
     * @param[in] youngs_modulus: the youngs modulus
     */
    void setYoungsModulus(const float& youngs_modulus);

    /**
    * @brief Set the grid boundary
    * @param[in] x_min: the minimum x value of the grid boundary
    * @param[in] x_max: the maximum x value of the grid boundary
    * @param[in] y_min: the minimum y value of the grid boundary
    * @param[in] y_max: the maximum y value of the grid boundary
    * @param[in] z_min: the minimum z value of the grid boundary
    * @param[in] z_max: the maximum z value of the grid boundary
    * @note: the grid boundary is a cuboid
    */
    void setGridBoundary(const float& x_min, const float& x_max, const float& y_min, const float& y_max, const float& z_min, const float& z_max);

    /**
     * @brief Set the poisson ratio
     * @param[in] poissom_ratio: the poisson ratio
     */
    void setPoissonRatio(const float& poisson_ratio);

    /**
     * @brief Set the hardening coefficient
     * @param[in] hardening: the hardening coeffcient. (0, 1), values out of this range will be clamped
     */
    void setHardeningCeoff(const float& hardening);

    /**
     * @brief Set the compress coefficient
     * @param[in] compress: the compress coeffcient. (0, 0.1), values out of this range will be clamped.
     */
    void setCompressionCoeff(const float& compress);

    /**
     * @brief Set the frictionCoeff
     * @param[in] frictionCoeff: On the premise of allowing viscosity, the friction effect coefficient with the contact surface
     */
    void setFrictionCoeff(const float& frictionCoeff);

    /**
     * @brief Set the stretch coefficient
     * @param[in] stretch: the stretch coeffcient. Not recommended to change this value.This value is sensitive and easy to cause instability.
     */
    void setStretch(const float& stretch);

    /**
     * @brief Set the Stick behavior
     * @param[in] Stick: whether the snow particles stick to the solid boundary. Default is false.
     */
    void setStick(const bool& stick);

    /**
     * @brief Set the Particle perties From host array.
     * @param[in] particles: the particle from host.
     */
    void setParticle(const std::vector<Point>& particles);

    /**
    * @brief Set the particle velocity from host array.
    * @param[in] vel: the particle velocity array.
    */
    void setParticleVelocity(const std::vector<float>& vel);

    /**
     * @brief Set the particle velocity from host array.
     * @param[in] pos: the particle position array.
     */
    void setParticlePosition(const std::vector<float>& pos);

    /**
    * @brief Set the Particle Position From dev
    * @param[in] pos: the particle position from device. (this must be a gpu pointer.)
    * @param[in] numParticles: the number of particles.
    * @note: make sure the size of the pos array is 3 * numParticles, the program cannot chack the size of a gpu pointer, so make sure the size is correct.
    */
    void setParticlePositionFromdev(float* pos, unsigned int numParticles);

    /**
	 * @brief Set the Particle Velocity
	 * @param[in] vel: the particle velocity from host.
	 */
    void setWorldBoundary(const float& x_min, const float& x_max, const float& y_min, const float& y_max, const float& z_min, const float& z_max);
protected:
    /**
     * @brief write the snow particles to ply file.
     */
    void writeToPly(const int& step_id);

private:
    Object*            m_snow_particles;
    bool               m_is_init;
    double             m_cur_time;
    SolverConfig       m_config{ 1.f / 60.f, 0.f };
    SolverParam        m_params;
    std::vector<float> m_hostParticles;

    // for height field
    std::vector<float> m_host_height;
    float* m_device_height;
    float  m_unit_height;
    int    m_height_x_num;
    int    m_height_z_num;
};

}  // namespace Physika