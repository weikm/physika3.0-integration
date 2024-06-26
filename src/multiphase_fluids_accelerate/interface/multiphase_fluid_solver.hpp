/*
 * @author     : Yan Xiao Xiao (1982126814@qq.com)
 * @date       : 2023-06-10
 * @description: declaration of MultiphaseFluidSolver class,used in the scene of porous materials
 *               interacting with multiphase fluids
 * @version    : 1.0
 */

#pragma once

#include "framework/solver.hpp"
#include "multiphase_fluids_accelerate/source/common_defs.h"
#include "multiphase_fluids_accelerate/source/vector.h"
#include "multiphase_fluids_accelerate/source/particle_defs.hpp"
#include "multiphase_fluids_accelerate/source/kernel_host.cuh"

namespace Physika {
/**
 * component interface the solver use ,this solver need four kind of components
 *
 *
 */
struct MultiphaseFluidComponent
{
    void reset()
    {
        m_pos = nullptr;
        // m_mix_velocity = nullptr;
        m_alpha  = nullptr;
        pnum     = 0;
        load_num = 0;
    }
    float* m_pos;
    // float* m_mix_velocity;
    float* m_device_pos;  // gpu
    float* m_color;       // gpu
    float* m_alpha;//<! the volume fraction of particles
    /* m_alpha notes,fluid phase can stand for sand,water,gas particles:
    * 0 0 1 0 0 rigid particle
    * 1 0 0 0 0 fluid phase 1
    * 0 1 0 0 0 fluid phase 2
    * 0 0 0 1 0 fluid phase 3
    * 0 0 0 0 1 fluid phase 4
    */
    float* m_external_force;
    int    pnum;
    int    load_num;
};

class MultiphaseFluidSolver : public Solver
{
public:
    /**
     * a sample of simulation configs
     */
    struct SolverConfig
    {
        double m_dt;
        double m_total_time;
    };

    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    MultiphaseFluidSolver();
    ~MultiphaseFluidSolver();

    bool initialize() override;

    bool isInitialized() const override;

    bool reset() override;

    bool step() override;

    bool run() override;

    bool isApplicable(const Object* object) const override;

    bool attachObject(Object* object) override;

    bool detachObject(Object* object) override;

    void clearAttachment() override;

    /**
     * functions below ars preparations before call the solver
     */
    void config(const SolverConfig& config);
    /**
     * @brief get config information from SolverConfig
     *
     * @param[in] const SolverConfig&    the solver config type
     *
     * @return
     */
    void setParametersAll();
    /**
     * @brief set all the parameters needed for solver
     *
     */
    void setParameterFloat(int name, float value);
    /**
     * @brief set float parameter m_param[name] by value
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] float the value of this param
     *
     */
    void setBoundaryParam(Vector3DF max_corner, Vector3DF min_corner);
    /**
     * @brief set Boundary with max corner and min corner
     *
     * @param[in] Vector3DF   the max_corner
     * @param[in] Vector3DF   the min_corner
     * 
     */
    Vector3DF getBoundaryParamMax();
    /**
     * @brief get Boundary of max corner 
     *
     * @return Vector3DF    max corner of boundary
     */
    Vector3DF getBoundaryParamMin();
    /**
     * @brief get Boundary of min corner
     *
     * @return Vector3DF    min corner of boundary
     */
    void setPhaseDensityParam(int phase, float density);
    /**
     * @brief set phase density 
     *
     * @param[in] int   the phase of particle,phase range from 0-4
     * @param[in] float the value of phase density
     *
     */
    float getPhaseDensityParam(int phase);
    /**
     * @brief get phase density
     *
     * @param[in] int   the phase of particle,phase range from 0-4
     *
     * @return float    the value of phase density
     */
    float getParameterFloat(int name) const;
    /**
     * @brief get float parameter m_param[name] by name
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return float    the value of m_param[name]
     */
    void setParameterVec3F(int name, float x, float y, float z);
    /**
     * @brief set float parameter m_vec[name] by x,y,z
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] float the x value of this param
     * @param[in] float the y value of this param
     * @param[in] float the z value of this param
     */
    Vector3DF getParameterVec3F(int name) const;
    /**
     * @brief get Vector3DF param m_vec[name]
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return  Vector3DF   the value of m_vec[name]
     */
    void setParameterVec3I(int name, int x, int y, int z);
    /**
     * @brief set float parameter m_vec_int[name] by x,y,z
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] int the x value of this param
     * @param[in] int the y value of this param
     * @param[in] int the z value of this param
     */
    Vector3DI getParameterVec3I(int name) const;
    /**
     * @brief get Vector3DI param m_vec_int[name]
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return  Vector3DI   the value of m_vec_int[name]
     */
    void setParameterMultiphaseFloat(int name, float value);
    /**
     * @brief set float parameter m_multi_phase[name] by value
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] float the value of this param
     *
     */
    float getParameterMultiphaseFloat(int name) const;
    /**
     * @brief get float parameter m_multi_phase[name] by name
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return float    the value of m_multi_phase[name]
     */
    void setParameterMultiphaseVec3F(int name, float x, float y, float z);
    /**
     * @brief set float parameter m_multi_phase_vec[name] by x,y,z
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] float the x value of this param
     * @param[in] float the y value of this param
     * @param[in] float the z value of this param
     */
    float3 getParameterMultiphaseVec3F(int name) const;
    /**
     * @brief get float3 param m_multi_phase_vec[name]
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return  float3   the value of m_multi_phase_vec[name]
     */
    void setParameterBool(int name, bool value);
    /**
     * @brief set float parameter m_flag[name] by value
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     * @param[in] bool the value of this param
     *
     */
    bool getParameterBool(int name) const;
    /**
     * @brief get float parameter m_flag[name] by name
     *
     * @param[in] int   the name of param refer to particle_defs.hpp
     *
     * @return float    the value of m_flag[name]
     */
    void AllocateParticleMemory(int Number);
    /**
     * @brief allocate cpu memory for attribute buffers
     *
     * @param[in] int Number    the number of all the particles
     *
     */
    void InitialiseRigid();
    /**
     * @brief initialize rigid particles with set coordinates
     *
     */
    void InitialiseRotateRigid();
    /**
     * @brief initialize rotated rigid particles with set coordinates
     *
     */
    void InitialisePositionPhase();
    /**
     * @brief print x,y,z number of phase particles with set coordinates
     *
     */
    void InitialisePreLoadParticles();
    /**
     * @brief not used function, but remain
     *
     */
    void InitialiseBoundary();
    /**
     * @brief initialize boundary particles with set coordinates
     *
     */

    void SetKernels();
    /**
     * @brief compute kernel function
     *
     */
    void SetTimeStep();
    /**
     * @brief set time step of simulation
     *
     */
    void EnableHighResolution();
    /**
     * @brief set parameters that related to resolution so that high resolution is used
     *
     */
    float kernelM4(float dist, float sr);
    /**
     * @brief kernel setting
     *
     * @param[in] float dist    kernel distance
     * @param[in] float sr    smooth radius to compute
     * @return float    computed kernel
     */
    Vector3DI GetCell(const Vector3DF& position);
    /**
     * @brief get cell position with input position
     *
     * @param[in] const Vector3DF&    input position
     * @param[in] float sr    smooth radius to compute
     *
     * @return Vector3DI    cell of particle position
     */
    void RefineGridResolution();
    /**
     * @brief set grid resolution
     *
     */
    void AllocateGrid();
    /**
     * @brief allocate grid memory
     *
     */
    int LoadPointCloud(const char* pointdata, float*& load_array);
    /**
     * @brief load particle from outside
     *
     * @param[in] const char*    particle file path
     * @param[in,out] float*&    buffer to store particle information
     *
     * @return int  num of point load
     */
    void TransferToCUDA();
    /**
     * @brief transfer params and buffers to gpu
     *
     */
    void TransferFromCUDA();
    /**
     * @brief transfer params and buffers to cpu
     *
     */
    void TransferParticleToSolver();
    /**
     * @brief transfer params and buffers to gpu
     *
     */
    inline void tick(int label);
    /**
     * @brief time counting function,set start time
     *
     * @param[in] int    time counting id,corresponding to "tock" with same id
     *
     */
    inline void tock(const std::string& info, int label);
    /**
     * @brief end time counting and print fps information
     *
     * @param[in] const std::string&    counting message
     * @param[in] int    time counting id,corresponding to "tick" with same id
     *
     */

private:
    // * * * system parameters * * * //
    bool         m_is_init;     //!< flag of init
    SolverConfig m_config;      //!< config struct
    Object*      m_fluid;       //!< fluid object
    Object*      m_rigid;       //!< rigid object
    Object*      m_gas;         //!< gas object
    Object*      m_boundary;    //!< boundary object
    double       m_cur_time;    //!< current time
    double       m_dt;          //!< time step that used
    double       m_total_time;  // total simulation time
    // parameters
    float     m_param[MAX_PARAM_NUM];            //!< float type param array
    Vector3DI m_vec_int[MAX_PARAM_NUM];          //!< int vector type param array
    Vector3DF m_vec[MAX_PARAM_NUM];              //!< float vector type param array
    bool      m_flag[MAX_PARAM_NUM];             //!< bool flag param array
    float     m_multi_phase[MAX_PARAM_NUM];      //!< float type multiphase param array
    float3    m_multi_phase_vec[MAX_PARAM_NUM];  //!< float vector type param array
    int       m_iteration;                       //!< counting of step
    int       m_output_frame;                    //!< render output frame num

    // particle attributes
    float*     m_position_drawn;        //!< render used array
    float*     m_position_drawn_rigid;  //!< render used array
    float*     m_position_point_cloud;  //!< render used array
    Vector3DF* m_position;              //!< position of particles
    Vector3DF* m_external_force;              //!< external force of particles
    Vector3DF* m_velocity;              //!< velocity of particles
    Vector3DF* m_mix_velocity;          //!< muti-phase velocity
    Vector3DF* m_acceleration;          //!< acceleration of particles
    Vector3DF* m_mforce;                //!< muti-phase force
    float3*    m_color;                 //!< particle color
    float*     m_density;               //!< particle density
    float*     m_pressure;              //!< particle pressure
    float*     m_mix_pressure;          //!< muti-phase mix pressure
    float*     m_alpha;                 //!< muti-phase volume fraction
    float*     m_alpha_advanced;        //!< alpha after advance
    float*     m_restmass;              //!< rest mass of mixed particles
    float*     m_mrest_density;         //!< mix rest density   rho_m
    float*     m_eff_v;                 //!< effective volume
    float*     m_smooth_raidus;         //!< smooth radius  h
    float*     m_particle_radius;       //!< particle radius r
    float*     m_delta_mass;            //!< delta mass
    float*     m_delta_mass_k;          //!< delta mass in each phase
    float*     m_delta_alpha_k;         //!< delta alpha in each phase
    float*     m_rest_mass_k;           //!< rest mass in each phase
    float*     m_surface_scalar_field;  //!< scalar field for surface tracking
    float*     m_concentration;         //!< consentration for render

    Vector3DF* m_center;  //!< center of rigid

    int* m_type;       //!< type pf particles
    int* m_phase;      //!< phase of particles
    int* m_explosion;  //!< attribute for explosion effect
    int* m_lock;       //!< particles locked can not move

    bool* m_active;  //!< particle active is able to move
    bool* m_render;  //!< decide particle render or not
    bool* m_rotate;  //!< rotate or not
    bool* m_mix;     //!< mix == false means no alpha change

    // bubble
    int*    m_idx_to_list_idx;  //!< for bubble simulation
    int*    m_bubble_list;      //!< for bubble simulation
    float3* m_bubble_pos_list;  //!< for bubble simulation

    // particles' global
    int m_num_particle;      //!< num of all particle
    int m_max_num_particle;  //!< max num of particle
    int m_num_fluid;         //!< num of fluid
    int m_num_bound;         //!< num of boundary
    int m_num_rigid;         //!< num of rigid
    int m_num_bubble;        //!< num of bubbles
    int m_test_index;        //!< for test
    int m_unit_size;         //!< for render element

    static int const m_lut_size = 100000;                     //!< kernel function param
    float            m_lut_kernel_m4[m_lut_size];             //!< kernel function param
    float            m_lut_kernel_pressure_grad[m_lut_size];  //!< kernel function param

    float m_time_step;  //!< time step used

    // * * * accelerated data structure * * *//

    // grid
    uint* m_grid_particle_table;      //!< grid particle table for neighbor search
    uint* m_next_particle_index;      //!< point to next particle in the same cell
    uint* m_num_particle_grid;        //!< num of particles in each grid
    uint* m_particle_grid_cellindex;  //!< cell index that particle in
    int*  m_grid_search_offset;       //!< neighbor search offset
    int   m_grid_number;              //!< number of grid

    // mc grid
    int3   m_mc_grid_resolution;  //!< marching cube grid resolution, for solid
    float3 m_mc_grid_min;         //!< min coordinate of mc grid
    float3 m_mc_grid_max;         //!< max coordinate of mc grid
    float  m_mc_grid_radius;      //!< mc grid radius
    int    m_mc_grid_number;      //!< number of mc grid
    int    m_mc_grid_ver_number;  //!< mc grid number with one unit added at x,y,z

    float* m_mc_scalar_value_field;  //!< mc scalar value field needed transfer to cuda
    float* m_mc_color_field;         //!< mc color field needed transfer to cuda
    int*   m_mc_grid_ver_idx;        //!< index array of mc grid

    // time
    TimePoint m_last_time_point[3];  //!< for time clock label
};

}  // namespace Physika