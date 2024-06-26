/*
 * @Author: pgpgwhp 1213388412@qq.com
 * @Date: 2023-09-19 15:47:53
 * @LastEditors: pgpgwhp 1213388412@qq.com
 * @LastEditTime: 2023-11-16 18:38:22
 * @FilePath: \physika\src\pbd-elastic\pbd_elastic_solver.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef ELASTIC_SOLVER_H
#define ELASTIC_SOLVER_H

#include <iostream>
#include <vector>
#include <string>

#include "framework/solver.hpp"
#include "include/mat3.cuh"

namespace Physika {

    struct ElasticSolverParams;

    enum PARTICLE_PHASE
    {
        SOLID         = 0,
        FIRSTSURFACE  = 1,
        SECONDSURFACE = 2,
        RIGID         = 3,
    };

    struct ElasticComponent
    {
        bool m_bInitialized;  // define if the component is initialized

        std::vector<float>        m_host_pos; // store position of particles
        std::vector<float>        m_host_vel;  // store velocity of particles
        std::vector<float>         m_host_phase;            // mark particle phase , 0:solid, 1:first surface, 2:second surface
        std::vector<unsigned int> m_host_index; // store particle index
        std::vector<float>         m_host_external_force; // store external force

        std::vector<int> m_host_neighbour_vector;  // store particle neighbour vector
        std::vector<int> m_host_neighbour_vector_start; // store particle neighbour vector start

        float* m_device_pos; // store device position of particles
        float* m_device_external_force; // store the external force of other objects
        float* m_device_next_pos; // store device next position of particles
        float* m_device_delte_pos; // store device delta position of particles
        float* m_device_vel; // store device velocity of particles
        float* m_device_initial_pos; // store device initial position of particles
        float* m_device_energy; // store device energy of particles
        unsigned int* m_device_index; // store device index of particles
        float*        m_device_phase; // mark particle phase, 0:solid, 1:first surface, 2:second surface
        unsigned int* m_device_cell_start; // store device cell start
        unsigned int* m_device_cell_end; // store device cell end
        unsigned int* m_device_grid_particle_hash; // store device grid particle hash

        float* m_device_lm; // store device lagrangian multiplier for XPBD
        float* m_device_delte_lm; // store device delta lagrangian multiplier for XPBD

        mat3* m_device_F; // store the device deformation gradient
        mat3* m_device_P; // store the device PK1
        mat3* m_device_R; // store the device rotation matrix

        // precompute
        float* m_device_sum; // store the device sum of kernel
        float3* m_device_y; // store the device y of kernel 
        mat3*  m_device_L; // store the device corrected kernel
        float* m_device_sph_kernel; // store the device sph kernel value of neighbor
        float* m_device_sph_kernel_inv; //  store the device inverse sph kernel value of neighbor

        // the initial neighbour
        unsigned int* m_device_neighbour_vector; // store the device neighbour of particles
        unsigned int* m_device_neighbour_vector_start; // store the device start neighbour of particles

        ElasticSolverParams* m_params; // some parameters of the elastic component
        unsigned int                  m_num_particle; // the number of particles in the component

        ElasticComponent();
        ElasticComponent(float radius, float3 lb, float3 rt); 
        ~ElasticComponent();

        // four elastic boxs collision demo
        void demo1(); 
        //  push anisotropy elastic box
        void demo2(); 
        // stretch anisotropy elastic box
        void demo3();
        // the anisotropy elastic box fall into a rigid cuboid
        void demo4();
        
        /**
         *  @brief add instance of elastic component before simulation
         * 
         *  @param category which category to create instances
         *  @param radius the radius of particle
         *  @param lb left button of the cube
         *  @param rt right top of the cube
         */
        void addInstance(float category, float radius, float3 lb, float3 rt);
        
        /**
         *  @brief add input instance vector of elastic component before simulation
         *  @param pos  the input position vector
         *  @param phase  the input phase vector
         */
        void addParticlePosition(std::vector<float>& pos, std::vector<float>& phase);

        /**
         * @brief initialize elastic component before simulation
         * 
         * @param numParticles  the number of particles in the component
         */
        void initialize(int numParticles);
        
        // not test temporarily
        /**
         * @brief reset the attribute of elastic component
         *  
         */
        void reset();

        /**
         * @brief free the memory of elastic component
         * 
         */
        void free();

        /**
         * @brief initial the neighbour of particles in the component
         * 
         * @param pos 
         * @param num_particle 
         */
        void initialNeighbour(std::vector<float> pos, int num_particle);
        
         /**
         * @brief add cube to the rigid component
         *
         * @param lb left bottom corner
         * @param rt right top corner
         * @param position position of the cube
         * @param velocity velocity of the cube
         * @param phase of the cube
         */
        void addRigid(float3 lb, float3 rt, std::vector<float>& position, std::vector<float>& velocity, std::vector<float>& phase);
        /**
         * @brief add cube to the elastic component
         * 
         * @param lb left bottom corner
         * @param rt right top corner
         * @param position position of the cube
         * @param velocity velocity of the cube 
         * @param phase of the cube 
         */
        void addCube(float3 lb, float3 rt, std::vector<float>& position, std::vector<float>& velocity, std::vector<float>& phase);
    };

    class ElasticParticleSovler : public Solver
    {
    public:
        struct SolverConfig
        {
            double       m_dt{ 1.f / 60.f };              // Time step
            double       m_total_time{ 5.f };              // total time of simulation
            float        m_stiffness{ 1.0f };              // Stiffness constant for the pressure forces in the SPH simulation
            float3       m_gravity = { 0.f, -9.8f, 0.f };  // The acceleration due to gravity in the simulation
            float        m_sleep_threshold{ 0.02f };       // Velocity threshold below which particles are considered as 'sleeping'
            unsigned int m_solver_iteration{ 2 };          // Maximum number of iterations for the constraint solvera
        };
        
        ElasticParticleSovler();

        ~ElasticParticleSovler();

        /**
         * @brief initialize the granular solver to get it ready for execution.
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
         * @brief set elastic particles' positions
         *
         * @param[in] host data of particle position array
         * @return true if position data is successfully set
         */
        bool setParticlePosition(const std::vector<float>& position);

        /**
         * @brief set elastic particles' velocity
         *
         * @param[in] host data of particle velocity array
         * @return true if velocity data is successfully set
         */
        bool setParticleVelocity(const std::vector<float>& velocity);


        /**
         * @brief set elastic particles' phase for collision detection
         *        if want to perform elastic collision , you need to set the correct particle phase: 
         *        SOLID 0 ， FIRST_SURFACE 1， SECOND_SURFACE 2 , RIGID = 3
         *
         * @param[in] host data of particle phase array
         * @return true if phase data is successfully set
         */
        bool setParticlePhase(const std::vector<float>& phase);


         /**
         * @brief set elastic particles' external force
         *
         * @param[in] host data of particle external force array
         * @return true if external force data is successfully set
         */
        bool setParticleExternalForce(const std::vector<float>& velocity);

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
         * @brief get the device pointer of particle velocity
         *
         * float* particleVelocityPtr = getParticleVelocityPtr();
         * data layout: v.x, v.y, v.z;
         *
         * @return the device pointer of particle velocity.
         *         nullptr if no valid object/component is attached to this solver.
         */
        float* getParticleVelocityPtr();

          /**
         * @brief get the device pointer of particle phase
         *
         * float* particlePhasePtr = getParticlePhasePtr();
         * data layout:0, 1, 2;
         *
         * @return the device pointer of particle phase.
         *         nullptr if no valid object/component is attached to this solver.
         */
        float* getParticlePhasePtr();

        /**
         * @brief get the granular particle radius. (eg: used for rendering or collision detect)
         *
         * @param[out] particleRadius : the radius of granular particle.
         *
         */
        void getParticleRadius(float& particleRadius);

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
         * @brief set the stitffness coefficient of position based dynamics solver
         * If the stiffness is too large, simulation will be crash.
         *
         * @param[in] stiffness : Stiffness coefficient that determines the displacement of PBD particles
         * The value of the static Stiffness coefficient should be between (0, 1].
         * Settings outside this range are invalid and and will be clamped to this range.
         */
        void setStiffness(float& stiffness);

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
         * @brief Set the Young Modules of the elastic component
         * 
         * @param young_modules 
         */
        void setYoungModules(const float& young_modules);

        /**
         * @brief Set the Possion Ratio of the elastic component
         * 
         * @param possion_ratio 
         */
        void setPossionRatio(const float& possion_ratio);

         /**
         * @brief Get the Young Modules of the elastic component
         *
         * @param young_modules
         */
        void getYoungModules(float& young_modules);

        /**
         * @brief Get the Possion Ratio of the elastic component
         *
         * @param possion_ratio
         */
        void getPossionRatio(float& possion_ratio);

        /**
         * @brief whether write to ply
         *
         * @param true / false
         */
        void setWriteToPly(const bool write_to_ply);

         /**
         * @brief whether solid is Anisotropy
         *
         * @param true / false
         */
        void setAnisotropy(const float3 Anisotropy);

    protected:

        /**
         * @brief perform neighbour search
         *        do this after advect and before the simulation step
         */
        bool neighbourSearch();
        
        /**
         * @brief pre-compute SPH before the simulation 
         *        do this after attaching the component and finish initial neighbour search
         */ 
        bool preComputeSPH();

        /**
         * @brief destroy the solver
         */
        void _finalize();

        /**
         * @brief write the particles data to ply file
         */
        void writeToPly(const int& step_id);

        /**
         * @brief free particle memory of the solver and component 
         */
        void freeParticleMemory(const int& numParticles);

        /**
         * @brief malloc particle memory of the solver and component
         */
        void mallocParticleMemory(const int& numParticles);

    private:
        Object*      m_elastic_particle;
        bool         m_is_init;
        bool         m_write_ply;
        float3       m_anisotropy;
        double       m_cur_time;
        SolverConfig m_config;

    private:
        // GPU parameters
        // semi-lagrangian advect
        float*       m_host_pos;
        unsigned int m_num_particles;
    };

}  // namespace physika
#endif