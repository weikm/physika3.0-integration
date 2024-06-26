/*
 * @Author: pgpgwhp 1213388412@qq.com
 * @Date: 2023-09-19 15:48:09
 * @LastEditors: pgpgwhp 1213388412@qq.com
 * @LastEditTime: 2023-09-21 16:15:36
 * @FilePath: \physika\src\pbd-elastic\pbd_elastic_solver.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "pbd_elastic_solver.hpp"

#include <string>
#include <fstream>

#include "pbd_elastic_params.hpp"
#include "framework/object.hpp"

namespace Physika {
    
    extern void getLastCudaError(const char* errorMessage);
	
    extern void setParamters( ElasticSolverParams* p);

    extern void computeHash(
        unsigned int* gridParticleHash,
        float*        pos,
        float3        m_world_orgin,
        float3        m_cell_size,
        uint3          m_grid_num,
        int           numParticles);

    extern void sortParticles(
        unsigned int* deviceGridParticleHash,
        unsigned int* m_device_index,
        unsigned int  numParticles);

    extern void findCellRange(
        unsigned int* cellStart,
        unsigned int* cellEnd,
        unsigned int* gridParticleHash,
        unsigned int  numParticles,
        unsigned int  numCell);

    extern void calCollisionDeltex(
        float3*       m_deviceInitialPos,
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        float*        m_device_phase,
        unsigned int* m_device_index,
        unsigned int* cellStart,
        unsigned int* cellEnd,
        unsigned int* gridParticleHash,
        float         m_sph_radius,
        float3        m_world_orgin,
        float3        m_celll_size,
        uint3          m_grid_num,
        int           m_num_particle);

    extern void updateCollDeltx(
        float3*       m_deviceDeltex,
        float3*       m_deviceNextPos,
        unsigned int* m_device_index,
        int           m_num_particle);

    extern void update(
		float3* m_deviceNextPos,
		float3* m_devicePos,
		float3* m_deviceVel,
        float*  m_device_phase,
		float deltaT,
		int m_numParticles
	);

    extern void advect(
		float3* m_devicePos,
		float3* m_deviceNextPos,
		float3* m_deviceVel,
		float3* m_deviceDeltex,
        float3* m_deviceExternalForce,
        float*  m_device_phase,
		float3 acc,
		float deltaT,
		int m_numParticles
	);

    extern void solveBoundaryConstraint(
		float3* m_deviceNextPos,
		float3* m_deviceVel,
		float3 LB,
		float3 RT,
		float radius,
		int m_numParticle
	);

    extern void calL(
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float*        m_device_sum,
        mat3*         m_deivce_L,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_sph_radius,
        float         m_volume,
        int           m_num_particle); 

    extern void calY(
        float3*       m_device_initial_pos,
        float3*       m_deivce_y,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        float         m_sph_radius,
        int           m_num_particle);

    extern void calSumKernelPoly(
        float3*       m_device_initial_pos,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        float         m_sph_radius,
        int           m_num_particle);


    extern void calGradientDeformation(
        float3*       m_device_sph_kernel,
        float3*       m_device_next_pos,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float*        m_device_sum,
        mat3*         m_device_L,
        mat3*         m_device_F,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        int           m_num_particle);

    extern void calR(
		mat3* m_device_F,
		mat3* m_device_R,
		int m_num_particle
	);

   extern void calP(
        float3 m_anisotropy,
        mat3*  m_device_F,
        mat3*  m_device_R,
        mat3*  m_device_P,
        int    m_num_particle);

    extern void calEnergy(
        float3 m_anisotropy,
        mat3*  m_device_F,
        mat3*  m_device_R,
        float* m_device_energy,
        int    m_num_particle);


    extern void calLM(
		float* m_device_energy,
		float* m_device_lm,
		int m_num_particle
	);


    extern void solveDelteLagrangeMultiplier(
        float3*       m_device_sph_kernel,
        float*        m_device_energy,
        float*        m_device_lm,
        float*        m_device_delte_lm,
        float*        m_device_sum,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        mat3*         m_device_L,
        mat3*         m_device_P,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_h,
        float         m_volume,
        int           m_num_particle);

    extern void solveDelteDistance(
        float3*       m_device_sph_kernel,
        float3*       m_device_sph_kernel_inv,
        mat3*         m_device_P,
        mat3*         m_device_L,
        float*        m_device_delte_lm,
        float*        m_device_sum,
        float3*       m_device_delte_x,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_volume,
        int           m_num_particle);

    extern void updateDeltxlm(
        float3* m_deviceDeltex,
        float3* m_deviceNextPos,
        float* m_device_delte_lm,
        float* m_device_lm,
        int m_num_particle
    );

    extern void calCorrGradientKenrelSum(
        mat3*         m_device_L,
        float3*       m_device_initial_pos,
        float3*       m_device_y,
        float3*       m_device_sph_kernel,
        float3*       m_device_sph_kernel_inv,
        float3*       m_device_external_force,
        float*        m_device_sum,
        unsigned int* grid,
        unsigned int* gridStart,
        float         m_sph_radius,
        int           m_num_particle);


     ElasticComponent::ElasticComponent()
        : m_bInitialized(false), m_device_pos(nullptr), m_device_vel(nullptr), m_device_initial_pos(nullptr), m_device_energy(nullptr), m_device_F(nullptr), m_device_P(nullptr), m_device_R(nullptr), m_device_sum(nullptr), m_device_y(nullptr), m_device_L(nullptr), m_device_sph_kernel(nullptr), m_device_neighbour_vector(nullptr), m_device_neighbour_vector_start(nullptr), m_device_index(nullptr), m_device_phase(nullptr), m_device_cell_start(nullptr), m_device_cell_end(nullptr), m_device_grid_particle_hash(nullptr), m_device_delte_lm(nullptr), m_device_delte_pos(nullptr), m_device_lm(nullptr), m_device_next_pos(nullptr), m_device_sph_kernel_inv(nullptr)
    {
        m_num_particle                 = 0;
        m_params                       = new ElasticSolverParams();
       /* m_params->m_lb_boundary        = lb;
        m_params->m_rt_boundary        = rt;
        m_params->m_particle_radius    = radius;
        m_params->m_particle_dimcenter = 2 * radius;
        m_params->m_sph_radius         = 4 * radius;

        m_params->m_volume    = 4 / 3 * M_PI * pow(radius, 3);
        m_params->m_cell_size = make_float3(m_params->m_sph_radius, m_params->m_sph_radius, m_params->m_sph_radius);*/

        /* m_params->m_grid_num = { static_cast<unsigned int>((rt.x - lb.x) / m_params->m_sph_radius),
                                  static_cast<unsigned int>((rt.y - lb.y) / m_params->m_sph_radius),
                                  static_cast<unsigned int>((rt.z - lb.z) / m_params->m_sph_radius) };*/

        m_params->m_mass     = 1.f;
        m_params->m_invMass  = 1.f;
        m_params->m_grid_num = make_uint3(64, 64, 64);
        m_params->grid_size = m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z;
        m_params->m_world_orgin = make_float3(0.f, 0.f, 0.f);

        m_params->young_modules = 3000;
        m_params->possion_ratio = 0.2;
        m_params->lame_first    = m_params->young_modules * m_params->possion_ratio / ((1 + m_params->possion_ratio) * (1 - 2 * m_params->possion_ratio));
        m_params->lame_second   = m_params->young_modules / (2 * (1 + m_params->possion_ratio));

        // demo4();
        // initialize(m_num_particle);
    }


    
    ElasticComponent::ElasticComponent(float radius, float3 lb, float3 rt)
        : m_bInitialized(false),m_device_pos(nullptr), m_device_vel(nullptr), m_device_initial_pos(nullptr), m_device_energy(nullptr), m_device_F(nullptr), \
    m_device_P(nullptr),m_device_R(nullptr),m_device_sum(nullptr) ,m_device_y(nullptr),m_device_L(nullptr),m_device_sph_kernel(nullptr), m_device_neighbour_vector(nullptr),\
    m_device_neighbour_vector_start(nullptr), m_device_index(nullptr), m_device_phase(nullptr), m_device_cell_start(nullptr), m_device_cell_end(nullptr), m_device_grid_particle_hash(nullptr),\
    m_device_delte_lm(nullptr), m_device_delte_pos(nullptr), m_device_lm(nullptr), m_device_next_pos(nullptr), m_device_sph_kernel_inv(nullptr)
    {
        m_num_particle = 0;
        m_params = new ElasticSolverParams();
        m_params->m_lb_boundary = lb;
        m_params->m_rt_boundary = rt;
        m_params->m_particle_radius = radius;
        m_params->m_particle_dimcenter = 2 * radius;
        m_params->m_sph_radius = 4 * radius;

        m_params->m_mass      = 1.f;
        m_params->m_invMass   = 1.f;
        m_params->m_volume    = 4 / 3 * M_PI * pow(radius, 3);
        m_params->m_cell_size = make_float3(m_params->m_sph_radius, m_params->m_sph_radius, m_params->m_sph_radius);

       /* m_params->m_grid_num = { static_cast<unsigned int>((rt.x - lb.x) / m_params->m_sph_radius),
                                 static_cast<unsigned int>((rt.y - lb.y) / m_params->m_sph_radius),
                                 static_cast<unsigned int>((rt.z - lb.z) / m_params->m_sph_radius) };*/
        m_params->m_grid_num = make_uint3(64, 64, 64);    

        m_params->grid_size = m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z;

        m_params->m_world_orgin = make_float3(0.f, 0.f, 0.f);

        m_params->young_modules = 3000;
        m_params->possion_ratio = 0.2;
        m_params->lame_first = m_params->young_modules * m_params->possion_ratio / ((1 + m_params->possion_ratio) * (1 - 2 * m_params->possion_ratio));
		m_params->lame_second = m_params->young_modules / (2 * (1 + m_params->possion_ratio));

        //demo4();
        //initialize(m_num_particle);
    }

    ElasticComponent::~ElasticComponent() {
        // free gpu memory
        if (m_bInitialized)
        {
            m_bInitialized = false;
            printf("elastic component initialized false \n");
        }
    }

    void ElasticComponent::demo1() {
        float3 lb0 = make_float3(0.f, 0.0f, 0.f);
        float3 rt0 = make_float3(6.f, 8.f, 6.f);
        float3 lb1 = make_float3(-6.f, -7.0f, -6.f);
        float3 rt1 = make_float3(0.f, -1.f, 0.f);
        float3 lb2 = make_float3(-7.f, 0.0f, -7.f);
        float3 rt2 = make_float3(-1.f, 6.f, -1.f);
        float3 lb3 = make_float3(1.f, -9.0f, 1.f);
        float3 rt3 = make_float3(9.f, -1.f, 9.f);
        std::vector<float> m_host_pos_0;
        std::vector<float> m_host_vel_0;
        std::vector<float> m_host_phase_0;
        std::vector<float> m_host_pos_1;
        std::vector<float> m_host_vel_1;
        std::vector<float> m_host_phase_1;
        std::vector<float> m_host_pos_2;
        std::vector<float> m_host_vel_2;
        std::vector<float> m_host_phase_2;
        std::vector<float> m_host_pos_3;
        std::vector<float> m_host_vel_3;
        std::vector<float> m_host_phase_3;

        m_host_neighbour_vector_start.push_back(0);
        addCube(lb0, rt0, m_host_pos_0, m_host_vel_0, m_host_phase_0);
        initialNeighbour(
            m_host_pos_0,
            m_num_particle
            );
        m_num_particle += m_host_pos_0.size() / 3;
        printf("cube0 : %d \n", m_host_pos_0.size() / 3);
        addCube(lb1, rt1, m_host_pos_1, m_host_vel_1, m_host_phase_1);
        initialNeighbour(
            m_host_pos_1,
            m_num_particle
            );
        m_num_particle += m_host_pos_1.size() / 3;
        printf("cube1 : %d \n", m_host_pos_1.size() / 3);
        addCube(lb2, rt2, m_host_pos_2, m_host_vel_2, m_host_phase_2);
        initialNeighbour(
            m_host_pos_2,
            m_num_particle);
        m_num_particle += m_host_pos_2.size() / 3;
        printf("cube2 : %d \n", m_host_pos_2.size() / 3);
        addCube(lb3, rt3, m_host_pos_3, m_host_vel_3, m_host_phase_3);
        initialNeighbour(
            m_host_pos_3,
            m_num_particle);
        m_num_particle += m_host_pos_3.size() / 3;
        printf("cube3 : %d \n", m_host_pos_3.size() / 3);
        for (auto p : m_host_pos_0)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);
        }
        for (auto p : m_host_vel_0)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_0)
        {
            m_host_phase.push_back(p);
        }
        for (auto p : m_host_pos_1)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);
        }
        for (auto p : m_host_vel_1)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_1)
        {
            m_host_phase.push_back(p);
        }
        for (auto p : m_host_pos_2)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);
        }
        for (auto p : m_host_vel_2)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_2)
        {
            m_host_phase.push_back(p);
        }
        for (auto p : m_host_pos_3)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);     
        }
        for (auto p : m_host_vel_3)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_3)
        {
            m_host_phase.push_back(p);
        }
        for (int i = 0; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        std::vector<float>().swap(m_host_pos_0);
        std::vector<float>().swap(m_host_vel_0);
        std::vector<float>().swap(m_host_phase_0);
        std::vector<float>().swap(m_host_pos_1);
        std::vector<float>().swap(m_host_vel_1);
        std::vector<float>().swap(m_host_phase_1);
        std::vector<float>().swap(m_host_pos_2);
        std::vector<float>().swap(m_host_vel_2);
        std::vector<float>().swap(m_host_phase_2);
        std::vector<float>().swap(m_host_pos_3);
        std::vector<float>().swap(m_host_vel_3);
        std::vector<float>().swap(m_host_phase_3);
    }

    void ElasticComponent::demo2() {
        float3             lb4 = make_float3(0.f, -9.9f, 0.f);
        float3             rt4 = make_float3(9.f, 0.1f, 9.f);


        std::vector<float> m_host_pos_4;
        std::vector<float> m_host_vel_4;
        std::vector<float> m_host_phase_4;

        m_host_neighbour_vector_start.push_back(0);
        addCube(lb4, rt4, m_host_pos_4, m_host_vel_4, m_host_phase_4);
        initialNeighbour(
            m_host_pos_4,
            m_num_particle);
        m_num_particle += m_host_pos_4.size() / 3;
        printf("cube4 : %d \n", m_host_pos_4.size() / 3);

        for (auto p : m_host_pos_4)
        {
            m_host_pos.push_back(p);
        }
        for (auto p : m_host_vel_4)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_4)
        {
            m_host_phase.push_back(p);
        }
        for (int i = 0; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        std::vector<float>().swap(m_host_pos_4);
        std::vector<float>().swap(m_host_vel_4);
        std::vector<float>().swap(m_host_phase_4);
    }

    void ElasticComponent::demo3() {
        float3 lb4 = make_float3(0.f, -0.1f, 0.f);
        float3 rt4 = make_float3(9.f, 9.9f, 9.f);

        std::vector<float> m_host_pos_4;
        std::vector<float> m_host_vel_4;
        std::vector<float> m_host_phase_4;

        m_host_neighbour_vector_start.push_back(0);
        addCube(lb4, rt4, m_host_pos_4, m_host_vel_4, m_host_phase_4);
        initialNeighbour(
            m_host_pos_4,
            m_num_particle);
        m_num_particle += m_host_pos_4.size() / 3;
        printf("cube4 : %d \n", m_host_pos_4.size() / 3);

        for (auto p : m_host_pos_4)
        {
            m_host_pos.push_back(p);
        }
        for (auto p : m_host_vel_4)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_4)
        {
            m_host_phase.push_back(p);
        }
        for (int i = 0; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        std::vector<float>().swap(m_host_pos_4);
        std::vector<float>().swap(m_host_vel_4);
        std::vector<float>().swap(m_host_phase_4);
    }

    void ElasticComponent::demo4()
    {
        float3 lb4 = make_float3(-5.f, 0.f, -5.f);
        float3 rt4 = make_float3(5.f, 2.f, 5.f);
        float3 l0 = make_float3(-1.f, -9.9f, -1.f);
        float3 r0 = make_float3(1.f, -2.f, 1.f);

        std::vector<float> m_host_pos_4;
        std::vector<float> m_host_vel_4;
        std::vector<float> m_host_phase_4;

        std::vector<float> m_host_pos_5;
        std::vector<float> m_host_vel_5;
        std::vector<float> m_host_phase_5;

        m_host_neighbour_vector_start.push_back(0);
        addCube(lb4, rt4, m_host_pos_4, m_host_vel_4, m_host_phase_4);
        initialNeighbour(
            m_host_pos_4,
            m_num_particle);
        m_num_particle += m_host_pos_4.size() / 3;
        printf("cube4 : %d \n", m_host_pos_4.size() / 3);

        addRigid(l0, r0, m_host_pos_5, m_host_vel_5, m_host_phase_5);
        initialNeighbour(
            m_host_pos_5,
            m_num_particle);
        m_num_particle += m_host_pos_5.size() / 3;
        printf("cube5 : %d \n", m_host_pos_5.size() / 3);

        for (auto p : m_host_pos_4)
        {
            m_host_external_force.push_back(0.f);
            m_host_pos.push_back(p);
        }
        for (auto p : m_host_vel_4)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_4)
        {
            m_host_phase.push_back(p);
        }
        for (auto p : m_host_pos_5)
        {
            m_host_external_force.push_back(0.f);
            m_host_pos.push_back(p);
        }
        for (auto p : m_host_vel_5)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_host_phase_5)
        {
            m_host_phase.push_back(p);
        }
        for (int i = 0; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        std::vector<float>().swap(m_host_pos_4);
        std::vector<float>().swap(m_host_vel_4);
        std::vector<float>().swap(m_host_phase_4);
        std::vector<float>().swap(m_host_pos_5);
        std::vector<float>().swap(m_host_vel_5);
        std::vector<float>().swap(m_host_phase_5);
    }

    void ElasticComponent::addInstance(float category, float radius, float3 lb, float3 rt)
    {
        std::vector<float> m_pos;
        std::vector<float> m_vel;
        std::vector<float> m_phase;

        if (m_num_particle == 0)
            m_host_neighbour_vector_start.push_back(0);
        int start = m_num_particle;
        if (category == PARTICLE_PHASE::RIGID)
        {
            addRigid(lb, rt, m_pos, m_vel, m_phase);
            initialNeighbour(
                m_pos,
                m_num_particle);
            m_num_particle += m_pos.size() / 3;
            printf("create rigid : %d \n", m_pos.size() / 3);
        }
        else if (category == PARTICLE_PHASE::SOLID)
        {
            addCube(lb, rt, m_pos, m_vel, m_phase);
            printf("m_num_particle: %d \n", m_num_particle);
            initialNeighbour(
                m_pos,
                m_num_particle);
            m_num_particle += m_pos.size() / 3;
            printf("create cube : %d \n", m_pos.size() / 3);
        }
        else
        {
            printf("create error ! \n");
            return;
        }

        for (auto p : m_pos)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);
        }
        for (auto p : m_vel)
        {
            m_host_vel.push_back(p);
        }
        for (auto p : m_phase)
        {
            m_host_phase.push_back(p);
        }
    
        for (int i = start; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        printf("Create Instance successfully !\n");
        std::vector<float>().swap(m_pos);
        std::vector<float>().swap(m_vel);
        std::vector<float>().swap(m_phase);
    }


    void ElasticComponent::addParticlePosition(std::vector<float>& pos, std::vector<float>& phase)
    {
        std::vector<float> m_pos;
        std::vector<float> m_phase;
        int                start = m_num_particle;
        if (m_num_particle == 0)
            m_host_neighbour_vector_start.push_back(0);
        
        m_pos = std::move(pos);
        m_phase = std::move(phase);
        initialNeighbour(
            m_pos,
            m_num_particle);
        m_num_particle += m_pos.size() / 3;

        for (auto p : m_pos)
        {
            m_host_pos.push_back(p);
            m_host_external_force.push_back(0.f);
            m_host_vel.push_back(0.f);
        }
        for (auto p : m_phase)
        {
            m_host_phase.push_back(p);
        }
        for (int i = start; i < m_num_particle; i++)
        {
            m_host_index.push_back(i);
        }
        printf("Create Instance successfully !\n");
        std::vector<float>().swap(m_pos);
        std::vector<float>().swap(m_phase);

    }

    void ElasticComponent::addRigid(float3 lb, float3 rt, std::vector<float>& position, std::vector<float>& velocity, std::vector<float>& phase)
    {
        if (m_bInitialized)
        {
            std::cout << "Already initialized.\n";
            return;
        }
        float deltx = rt.x - lb.x;
        float delty = rt.y - lb.y;
        float deltz = rt.z - lb.z;
        int   sum   = 0;
        int3  delte = { int(deltx / m_params->m_particle_dimcenter), int(delty / m_params->m_particle_dimcenter), int(deltz / m_params->m_particle_dimcenter) };
        for (int i = 0; i < delte.x; i++)
        {
            for (int j = 0; j < delte.y; j++)
            {
                for (int k = 0; k < delte.z; k++)
                {
                    float3 pos_ = make_float3(lb.x + m_params->m_particle_dimcenter / 2 + i * m_params->m_particle_dimcenter,
                                              lb.y + m_params->m_particle_dimcenter / 2 + j * m_params->m_particle_dimcenter,
                                              lb.z + m_params->m_particle_dimcenter / 2 + k * m_params->m_particle_dimcenter);

                   
                    phase.push_back(static_cast<float>(Physika::PARTICLE_PHASE::RIGID));
                   

                    position.push_back(pos_.x);
                    position.push_back(pos_.y);
                    position.push_back(pos_.z);

                    velocity.push_back(0.0f);
                    velocity.push_back(0.0f);
                    velocity.push_back(0.0f);
                    sum++;
                    // printf("number %d", i * 100 + 10 * j + k);
                }
            }
        }
    }

    void ElasticComponent::addCube(float3 lb, float3 rt, std::vector<float>& position, std::vector<float>& velocity, std::vector<float>& phase)
    {
        if(m_bInitialized) {
            std::cout << "Already initialized.\n";
            return;
        }       
        float deltx = rt.x - lb.x;
        float delty = rt.y - lb.y;
        float deltz = rt.z - lb.z;	
        int sum = 0;
        int3  delte = { int(deltx / m_params->m_particle_dimcenter), int(delty / m_params->m_particle_dimcenter), int(deltz / m_params->m_particle_dimcenter) };
        for (int i = 0; i < delte.x; i++)
        {
            for (int j = 0; j < delte.y; j++)
            {
                for (int k = 0; k < delte.z; k++)
                {
                    float3 pos_ = make_float3(lb.x  + m_params->m_particle_dimcenter / 2 + i * m_params->m_particle_dimcenter,
                        lb.y + m_params->m_particle_dimcenter / 2 + j * m_params->m_particle_dimcenter,
                        lb.z + m_params->m_particle_dimcenter / 2 + k * m_params->m_particle_dimcenter
                    );
                    
                    if (i == 0 || i == delte.x - 1 || j == 0 || j == delte.y - 1 || k == 0 || k == delte.z - 1)
                    {
                        phase.push_back(static_cast<float>(Physika::PARTICLE_PHASE::FIRSTSURFACE));
                    }
                    else if (i == 1 || i == delte.x - 2 || j == 1 || j == delte.y - 2 || k == 1 || k == delte.z - 2)
                    {
                        phase.push_back(static_cast<float>(Physika::PARTICLE_PHASE::SECONDSURFACE));
                    }
                    else
                    {
                        phase.push_back(static_cast<float>(Physika::PARTICLE_PHASE::SOLID));
                    }

                    position.push_back(pos_.x);
                    position.push_back(pos_.y);
                    position.push_back(pos_.z);

                    velocity.push_back(0.0f);
                    velocity.push_back(0.0f);
                    velocity.push_back(0.0f);
                    sum ++;
                    //printf("number %d", i * 100 + 10 * j + k);
                }
            }
	    }
    }

    void ElasticComponent::initialNeighbour(std::vector<float> pos, int num_particle)
    {

        if(m_bInitialized) {
            std::cout << "Already initialized.\n";
            return;
        } 

        int particle_count = pos.size() / 3;
        int number = 0;
        int gridsize = m_host_neighbour_vector.size();
        std::vector<int> m_grid_offset(m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z);
        std::vector<int> m_grid_start(m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z + 1);
        std::vector<int> m_grid(particle_count);
        float grid_size = m_params->m_sph_radius;

       for (int i = 0; i < particle_count; i++)
        {
            number = int((pos[3 * i + 2] - m_params->m_lb_boundary.z) / grid_size)
                + int((pos[3 * i + 1] - m_params->m_lb_boundary.y) / grid_size)* m_params->m_grid_num.z
                + int((pos[3 * i] - m_params->m_lb_boundary.x) / grid_size)* m_params->m_grid_num.z * m_params->m_grid_num.y;
            m_grid_offset[number] ++;
        }

        m_grid_start[0] = 0;
        for (int i = 1; i <= m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z; i++) {
            m_grid_start[i] = m_grid_offset[i - 1] + m_grid_start[i - 1];
            m_grid_offset[i - 1] = 0;
        }
        m_grid_offset[m_params->m_grid_num.x * m_params->m_grid_num.y * m_params->m_grid_num.z - 1] = 0;

        for (int i = 0; i < particle_count; i++)
        {
            number = int((pos[3 * i + 2] - m_params->m_lb_boundary.z) / grid_size)
                + int((pos[3 * i + 1] - m_params->m_lb_boundary.y) / grid_size) * m_params->m_grid_num.z
                + int((pos[3 * i] - m_params->m_lb_boundary.x) / grid_size) * m_params->m_grid_num.z * m_params->m_grid_num.y;

            m_grid[m_grid_start[number] + m_grid_offset[number]] = i;
            m_grid_offset[number] ++;
        }

        int sum = 0;
        for (int i = 0; i < particle_count; i++)
        {
            int3 curGrid = make_int3(int((pos[3 * i] - m_params->m_lb_boundary.x) / grid_size),
            int((pos[3 * i + 1] - m_params->m_lb_boundary.y) / grid_size),
            int((pos[3 * i + 2] - m_params->m_lb_boundary.z) / grid_size));
#pragma omp parallel 
            {
                for (int x = -1; x < 2; x++) {
#pragma omp parallel 
                    {
                        for (int y = -1; y < 2; y++) {
#pragma omp parallel 
                            {
                                for (int z = -1; z < 2; z++) {
                                    int3 offsetGrid = make_int3(curGrid.x + x, curGrid.y + y, curGrid.z + z);

                                    if (offsetGrid.x < 0 || offsetGrid.y < 0 || offsetGrid.z < 0 || offsetGrid.x >= m_params->m_grid_num.x \
                                        || offsetGrid.y > m_params->m_grid_num.y || offsetGrid.z >= m_params->m_grid_num.z)
                                        continue;
                                    int offset = offsetGrid.z + 
                                        + offsetGrid.y * m_params->m_grid_num.z
                                        + offsetGrid.x * m_params->m_grid_num.z * m_params->m_grid_num.y;

                                    int start = m_grid_start[offset];
                                    int end = m_grid_start[offset + 1];
#pragma omp parallel 
                                    {
                                        for (int indexj = start; indexj < end; indexj++) {
                                            int j = m_grid[indexj];
                                            if (i == j)
                                                continue;
                                            if ((pos[3 * i] - pos[3 * j]) * (pos[3 * i] - pos[3 * j]) + \
                                                (pos[3 * i + 1] - pos[3 * j + 1]) * (pos[3 * i + 1] - pos[3 * j + 1]) + \
                                                (pos[3 * i + 2] - pos[3 * j + 2]) * (pos[3 * i + 2] - pos[3 * j + 2]) < 4 * m_params->m_particle_dimcenter * m_params->m_particle_dimcenter) 
                                            {
                                                m_host_neighbour_vector.push_back(num_particle + j);
                                                sum++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            m_host_neighbour_vector_start.push_back(gridsize + sum);
        }
    }

    void ElasticComponent::initialize(int num_particle) {
        if (m_bInitialized)
        {
            std::cout << "Already initialized.\n";
            return;
        }
        unsigned int memMatSize    = sizeof(mat3) * m_num_particle;
        unsigned int memFloatSize  = sizeof(float) * m_num_particle;
        unsigned int memFloat3Size = sizeof(float) * 3 * m_num_particle; 
		unsigned int menIntSize    = sizeof(int) * m_num_particle;
        unsigned int menUIntSize    = sizeof(unsigned int) * m_num_particle;
        unsigned int menGridSize   = sizeof(unsigned int) * m_params->grid_size;

        cudaMalloc((void**)&m_device_pos, memFloat3Size);
        cudaMalloc(( void** )&m_device_external_force, memFloat3Size);
        cudaMalloc((void**)&m_device_next_pos, memFloat3Size);
        cudaMalloc((void**)&m_device_delte_pos, memFloat3Size);
        cudaMalloc((void**)&m_device_initial_pos, memFloat3Size);
        cudaMalloc((void**)&m_device_vel, memFloat3Size);
        cudaMalloc((void**)&m_device_sum, memFloatSize);
		cudaMalloc((void**)&m_device_lm, memFloatSize);
        cudaMalloc((void**)&m_device_delte_lm, memFloatSize);
		cudaMalloc((void**)&m_device_y, memFloat3Size);
		cudaMalloc((void**)&m_device_L, memMatSize);
		cudaMalloc((void**)&m_device_F, memMatSize);
		cudaMalloc((void**)&m_device_P, memMatSize);
        cudaMalloc((void**)&m_device_R, memMatSize);
		cudaMalloc((void**)&m_device_energy, memFloatSize);
        cudaMalloc(( void** )&m_device_index, menUIntSize);
        cudaMalloc(( void** )&m_device_phase, memFloatSize);
        cudaMalloc(( void** )&m_device_cell_start, menGridSize);
        cudaMalloc(( void** )&m_device_cell_end, menGridSize);
        cudaMalloc(( void** )&m_device_grid_particle_hash, menUIntSize);
        getLastCudaError("component malloc!");

        cudaMemcpy(( char* )m_device_index, m_host_index.data(), m_num_particle * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_phase, m_host_phase.data(), m_num_particle * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_external_force, m_host_external_force.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)m_device_next_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_initial_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_vel, m_host_vel.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        getLastCudaError("component memcpy!");

        int length       = m_host_neighbour_vector.size();
        int length_start = m_host_neighbour_vector_start.size();
        cudaMalloc(( void** )&m_device_neighbour_vector, length * sizeof(unsigned int));
        cudaMalloc(( void** )&m_device_neighbour_vector_start, length_start * sizeof(unsigned int));
        cudaMalloc(( void** )&m_device_sph_kernel, length * sizeof(float) * 3);
        cudaMalloc(( void** )&m_device_sph_kernel_inv, length * sizeof(float) * 3);
        cudaMemcpy(( char* )m_device_neighbour_vector, m_host_neighbour_vector.data(), length * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_neighbour_vector_start, m_host_neighbour_vector_start.data(), length_start * sizeof(unsigned int), cudaMemcpyHostToDevice);
        getLastCudaError("grid malloc!");

        m_bInitialized = true;
        printf("Particle initialize successfully!~");
        
    }

    void ElasticComponent::free() {

        std::vector<float>().swap(m_host_pos);
        std::vector<float>().swap(m_host_vel);
        std::vector<float>().swap(m_host_phase);
        std::vector<unsigned int>().swap(m_host_index);
        std::vector<int>().swap(m_host_neighbour_vector);
        std::vector<int>().swap(m_host_neighbour_vector_start);       

        delete this->m_params;

        m_bInitialized = false;

        cudaFree(m_device_pos);
        cudaFree(m_device_external_force);
        cudaFree(m_device_next_pos);
		cudaFree(m_device_vel);
		cudaFree(m_device_initial_pos);
        cudaFree(m_device_delte_pos);
		cudaFree(m_device_neighbour_vector);
		cudaFree(m_device_neighbour_vector_start);
		cudaFree(m_device_sph_kernel_inv);
		cudaFree(m_device_sph_kernel);
		cudaFree(m_device_sum);
        cudaFree(m_device_delte_lm);
		cudaFree(m_device_lm);
		cudaFree(m_device_y);
		cudaFree(m_device_L);
		cudaFree(m_device_F);
		cudaFree(m_device_P);
		cudaFree(m_device_energy);
		cudaFree(m_device_R);
        cudaFree(m_device_index);
        cudaFree(m_device_phase);
        cudaFree(m_device_cell_start);
        cudaFree(m_device_cell_end);
        cudaFree(m_device_grid_particle_hash);
        getLastCudaError("cuda free error! \n");
        printf("Free all the attribute! \n");
    }

    void ElasticComponent::reset() { 
        cudaMemcpy(( char* )m_device_index, m_host_index.data(), m_num_particle * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_phase, m_host_phase.data(), m_num_particle * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_next_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_initial_pos, m_host_pos.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_vel, m_host_vel.data(), m_num_particle * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_neighbour_vector, m_host_neighbour_vector.data(), m_host_neighbour_vector.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(( char* )m_device_neighbour_vector_start, m_host_neighbour_vector_start.data(), m_host_neighbour_vector_start.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        
    }

    ElasticParticleSovler::ElasticParticleSovler()
        : m_is_init(false), m_write_ply(true), m_elastic_particle(nullptr), m_cur_time(0.f), m_num_particles(0), m_host_pos(nullptr)
    {
        m_anisotropy = make_float3(1, 1, 1);
    }

    ElasticParticleSovler::~ElasticParticleSovler() {
        if(m_is_init) {
            _finalize();
        }
    }

    bool ElasticParticleSovler::initialize() {
        if (m_is_init)
            return true;
        if (m_elastic_particle == nullptr)
        {
            std::cout << "ERROR: Must set elastic particle object first.\n";
            return false;
        }
        if (m_elastic_particle->hasComponent<ElasticComponent>() == false)
        {
            std::cout << "ERROR: elastic particle object has no elastic component.\n";
            return false;
        }
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
        {
            std::cout << "ERROR: no elastic component.\n";
            return false;
        }
        mallocParticleMemory(m_num_particles);
        preComputeSPH();

        m_is_init = true;
        std::cout << "elastic solver initialized successfully.\n";
        return true;
    }

    bool ElasticParticleSovler::isInitialized() const {
        return m_is_init;
    }

    bool ElasticParticleSovler::reset() {
        m_is_init             = false;
        m_config.m_dt         = 0.0;
        m_config.m_total_time = 0.0;
        m_elastic_particle   = nullptr;
        m_cur_time            = 0.0;
        if(m_elastic_particle != nullptr) {
            m_elastic_particle->getComponent<ElasticComponent>()->reset();
        }
        preComputeSPH();
        return true;
    }

    bool ElasticParticleSovler::preComputeSPH() {

        if (!m_elastic_particle->hasComponent<ElasticComponent>())
        {
            std::cout << "no component.\n";
            return false;
        }
        ElasticComponent* elastic_compenent = m_elastic_particle->getComponent<ElasticComponent>();

        if (elastic_compenent->m_bInitialized == false)
        {
            std::cout << "already initialized. \n";
            return false; 
        }

        if (elastic_compenent == nullptr)
            return false;
        setParamters(elastic_compenent->m_params);
              
        calY(( float3* )elastic_compenent->m_device_initial_pos,
			(float3*)elastic_compenent->m_device_y,
			elastic_compenent->m_device_neighbour_vector,
			elastic_compenent->m_device_neighbour_vector_start,
             elastic_compenent->m_params->m_volume,
             elastic_compenent->m_params->m_sph_radius,
			elastic_compenent->m_num_particle
		);

		calSumKernelPoly(
			(float3*)elastic_compenent->m_device_initial_pos,
			elastic_compenent->m_device_sum,
			elastic_compenent->m_device_neighbour_vector,
			elastic_compenent->m_device_neighbour_vector_start,
            elastic_compenent->m_params->m_volume,
            elastic_compenent->m_params->m_sph_radius,
			elastic_compenent->m_num_particle
		);

		calL(( float3* )elastic_compenent->m_device_initial_pos,
			(float3*)elastic_compenent->m_device_y,
			elastic_compenent->m_device_sum,
			elastic_compenent->m_device_L,
			elastic_compenent->m_device_neighbour_vector,
			elastic_compenent->m_device_neighbour_vector_start,
            elastic_compenent->m_params->m_sph_radius,
            elastic_compenent->m_params->m_volume,
			elastic_compenent->m_num_particle
		);

		calCorrGradientKenrelSum(
			elastic_compenent->m_device_L,
			(float3*)elastic_compenent->m_device_initial_pos,
			(float3*) elastic_compenent->m_device_y,
			(float3*) elastic_compenent->m_device_sph_kernel,
			(float3*) elastic_compenent->m_device_sph_kernel_inv,
            ( float3* )elastic_compenent->m_device_external_force,
			elastic_compenent->m_device_sum,
			elastic_compenent->m_device_neighbour_vector,
			elastic_compenent->m_device_neighbour_vector_start,
            elastic_compenent->m_params->m_sph_radius,
			elastic_compenent->m_num_particle
		);

        calLM(
            elastic_compenent->m_device_energy,
            elastic_compenent->m_device_lm,
            elastic_compenent->m_num_particle
        );
        std::cout << "Pre Compute successfully first.\n";

        return true;
    }

    bool ElasticParticleSovler::neighbourSearch()
    {

        if (!m_elastic_particle->hasComponent<ElasticComponent>())
        {
            std::cout << "no component.\n";
            return false;
        }


        ElasticComponent* elastic_compenent = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_compenent == nullptr)
            return false;

        computeHash(
            elastic_compenent->m_device_grid_particle_hash,
            elastic_compenent->m_device_next_pos,
            elastic_compenent->m_params->m_world_orgin,
            elastic_compenent->m_params->m_cell_size,
            elastic_compenent->m_params->m_grid_num,
            elastic_compenent->m_num_particle
        );

        sortParticles(
            elastic_compenent->m_device_grid_particle_hash,
            elastic_compenent->m_device_index,
            elastic_compenent->m_num_particle
        );

        findCellRange(
            elastic_compenent->m_device_cell_start,
            elastic_compenent->m_device_cell_end,
            elastic_compenent->m_device_grid_particle_hash,
            elastic_compenent->m_num_particle,
            elastic_compenent->m_params->grid_size
        );

        //printf(" neighbour search success!~");
    }

    bool ElasticParticleSovler::step() {
        if (!m_is_init)
        {
            std::cout << "Must initialized first.\n";
            return false;
        }
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_compenent = m_elastic_particle->getComponent<ElasticComponent>();

        if (elastic_compenent == nullptr)
            return false;
       
        if (elastic_compenent->m_bInitialized == false)
        {
            std::cout << "Must initialized component first.\n";
            return false;
        }

        advect(
            (float3*)elastic_compenent->m_device_pos,
            (float3*)elastic_compenent->m_device_next_pos,
            (float3*)elastic_compenent->m_device_vel,
            (float3*)elastic_compenent->m_device_delte_pos,
            (float3*)elastic_compenent->m_device_external_force,
            elastic_compenent->m_device_phase,
            m_config.m_gravity,
            m_config.m_dt,
            elastic_compenent->m_num_particle
        );
         calLM(
            elastic_compenent->m_device_energy,
            elastic_compenent->m_device_lm,
            elastic_compenent->m_num_particle);


        neighbourSearch();

        int iter = 0;
        while (iter < m_config.m_solver_iteration)
        {
            {
				calGradientDeformation(
				(float3*)elastic_compenent->m_device_sph_kernel,
				(float3*)elastic_compenent->m_device_next_pos,
				(float3*)elastic_compenent->m_device_initial_pos,
				(float3*)elastic_compenent->m_device_y,
				elastic_compenent->m_device_sum,
				elastic_compenent->m_device_L,
				elastic_compenent->m_device_F,
				elastic_compenent->m_device_neighbour_vector,
				elastic_compenent->m_device_neighbour_vector_start,
                elastic_compenent->m_params->m_volume,
				elastic_compenent->m_num_particle
				);
				calR(
				elastic_compenent->m_device_F,
				elastic_compenent->m_device_R,
				elastic_compenent->m_num_particle
				);
				calP(
                this->m_anisotropy,
				elastic_compenent->m_device_F,
				elastic_compenent->m_device_R,
				elastic_compenent->m_device_P,
				elastic_compenent->m_num_particle
				);
				calEnergy(
                this->m_anisotropy,
				elastic_compenent->m_device_F,
				elastic_compenent->m_device_R,
				elastic_compenent->m_device_energy,
				elastic_compenent->m_num_particle
				);
				solveDelteLagrangeMultiplier(
				(float3*)elastic_compenent->m_device_sph_kernel,
				elastic_compenent->m_device_energy,
				elastic_compenent->m_device_lm,
				elastic_compenent->m_device_delte_lm,
				elastic_compenent->m_device_sum,
				(float3*)elastic_compenent->m_device_initial_pos,
				(float3*)elastic_compenent->m_device_y,
				elastic_compenent->m_device_L,
				elastic_compenent->m_device_P,
				elastic_compenent->m_device_neighbour_vector,
				elastic_compenent->m_device_neighbour_vector_start,
                m_config.m_dt,
                elastic_compenent->m_params->m_volume,
				elastic_compenent->m_num_particle
				);
				solveDelteDistance(
				(float3*)elastic_compenent->m_device_sph_kernel,
				(float3*)elastic_compenent->m_device_sph_kernel_inv,
				elastic_compenent->m_device_P,
				elastic_compenent->m_device_L,
				elastic_compenent->m_device_delte_lm,
				elastic_compenent->m_device_sum,
				(float3*)elastic_compenent->m_device_delte_pos,
				(float3*)elastic_compenent->m_device_initial_pos,
				(float3*)elastic_compenent->m_device_y,
				elastic_compenent->m_device_neighbour_vector,
				elastic_compenent->m_device_neighbour_vector_start,
                elastic_compenent->m_params->m_volume,
				elastic_compenent->m_num_particle
				);
				updateDeltxlm(
				(float3*)elastic_compenent->m_device_delte_pos,
				(float3*)elastic_compenent->m_device_next_pos,
				elastic_compenent->m_device_delte_lm,
				elastic_compenent->m_device_lm,
				elastic_compenent->m_num_particle
				);
			}

            // collision 
            solveBoundaryConstraint(
                ( float3* )elastic_compenent->m_device_next_pos,
                ( float3* )elastic_compenent->m_device_vel,
                elastic_compenent->m_params->m_lb_boundary,
                elastic_compenent->m_params->m_rt_boundary,
                elastic_compenent->m_params->m_particle_radius,
                elastic_compenent->m_num_particle);

            calCollisionDeltex(
                ( float3* )elastic_compenent->m_device_initial_pos,
                ( float3* )elastic_compenent->m_device_delte_pos,
                ( float3* )elastic_compenent->m_device_next_pos,
                elastic_compenent->m_device_phase,
                elastic_compenent->m_device_index,
                elastic_compenent->m_device_cell_start,
                elastic_compenent->m_device_cell_end,
                elastic_compenent->m_device_grid_particle_hash,
                elastic_compenent->m_params->m_sph_radius,
                elastic_compenent->m_params->m_world_orgin,
                elastic_compenent->m_params->m_cell_size,
                elastic_compenent->m_params->m_grid_num,
                elastic_compenent->m_num_particle
            );

            updateCollDeltx(
                ( float3* )elastic_compenent->m_device_delte_pos,
                ( float3* )elastic_compenent->m_device_next_pos,
                elastic_compenent->m_device_index,
                elastic_compenent->m_num_particle
            );

			iter++;
		}

		update(
			(float3*)elastic_compenent->m_device_next_pos,
			(float3*)elastic_compenent->m_device_pos,
			(float3*)elastic_compenent->m_device_vel,
            elastic_compenent->m_device_phase,
			m_config.m_dt,
			elastic_compenent->m_num_particle
		);

        return true;
    }   

    bool ElasticParticleSovler::run() {

        if (!m_is_init)
        {
            return false;
        }
        if (!m_elastic_particle)
        {
            return false;
        }
        // Update till termination
     
        float totaltime = 0.f;
        int step_id = 0;
        while (m_cur_time < m_config.m_total_time)
        {
            double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
            cudaEvent_t start, stop;
            float       milliseconds = 0.f;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            m_cur_time += dt;
            m_config.m_dt = dt;
            step();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            totaltime += milliseconds;
            ++step_id;
            printf("step_id: %d , frame time: %f ms ! frame: %f ! \n", step_id, totaltime / step_id, 1000 * step_id / totaltime);
            if (m_write_ply)
                writeToPly(step_id);
        }
        return true;
    }
    
   bool ElasticParticleSovler::isApplicable(const Object* object) const
    {
        if (!object)
            return false;

        return object->hasComponent<ElasticComponent>();
    }

    bool ElasticParticleSovler::attachObject(Object* object)
    {
        if (!object)
            return false;

        if (!this->isApplicable(object))
        {
            std::cout << "error: object is not applicable.\n";
            return false;
        }

        if (object->hasComponent<ElasticComponent>())
        {
            std::cout << "object attached as elastic particle system.\n";
            m_elastic_particle                    = object;
            ElasticComponent* elastic_compenet = m_elastic_particle->getComponent<ElasticComponent>();
            m_num_particles                       = elastic_compenet->m_num_particle;
            m_host_pos                            = new float[m_num_particles * 3];
        }
        initialize();
        return true;
    }

    bool ElasticParticleSovler::detachObject(Object* object)
    {
        if (!object)
            return false;

        if (m_elastic_particle == object)
        {
            m_elastic_particle = nullptr;
            return true;
        }
        std::cout << "    error: object is not attached to the solver.\n";
        return false;
    }

    void ElasticParticleSovler::clearAttachment()
    {
        m_elastic_particle = nullptr;
    }

    void ElasticParticleSovler::config(SolverConfig& config)
    {
        m_config.m_gravity = make_float3(0.f, -9.8f, 0.f);
        if (config.m_solver_iteration < 0)
            config.m_solver_iteration = 1;
        m_config                  = config;
    }


    bool ElasticParticleSovler::setWorldBoundary(float lx, float ly, float lz, float ux, float uy, float uz)
    {
       
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return false;

        elastic_component->m_params->m_lb_boundary = make_float3(lx, ly, lz);
        elastic_component->m_params->m_rt_boundary = make_float3(ux, uy, uz);
        // compute the grid and neighbour again

        return true;
    }
    void ElasticParticleSovler::setStiffness(float& stiffness)
    {
        if (stiffness > 1.f)
            stiffness = 1.f;
        else if (stiffness <= 0)
            stiffness = 0.1;
        m_config.m_stiffness = stiffness;
    }


    void ElasticParticleSovler::setSolverIteration(unsigned int& iteration)
    {
        if (iteration <= 0)
            iteration = 1;
        m_config.m_solver_iteration = iteration;
    }

    void ElasticParticleSovler::setGravity(const float& gravity)
    {
        m_config.m_gravity = make_float3(0, gravity, 0);
    }

    void ElasticParticleSovler::setGravity(const float& x, const float& y, const float& z)
    {
        m_config.m_gravity = make_float3(x, y, z);
    }


    void ElasticParticleSovler::setWorldOrigin(const float& x, const float& y, const float& z)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return ;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return ;
        elastic_component->m_params->m_world_orgin = make_float3(x, y, z);
    }

    void ElasticParticleSovler::_finalize()
    {
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
                m_elastic_particle->getComponent<ElasticComponent>()->free();
        }
        freeParticleMemory(m_num_particles);
    }

    bool ElasticParticleSovler::setParticlePosition(const std::vector<float>& position)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return false;

        // free particle component and solver data on device.
        if (elastic_component->m_bInitialized)
        {
            elastic_component->m_bInitialized = false;
            freeParticleMemory(position.size() / 3);
        }
        m_num_particles                     = static_cast<unsigned int>(position.size() / 3);

        elastic_component->m_host_neighbour_vector_start.push_back(0);
        elastic_component->initialNeighbour(position, 0);
        elastic_component->m_num_particle = m_num_particles;
        elastic_component->m_host_pos     = position;
        for (int i = 0; i < m_num_particles; i++)
            elastic_component->m_host_index.push_back(i);


        // malloc particle component data on device
        size_t mem_size = elastic_component->m_num_particle * 3 * sizeof(float);
        size_t length       = elastic_component->m_host_neighbour_vector.size();
        size_t length_start = elastic_component->m_host_neighbour_vector_start.size();

        unsigned int memMatSize    = sizeof(mat3) * m_num_particles;
        unsigned int memFloatSize  = sizeof(float) * m_num_particles;
        unsigned int memFloat3Size = sizeof(float) * 3 * m_num_particles;
        unsigned int menIntSize    = sizeof(int) * m_num_particles;
        unsigned int menUIntSize   = sizeof(unsigned int) * m_num_particles;
        unsigned int menGridSize   = sizeof(unsigned int) * elastic_component->m_params->grid_size;

        cudaMalloc(( void** )&elastic_component->m_device_pos, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_next_pos, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_delte_pos, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_initial_pos, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_vel, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_sum, memFloatSize);
        cudaMalloc(( void** )&elastic_component->m_device_lm, memFloatSize);
        cudaMalloc(( void** )&elastic_component->m_device_delte_lm, memFloatSize);
        cudaMalloc(( void** )&elastic_component->m_device_y, memFloat3Size);
        cudaMalloc(( void** )&elastic_component->m_device_L, memMatSize);
        cudaMalloc(( void** )&elastic_component->m_device_F, memMatSize);
        cudaMalloc(( void** )&elastic_component->m_device_P, memMatSize);
        cudaMalloc(( void** )&elastic_component->m_device_R, memMatSize);
        cudaMalloc(( void** )&elastic_component->m_device_energy, memFloatSize);
        cudaMalloc(( void** )&elastic_component->m_device_index, menUIntSize);
        cudaMalloc(( void** )&elastic_component->m_device_phase, memFloatSize);
        cudaMalloc(( void** )&elastic_component->m_device_cell_start, menGridSize);
        cudaMalloc(( void** )&elastic_component->m_device_cell_end, menGridSize);
        cudaMalloc(( void** )&elastic_component->m_device_grid_particle_hash, menUIntSize);
        cudaMalloc(( void** )&elastic_component->m_device_neighbour_vector, length * sizeof(unsigned int));
        cudaMalloc(( void** )&elastic_component->m_device_neighbour_vector_start, length_start * sizeof(unsigned int));
        cudaMalloc(( void** )&elastic_component->m_device_sph_kernel, length * sizeof(float) * 3);
        cudaMalloc(( void** )&elastic_component->m_device_sph_kernel_inv, length * sizeof(float) * 3);
        getLastCudaError("component malloc! \n");

        // malloc solver data on device
        //elastic_component->reset();
        cudaMemcpy(elastic_component->m_device_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(elastic_component->m_device_initial_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(elastic_component->m_device_next_pos, position.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(elastic_component->m_device_index, elastic_component->m_host_index.data(), elastic_component->m_num_particle * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(elastic_component->m_device_neighbour_vector, elastic_component->m_host_neighbour_vector.data(), length * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(elastic_component->m_device_neighbour_vector_start, elastic_component->m_host_neighbour_vector_start.data(), length_start * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemset(elastic_component->m_device_vel, 0.f, mem_size);
        cudaMemset(elastic_component->m_device_phase, 0.f, memFloatSize);
        getLastCudaError("component memcpy! \n");

        
        if (m_host_pos == nullptr)
            m_host_pos = new float[m_num_particles * 3];

        //mallocParticleMemory(m_num_particles);
        elastic_component->m_bInitialized = true;
        preComputeSPH();
        
        return true;
    }

    bool ElasticParticleSovler::setParticleVelocity(const std::vector<float>& velocity)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return false;

        if (static_cast<unsigned int>(velocity.size() / 3) != elastic_component->m_num_particle)
            return false;

        std::vector<float>().swap(elastic_component->m_host_vel);
        elastic_component->m_host_vel = velocity;
        cudaMemcpy(elastic_component->m_device_vel, velocity.data(), velocity.size() * sizeof(float), cudaMemcpyHostToDevice);

        // granular_component->m_bInitialized = true;
        return true;
    }

    bool ElasticParticleSovler::setParticlePhase(const std::vector<float>& phase) {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return false;

        // free particle component and solver data on device.
        if (static_cast<unsigned int>(phase.size())  != elastic_component->m_num_particle)
            return false;

        std::vector<float>().swap(elastic_component->m_host_phase);
        elastic_component->m_host_phase = phase;
        cudaMemcpy(elastic_component->m_device_phase, phase.data(), phase.size() * sizeof(float), cudaMemcpyHostToDevice);
        return true;

    }

     bool ElasticParticleSovler::setParticleExternalForce(const std::vector<float>& force)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return false;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return false;

        // free particle component and solver data on device.
        if (static_cast<unsigned int>(force.size() / 3) != elastic_component->m_num_particle)
            return false;

        std::vector<float>().swap(elastic_component->m_host_external_force);
        elastic_component->m_host_external_force = force;
        cudaMemcpy(elastic_component->m_device_external_force, force.data(), force.size() * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    }
   

    float* ElasticParticleSovler::getParticlePositionPtr()
    {
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                    return elastic_component->m_device_pos;
            }
        }
        return nullptr;
    }

    float* ElasticParticleSovler::getParticleVelocityPtr()
    {
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                    return elastic_component->m_device_vel;
            }
        }
        return nullptr;
    }

   float* ElasticParticleSovler::getParticlePhasePtr() 
   {
       if (m_elastic_particle != nullptr)
       {
           if (m_elastic_particle->hasComponent<ElasticComponent>())
           {
               ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
               if (elastic_component != nullptr)
                   return elastic_component->m_device_phase;
           }
       }
   }

    void ElasticParticleSovler::getParticleRadius(float& particleRadius)
    {
       
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                    particleRadius = elastic_component->m_params->m_particle_radius;
            }
        }
    }

    void ElasticParticleSovler::setParticleRadius(const float& particleRadius)
    {
      
        if (particleRadius <= 0)
            return;


        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr) 
                {
                    elastic_component->m_params->m_particle_radius = particleRadius;
                    elastic_component->m_params->m_particle_dimcenter = static_cast<float>(2 * particleRadius);
                    elastic_component->m_params->m_sph_radius = static_cast<float>(4 * particleRadius);
                    elastic_component->m_params->m_volume             = 4 / 3 * M_PI * pow(particleRadius, 3);
                    elastic_component->m_params->m_cell_size          = make_float3(elastic_component->m_params->m_sph_radius, elastic_component->m_params->m_sph_radius, elastic_component->m_params->m_sph_radius);
                }
            }
        }
    }

    void ElasticParticleSovler:: setYoungModules(const float& young_modules)
    {
       
        if (young_modules <= 0)
            return;

        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                {
                    elastic_component->m_params->young_modules        = young_modules;
                    elastic_component->m_params->lame_first           = elastic_component->m_params->young_modules * elastic_component->m_params->possion_ratio / ((1 + elastic_component->m_params->possion_ratio) * (1 - 2 * elastic_component->m_params->possion_ratio));
                    elastic_component->m_params->lame_second          = elastic_component->m_params->young_modules / (2 * (1 + elastic_component->m_params->possion_ratio));
                    setParamters(elastic_component->m_params);
                }
            }
        }
    }

    void ElasticParticleSovler::setPossionRatio(const float& possion_ratio) 
    {
        if (possion_ratio <= 0 || possion_ratio >= 0.5)
            return;

        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                {
                    elastic_component->m_params->possion_ratio      = possion_ratio;
                    elastic_component->m_params->lame_first    = elastic_component->m_params->young_modules * elastic_component->m_params->possion_ratio / ((1 + elastic_component->m_params->possion_ratio) * (1 - 2 * elastic_component->m_params->possion_ratio));
                    elastic_component->m_params->lame_second   = elastic_component->m_params->young_modules / (2 * (1 + elastic_component->m_params->possion_ratio));
                    setParamters(elastic_component->m_params);
               
                }
            }
        }
    }

    void ElasticParticleSovler::getYoungModules(float& young_modules)
    {
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                {
                    young_modules = elastic_component->m_params->young_modules;
                }
            }
        }
        else
            young_modules = 3e5;

    }

    void ElasticParticleSovler::getPossionRatio(float& possion_ratio)
    {
        if (m_elastic_particle != nullptr)
        {
            if (m_elastic_particle->hasComponent<ElasticComponent>())
            {
                ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
                if (elastic_component != nullptr)
                {
                    possion_ratio = elastic_component->m_params->possion_ratio;
                }
            }
        }
        else
            possion_ratio = 0.25;
    }

    void ElasticParticleSovler::setWriteToPly(const bool write_to_ply) {
        m_write_ply = write_to_ply;
     }

     void ElasticParticleSovler::setAnisotropy(const float3 Anisotropy)
     {
         m_anisotropy = Anisotropy;
     }


    void ElasticParticleSovler::writeToPly(const int& step_id)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
        {   
            std::cout << "error" << std::endl;
            return;
        }
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
        {
            std::cout << "error" << std::endl;
            return;
        }
        //  write to ply file
        std::string filename = "./ply/elastic_" + std::to_string(step_id) + ".ply";
        //std::cout << "write to ply: " << filename << std::endl;
        cudaMemcpy(m_host_pos, elastic_component->m_device_pos, sizeof(float3) * elastic_component->m_num_particle, cudaMemcpyDeviceToHost);
        getLastCudaError("write to ply! \n");
        // write pos ti ply file
        std::ofstream outfile(filename);
        outfile << "ply\n";
        outfile << "format ascii 1.0\n";
        outfile << "element vertex " << elastic_component->m_num_particle << "\n";
        outfile << "property float x\n";
        outfile << "property float y\n";
        outfile << "property float z\n";
        outfile << "property uchar red\n";
        outfile << "property uchar green\n";
        outfile << "property uchar blue\n";
        outfile << "end_header\n";
        for (unsigned int i = 0; i < elastic_component->m_num_particle * 3; i += 3)
        {   
            if ((i / 3) % 2250 < 45)
                outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " 255 0 0\n";
            else
                outfile << m_host_pos[i] << " " << m_host_pos[i + 1] << " " << m_host_pos[i + 2] << " 255 255 255\n";
        }
        outfile.close();
    }

    void ElasticParticleSovler::freeParticleMemory(const int& numParticles)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return;

        delete[] m_host_pos;
        
        elastic_component->free();
        //elastic_component->~ElasticComponent();
        if (m_host_pos != nullptr)
            m_host_pos = nullptr;

    }

    void ElasticParticleSovler::mallocParticleMemory(const int& numParticles)
    {
        if (!m_elastic_particle->hasComponent<ElasticComponent>())
            return;
        ElasticComponent* elastic_component = m_elastic_particle->getComponent<ElasticComponent>();
        if (elastic_component == nullptr)
            return;

        printf("malloc particle memory %d! \n ", numParticles);
        size_t memSize = numParticles * sizeof(float) * 3;
        elastic_component->initialize(numParticles);

        if (m_host_pos == nullptr)
            m_host_pos = new float[m_num_particles * 3];

    }

}
