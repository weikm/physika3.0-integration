/**
 * @file pbf_nonnewton_solver.cpp
 *
 * @author Long Shen (sl_111211@163.com)
 * @date 2023-10-07
 * @brief This file declares the solver which is designed to handle simulations of nonNewtonian fluid.
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * This solver can be employed for simulations requiring nonNewtonian-viscous behavior of particle fluid (like chocolate, paint, etc.)
 * The file defines the SimMaterial enum, NonNewtonFluidComponentpinn struct, and the PBFNonNewtonpinnSolver class which inherits from the base Solver class.
 * Various methods for managing the particle system such as initializing, resetting, stepping, attaching/detaching objects, etc. are also declared.
 * A SolverConfig struct is provided for configuring simulation time parameters.
 *
 * @section DEPENDENCIES
 *
 * This file includes several standard and third-party libraries: iostream, string, fstream, and vector_functions.
 * It also includes "granular_particle_solver.hpp" and "framework/object.hpp" from the Physika framework.
 *
 * @section USAGE
 *
 * To use this solver, you need to provide the initial positions of the particles(By defining the NonNewtonFluidComponentpinn).
 * The solver then calculates the positions of the particles at the next time step, which can be retrieved for further use or visualization.
 *
 * The solver can handle errors during computation. Specifically, it includes a method 'cudaGetLastError_t' to
 * retrieve the last CUDA error message.
 *
 * @section WARNING
 *
 * Ensure that the CUDA environment is properly set up, as the solver relies on it for computation.
 *
 */

#include <iomanip>
#include <string>
#include <ctime>
#include <chrono>
#include<random>
#include<filesystem>




#include "pbf_nonnewtonpinn_solver.hpp"
#include "framework/object.hpp"
#include "cuda_tool.cuh"
#include "cuda_kernel_api.cuh"
#include "IOTool.h"

#include <filesystem>
#include <Windows.h>


float VIS = 0.9f;  //prm default

namespace Physika {

PBFNonNewtonpinnSolver::PBFNonNewtonpinnSolver()
{
}

PBFNonNewtonpinnSolver::~PBFNonNewtonpinnSolver()
{
    finalize();
}

bool PBFNonNewtonpinnSolver::initialize()
{


     //zxc add
      std::string file_path;  
    file_path  = std::filesystem::current_path()
                               .parent_path()
                               .parent_path()
                               .parent_path()
                      .parent_path()
                      .string()
                  + "\\examples\\thickening_nonnewtonpinn_sample"  //prm
    +"\\pinnvis.txt";

    file_path = std::filesystem::current_path().string()+"\\pinn\\pinnvis.txt";
    std::ifstream file(file_path);
    std::cout << "solver[vis cfg dir]\t" << file_path << std::endl;
    if (!file.is_open())
    {
        std::cout << "solver[default VIS]\n";
    }
    else
    {
        std::string line;
        int         lineidx = 0;
        while (std::getline(file, line))
        {
            if (lineidx == 0)
            {

                VIS = float(stof(line));
                std::cout << "solver[cfg VIS]\t";
                std::cout << VIS << std::endl;
            }
            if (lineidx == 1)
            {
            }
            lineidx++;
        }
        file.close();

    }
    // Sleep(2000);






    if (m_particle_radius == 0.f)
    {
        std::cout << "ERROR:: PBFNonNewtonpinnSolver::initialize():: solver particle radius not set yet.\n";
    }

    m_is_init = true;

    if (m_fluid_obj)
    {
        auto nn_component = m_fluid_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_host_constPack.total_particle_num += nn_component->getParticleNum2();
        m_host_overall_pos.insert(m_host_overall_pos.end(), nn_component->m_host_cur_pos.begin(), nn_component->m_host_cur_pos.end());
        m_host_overall_vel.insert(m_host_overall_vel.end(), nn_component->m_host_cur_vel.begin(), nn_component->m_host_cur_vel.end());

        auto _mat_ = std::vector<SimMaterial>(nn_component->m_host_cur_pos.size(), nn_component->getMaterial2());
        m_host_overall_material.insert(m_host_overall_material.end(), _mat_.begin(), _mat_.end());
    }
    if (m_solid_obj)
    {
        auto nn_component = m_solid_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_host_constPack.total_particle_num += nn_component->getParticleNum2();
        m_host_overall_pos.insert(m_host_overall_pos.end(), nn_component->m_host_cur_pos.begin(), nn_component->m_host_cur_pos.end());
        m_host_overall_vel.insert(m_host_overall_vel.end(), nn_component->m_host_cur_vel.begin(), nn_component->m_host_cur_vel.end());

        auto _mat_ = std::vector<SimMaterial>(nn_component->m_host_cur_pos.size(), nn_component->getMaterial2());
        m_host_overall_material.insert(m_host_overall_material.end(), _mat_.begin(), _mat_.end());
    }
    if (m_bound_obj)
    {
        auto nn_component = m_bound_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_host_constPack.total_particle_num += nn_component->getParticleNum2();
        m_host_overall_pos.insert(m_host_overall_pos.end(), nn_component->m_host_cur_pos.begin(), nn_component->m_host_cur_pos.end());
        m_host_overall_vel.insert(m_host_overall_vel.end(), nn_component->m_host_cur_vel.begin(), nn_component->m_host_cur_vel.end());

        auto _mat_ = std::vector<SimMaterial>(nn_component->m_host_cur_pos.size(), nn_component->getMaterial2());
        m_host_overall_material.insert(m_host_overall_material.end(), _mat_.begin(), _mat_.end());
    }

    m_is_init &= initDeviceSource();

    return m_is_init;
}

bool PBFNonNewtonpinnSolver::isInitialized() const
{
    return m_is_init;
}

bool PBFNonNewtonpinnSolver::reset()
{
    if (m_config.m_use_qwUI)
    {
        return true;
    }
    freeMemory();
    m_is_init  = false;
    m_cur_time = 0.f;

    m_host_overall_pos.resize(0);
    m_host_overall_vel.resize(0);
    m_host_overall_material.resize(0);

    return true;
}

bool PBFNonNewtonpinnSolver::step()
{

    //std::cout << " zxc use this step !!!";//yes
    

       //zxc add
    static int I = 0;//record step num
    static int J = 0;//record export file num



    if (m_isStart)
    {
        computeRigidParticleVolume();
        m_isStart = false;
    }

    computeExtForce();

    updateNeighbors();

    for (int i = 0; i < m_config.m_iter_num; ++i)
    {
        computeDensity();

        computeDxFromDensityConstraint();
    }

    applyDx();

    computeVisForce();

    //    float v;
    //    avg_output(m_fluid_obj->getComponent<NNComponentpinn>()->getParticleNum(), m_device_vis, v);
    //    std::cout << v << '\n';


     //zxc add below--------------------------------------
    cudaMemcpy(m_host_overall_pos.data(), m_device_pos, m_host_constPack.total_particle_num * sizeof(float3), cudaMemcpyDeviceToHost);
    //write_ply("frame_" + std::to_string(i) + "qwe.ply", m_host_overall_pos);

    static int samplenum = 1000;  //prm
    static int  frameNum=50;  //prm


    if (J == frameNum)
    {
        std::cout << "[data generation done. no export from now]\n";
        return true; 
    }
                 
    static float    dt_pinn             = 0.1;//prm
    static float    dt_step        = m_config.m_dt;
    static int      sampleDistance      = int(dt_pinn/dt_step);  


    static float            lastx[1000];
    static float            lasty[1000];
    static float            lastz[1000];
    static float                   vx = 0.;
    static float                   vy = 0.;
    static float                   vz = 0.;
    static int                     allnum = m_fluid_obj->getComponent<NonNewtonFluidComponentpinn>()->getParticleNum2();  //m_host_overall_pos.size();

    static std::vector<int> numbers(allnum);
    if (I == 0)
    {
               for (int i = 0; i < numbers.size(); i++)
        {
            numbers[i] = i;
        }
        std::mt19937       g(1234);
        std::shuffle(numbers.begin(), numbers.end(), g);
        std::ofstream ofs1("pinnmk.txt ");//prm
        for (int number : numbers)
        {

            ofs1 << number << " ";
        }
        ofs1.close();
    }
                                                        

    if (I % sampleDistance == 0)  //保存本帧速度
    {
        for (size_t k = 0; k < samplenum; k++)
        {
            lastx[k] = m_host_overall_pos[numbers[k]].x;
            lasty[k] = m_host_overall_pos[numbers[k]].y;
            lastz[k] = m_host_overall_pos[numbers[k]].z;
        }
    }

    if (I % sampleDistance == 1)
    {



        std::string filename = "pinndt-phy" + std::to_string(J) + ".ply ";  //prm
        J++;
        std::cout << "[Export]" << J << "/"<< frameNum << "\n";
        std::cout << "[dt_step]" << dt_step << std::endl;
        std::ofstream ofs(filename);


        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << samplenum << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "property float vx\n";
        ofs << "property float vy\n";
        ofs << "property float vz\n";
        ofs << "end_header\n";

        for (size_t k = 0; k < samplenum; k++)
        {
            vx = (m_host_overall_pos[numbers[k]].x - lastx[k]) / dt_step;  //prm
            vy = (m_host_overall_pos[numbers[k]].y - lasty[k]) / dt_step;
            vz = (m_host_overall_pos[numbers[k]].z - lastz[k]) / dt_step;

            ofs << lastx[k] << " "  //cannot export at 1st frame
                << lasty[k] << " "
                << lastz[k] << " "
                << vx << " "
                << vy << " "
                << vz << "\n";
        }
        ofs.close();
    }

    I++;


    return true;
}
bool PBFNonNewtonpinnSolver::run()
{
    if (!m_is_init)
        initialize();

    int cnt = 500;
    for (int i = 0; i < cnt; ++i)
    {
        auto start = clock();
        step();
        auto end = clock();

        double tick = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "frame out: " << i + 1 << " / " << cnt << "  "
                  << "FPS: " << int(1 / tick) << std::endl;

        //        cudaMemcpy(m_host_overall_pos.data(), m_device_pos, m_host_constPack.total_particle_num * sizeof(float3), cudaMemcpyDeviceToHost);
        //        write_ply("./ply/frame_" + std::to_string(i) + ".ply", m_host_overall_pos);
        //        cudaMemcpy(m_host_overall_pos.data(), m_device_pos, m_host_constPack.total_particle_num * sizeof(float3), cudaMemcpyDeviceToHost);
        //        write_ply("C:\\Users\\sl936\\Desktop\\experiment\\ply\\qy-test\\thickening\\" + std::to_string(i) + ".ply", m_host_overall_pos);
    }

    return false;
}

bool PBFNonNewtonpinnSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<NonNewtonFluidComponentpinn>();
}

bool PBFNonNewtonpinnSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!isApplicable(object))
    {
        std::cout << "ERROR:: PBFNonNewtonpinnSolver::attachObject():: object is not applicable.\n";
        return false;
    }

    auto nn_component = object->getComponent<NonNewtonFluidComponentpinn>();

    if (nn_component->getMaterial2() == SimMaterial::FLUID)
        m_fluid_obj = object;

    else if (nn_component->getMaterial2() == SimMaterial::RIGID)
        m_solid_obj = object;

    else if (nn_component->getMaterial2() == SimMaterial::BOUND)
        m_bound_obj = object;

    std::cout << "object attached as non_newton particle system.\n";

    return true;
}

bool PBFNonNewtonpinnSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_fluid_obj == object)
    {
        reset();
        m_fluid_obj = nullptr;
        return true;
    }

    if (m_solid_obj == object)
    {
        reset();
        m_solid_obj = nullptr;
        return true;
    }

    if (m_bound_obj == object)
    {
        reset();
        m_bound_obj = nullptr;
        return true;
    }

    std::cout << "ERROR:: PBFNonNewtonpinnSolver::detachObject():: object is not attached to the solver.\n";
    return false;
}

void PBFNonNewtonpinnSolver::clearAttachment()
{
    m_fluid_obj = nullptr;
    m_solid_obj = nullptr;
    m_bound_obj = nullptr;
    reset();
}

void PBFNonNewtonpinnSolver::config(PBFNonNewtonpinnSolver::SolverConfig& config)
{
    m_config = config;
}

void PBFNonNewtonpinnSolver::setUnifiedParticleRadius(float radius)
{
    m_particle_radius = radius;
}
void PBFNonNewtonpinnSolver::setSceneBoundary(float3 scene_lb, float3 scene_size)
{
    m_host_constPack.ns_sceneLB   = scene_lb;
    m_host_constPack.ns_sceneSize = scene_size;
}

bool PBFNonNewtonpinnSolver::initDeviceSource()
{
    size_t size1 = m_host_constPack.total_particle_num * sizeof(float3);
    size_t size2 = m_host_constPack.total_particle_num * sizeof(float);
    size_t size3 = m_host_constPack.total_particle_num * sizeof(SimMaterial);

    cudaMalloc_t(( void** )&m_device_pos, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_predictPos, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_vel, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_acc, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_dx, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_density, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_volume, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_lam, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_vis, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_shearRate, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_material, size3, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ext_force, size1, m_device_mem);

    cudaMemcpy(m_device_pos, m_host_overall_pos.data(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_predictPos, m_host_overall_pos.data(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_vel, m_host_overall_vel.data(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(m_device_material, m_host_overall_material.data(), size3, cudaMemcpyHostToDevice);

        std::vector<float3> m_overall_ext_force;
    if (m_fluid_obj)
    {
        auto nn_component = m_fluid_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_overall_ext_force.insert(m_overall_ext_force.end(),
                                   nn_component->m_host_ext_force.begin(),
                                   nn_component->m_host_ext_force.end());
    }
    if (m_bound_obj)
    {
        auto nn_component = m_bound_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_overall_ext_force.insert(m_overall_ext_force.end(),
                                   nn_component->m_host_ext_force.begin(),
                                   nn_component->m_host_ext_force.end());
    }
    if (m_solid_obj)
    {
        auto nn_component = m_solid_obj->getComponent<NonNewtonFluidComponentpinn>();
        m_overall_ext_force.insert(m_overall_ext_force.end(),
                                   nn_component->m_host_ext_force.begin(),
                                   nn_component->m_host_ext_force.end());
    }
    if (m_overall_ext_force.empty())
    {
        m_overall_ext_force = std::vector<float3>(m_host_overall_pos.size(), make_float3(0, 0, 0));
    }
    cudaMemcpy(m_device_ext_force, m_overall_ext_force.data(), size1, cudaMemcpyHostToDevice);

    m_host_constPack.rest_density   = 1000.f;
    m_host_constPack.rest_mass      = 1.f;
    m_host_constPack.sph_h          = m_particle_radius * 4;
    m_host_constPack.pbf_Ks         = 0.1f;
    m_host_constPack.pbf_Dq         = 0.3f * m_host_constPack.sph_h;
    m_host_constPack.cross_vis0     = VIS;
    m_host_constPack.cross_visInf   = VIS;//prm
    std::cout<<"solver[set VIS]\t"<<VIS<<"\n";
    // Sleep(5500);
    m_host_constPack.cross_visBound = 0.01f;
    m_host_constPack.cross_K        = 2;

    m_host_constPack.total_particle_num = m_host_overall_pos.size();
    m_host_constPack.dt                 = static_cast<float>(m_config.m_dt);
    m_host_constPack.gravity            = m_config.m_gravity;

    std::vector<float> density(m_host_overall_pos.size(), m_host_constPack.rest_density);
    cudaMemcpy(m_device_density, density.data(), size2, cudaMemcpyHostToDevice);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    m_host_constPack.ns_threadPerBlock = prop.maxThreadsPerBlock;
    m_host_constPack.ns_blockNum       = (m_host_constPack.total_particle_num + m_host_constPack.ns_threadPerBlock - 1) / m_host_constPack.ns_threadPerBlock;
    m_host_constPack.ns_cellLength     = m_host_constPack.sph_h;
    m_host_constPack.ns_maxNeighborNum = 35;

    m_host_constPack.ns_gridSize = make_uint3(
        static_cast<uint32_t>(std::ceil(m_host_constPack.ns_sceneSize.x / m_host_constPack.ns_cellLength)),
        static_cast<uint32_t>(std::ceil(m_host_constPack.ns_sceneSize.y / m_host_constPack.ns_cellLength)),
        static_cast<uint32_t>(std::ceil(m_host_constPack.ns_sceneSize.z / m_host_constPack.ns_cellLength)));
    m_host_constPack.ns_cellNum = m_host_constPack.ns_gridSize.x * m_host_constPack.ns_gridSize.y * m_host_constPack.ns_gridSize.z;

    size1        = m_host_constPack.total_particle_num * sizeof(uint32_t);
    size2        = m_host_constPack.ns_cellNum * sizeof(uint32_t);
    size3        = m_host_constPack.total_particle_num * m_host_constPack.ns_maxNeighborNum * sizeof(uint32_t);
    size_t size4 = 27 * sizeof(int3);
    size_t size5 = sizeof(ConstPack);

    cudaMalloc_t(( void** )&m_device_ns_particleIndices, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_cellIndices, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_neighborNum, size1, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_cellStart, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_cellEnd, size2, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_neighbors, size3, m_device_mem);
    cudaMalloc_t(( void** )&m_device_ns_cellOffsets, size4, m_device_mem);
    cudaMalloc_t(( void** )&m_device_constPack, size5, m_device_mem);

    cudaMemcpy(m_device_constPack, &m_host_constPack, size5, cudaMemcpyHostToDevice);

    ConstPack cp{};
    cudaMemcpy(&cp, m_device_constPack, size5, cudaMemcpyDeviceToHost);

    std::vector<int3> offsets = {
        make_int3(-1, -1, -1),
        make_int3(-1, -1, 0),
        make_int3(-1, -1, 1),
        make_int3(-1, 0, -1),
        make_int3(-1, 0, 0),
        make_int3(-1, 0, 1),
        make_int3(-1, 1, -1),
        make_int3(-1, 1, 0),
        make_int3(-1, 1, 1),
        make_int3(0, -1, -1),
        make_int3(0, -1, 0),
        make_int3(0, -1, 1),
        make_int3(0, 0, -1),
        make_int3(0, 0, 0),
        make_int3(0, 0, 1),
        make_int3(0, 1, -1),
        make_int3(0, 1, 0),
        make_int3(0, 1, 1),
        make_int3(1, -1, -1),
        make_int3(1, -1, 0),
        make_int3(1, -1, 1),
        make_int3(1, 0, -1),
        make_int3(1, 0, 0),
        make_int3(1, 0, 1),
        make_int3(1, 1, -1),
        make_int3(1, 1, 0),
        make_int3(1, 1, 1),
    };

    cudaMemcpy(m_device_ns_cellOffsets, offsets.data(), size4, cudaMemcpyHostToDevice);

    if (cudaGetLastError_t("PBFNonNewtonpinnSolver::initDeviceSource() failed."))
    {
        updateNeighbors();
        //        dumpNeighborSearchInfo();
        return true;
    }

    return false;
}
void PBFNonNewtonpinnSolver::freeMemory()
{
    if (m_is_init)
    {
        std::cout << "PBFNonNewtonpinnSolver::freeMemory()...\n";
        cudaFree(m_device_ns_cellOffsets);
        cudaFree(m_device_ns_particleIndices);
        cudaFree(m_device_ns_cellIndices);
        cudaFree(m_device_ns_cellStart);
        cudaFree(m_device_ns_cellEnd);
        cudaFree(m_device_ns_neighborNum);
        cudaFree(m_device_ns_neighbors);
        cudaFree(m_device_pos);
        cudaFree(m_device_predictPos);
        cudaFree(m_device_vel);
        cudaFree(m_device_acc);
        cudaFree(m_device_dx);
        cudaFree(m_device_density);
        cudaFree(m_device_volume);
        cudaFree(m_device_material);
        cudaFree(m_device_lam);
        cudaFree(m_device_vis);
        cudaFree(m_device_shearRate);
        cudaFree(m_device_constPack);

        cudaGetLastError_t("PBFNonNewtonpinnSolver::freeMemory():: cudaFree() failed!");
    }
}

void PBFNonNewtonpinnSolver::finalize()
{
    freeMemory();
}

float3* PBFNonNewtonpinnSolver::getPosDevPtr() const
{
    return m_device_pos;
}

float3* PBFNonNewtonpinnSolver::getVelDevPtr() const
{
    return m_device_vel;
}

float* PBFNonNewtonpinnSolver::getVisDevPtr() const
{
    return m_device_vis;
}

void PBFNonNewtonpinnSolver::updateNeighbors()
{
    ns_resetDevPtr(m_host_constPack,
                   m_device_ns_cellStart,
                   m_device_ns_cellEnd,
                   m_device_ns_neighborNum,
                   m_device_ns_neighbors);

    ns_calcParticleHashValue(m_host_constPack,
                             m_device_constPack,
                             m_device_ns_particleIndices,
                             m_device_ns_cellIndices,
                             m_device_pos);

    ns_sortByHashValue(m_host_constPack,
                       m_device_ns_particleIndices,
                       m_device_ns_cellIndices);

    ns_findCellRange(m_host_constPack,
                     m_device_constPack,
                     m_device_ns_cellIndices,
                     m_device_ns_cellStart,
                     m_device_ns_cellEnd);

    ns_findNeighbors(m_host_constPack,
                     m_device_constPack,
                     m_device_ns_cellOffsets,
                     m_device_ns_particleIndices,
                     m_device_ns_cellStart,
                     m_device_ns_cellEnd,
                     m_device_ns_neighborNum,
                     m_device_ns_neighbors,
                     m_device_pos);
}
void PBFNonNewtonpinnSolver::dumpNeighborSearchInfo()
{
    auto  cellNum     = m_host_constPack.ns_cellNum;
    auto  particleNum = m_host_constPack.total_particle_num;
    auto* c_cellStart = new uint32_t[cellNum];
    auto* c_cellEnd   = new uint32_t[cellNum];
    cudaMemcpy(c_cellStart, m_device_ns_cellStart, cellNum * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_cellEnd, m_device_ns_cellEnd, cellNum * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t cnt = 0;
    for (int i = 0; i < cellNum; ++i)
    {
        if (c_cellStart[i] != UINT_MAX)
        {
            cnt++;
        }
    }
    delete[] c_cellStart;
    delete[] c_cellEnd;

    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Particle Num: " << std::setw(20)
              << particleNum << "\n";
    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Cell Num: " << std::setw(20) << cellNum
              << "\n";
    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Grid Size: " << std::setw(20)
              << std::to_string(m_host_constPack.ns_gridSize.x) + " * " + std::to_string(m_host_constPack.ns_gridSize.y)
                     + " * " + std::to_string(m_host_constPack.ns_gridSize.z)
              << "\n";
    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Allocated Mem: " << std::setw(20)
              << std::to_string(m_device_mem) + " MB"
              << "\n";
    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Average PartNum per cell: "
              << std::setw(20)
              << (particleNum / cnt) << "\n";
    std::cout << std::setw(25) << "UGBNeighborSearcher::" << std::setw(35) << "Activate Cell num: " << std::setw(20)
              << cnt << "\n\n";
}
void PBFNonNewtonpinnSolver::computeRigidParticleVolume()
{
    algo_computeRigidParticleVolume(m_host_constPack,
                                    m_device_constPack,
                                    m_device_ns_neighbors,
                                    m_device_ns_particleIndices,
                                    m_device_material,
                                    m_device_predictPos,
                                    m_device_volume);
}
void PBFNonNewtonpinnSolver::computeExtForce()
{
    algo_computeExtForce(m_host_constPack,
                         m_device_constPack,
                         m_device_ns_particleIndices,
                         m_device_material,
                         m_device_ext_force,
                         m_device_predictPos,
                         m_device_vel,
                         m_device_acc);
}
void PBFNonNewtonpinnSolver::computeDensity()
{
    algo_computeDensity(m_host_constPack,
                        m_device_constPack,
                        m_device_ns_neighbors,
                        m_device_ns_particleIndices,
                        m_device_material,
                        m_device_predictPos,
                        m_device_volume,
                        m_device_density);
}
void PBFNonNewtonpinnSolver::computeDxFromDensityConstraint()
{
    algo_computeDxFromDensityConstraint(m_host_constPack,
                                        m_device_constPack,
                                        m_device_ns_neighbors,
                                        m_device_ns_particleIndices,
                                        m_device_material,
                                        m_device_volume,
                                        m_device_density,
                                        m_device_predictPos,
                                        m_device_dx,
                                        m_device_lam);
}
void PBFNonNewtonpinnSolver::applyDx()
{
    algo_applyDx(m_host_constPack,
                 m_device_constPack,
                 m_device_ns_particleIndices,
                 m_device_material,
                 m_device_predictPos,
                 m_device_pos,
                 m_device_vel);
}
void PBFNonNewtonpinnSolver::computeVisForce()
{
    algo_computeVisForce(m_host_constPack,
                         m_device_constPack,
                         m_device_ns_neighbors,
                         m_device_ns_particleIndices,
                         m_device_material,
                         m_device_density,
                         m_device_predictPos,
                         m_device_vel,
                         m_device_acc,
                         m_device_vis,
                         m_device_shearRate);
}
void PBFNonNewtonpinnSolver::setParticleVelocity(const std::vector<float3>& velocity)
{
    m_host_overall_vel = velocity;
}
void PBFNonNewtonpinnSolver::setParticlePosition(const std::vector<float3>& position)
{
    m_host_overall_pos = position;
}
void PBFNonNewtonpinnSolver::setParticlePhase(const std::vector<float>& particlePhase)
{
    return;
}
void PBFNonNewtonpinnSolver::setExtForce(float3* device_ext_force)
{
    if (device_ext_force)
        cudaMemcpy(m_device_ext_force, device_ext_force, m_host_overall_pos.size() * sizeof(float3), cudaMemcpyDeviceToDevice);
}

void NonNewtonFluidComponentpinn::reset()
{
    m_host_cur_pos = m_host_meta_pos;
    m_host_cur_vel = m_host_meta_pos;
}

NonNewtonFluidComponentpinn::NonNewtonFluidComponentpinn(SimMaterial material)
{
    m_host_meta_material = material;
}

void NonNewtonFluidComponentpinn::addParticles(std::vector<float3> particles)
{
    m_host_meta_pos.insert(m_host_meta_pos.end(), particles.begin(), particles.end());
    m_host_cur_pos = m_host_meta_pos;
    setStartVelocity({ 0, 0, 0 });
    m_particleNum = m_host_meta_pos.size();
}

bool NonNewtonFluidComponentpinn::hasParticles() const
{
    return !m_host_meta_pos.empty();
}

void NonNewtonFluidComponentpinn::setStartVelocity(float3 vel_start)
{
    m_host_meta_vel = std::vector<float3>(m_host_meta_pos.size(), vel_start);
    m_host_cur_vel  = m_host_meta_vel;
}

SimMaterial NonNewtonFluidComponentpinn::getMaterial2() const
{
    return m_host_meta_material;
}
uint32_t NonNewtonFluidComponentpinn::getParticleNum2() const
{
    return m_particleNum;
}
void NonNewtonFluidComponentpinn::sestMaterial(SimMaterial material)
{
    m_host_meta_material = material;
}
void NonNewtonFluidComponentpinn::addInstance(ParticleModelConfig config)
{
    auto parts = ModelHelper::create3DParticleModel(config);
    addParticles(parts);
}

}  // namespace Physika