/**
 *
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-10
 * @version    : 1.0
 *
 * @file       mpm_snow.cpp
 * @brief      Functional implementation of snow simulation solver
 *
 * This file contains implementation details for the snow simulation solver. Defines various data structures and algorithms required to process and update snow simulations.
 * Including functions such as snow particle initialization, solver settings, execution of simulation steps, and data export. These features combine
 * CUDA is used to accelerate the calculation process and provides an efficient way to simulate the physical behavior of snow.
 *
 * Main classes and methods:
 * - SnowComponent: used to manage and store data of individual snow particles.
 * - MPM_Snow: Core class, responsible for initializing the simulation environment, executing simulation steps, and managing snow particles and mesh data.
 * - reset, initialize, step, run: These methods are used to control the entire life cycle of the simulation.
 * - attachObject, detachObject, isApplicable: used to manage the interaction between snow simulation components and physics objects.
 * - writeToPly: Output simulation results to PLY file format for easy visualization and analysis.

 * @dependencies: Depends on "mpm_snow.hpp" as well as the CUDA runtime library and other standard library files for basic data structures
 *                and CUDA accelerated computing.
 *
 * @note       : When modifying this document, some understanding of CUDA programming and parallel computing is required to ensure the accuracy and efficiency of the simulation.
 *
 * @remark     : This file is a critical part of the snow simulation project and is particularly critical to the performance and accuracy of the simulation.
 *               Any changes should be made carefully and fully tested.
 */
#include "mpm_snow.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <vector_functions.hpp>

#include "framework/object.hpp"

namespace Physika {
#define SNOW_GRID_SIZE 256

extern void update(
    Point* device_points,
    Grid*  device_grids,
    float* extern_force,
    float* height,
    float  unit_height,
    int    height_x_num,
    int    height_z_num,
    int    particle_num,
    int    grid_num);
extern void SetupParam(SolverParam& params);
extern void getPositionPtr(float* position, Point* dev_points, unsigned int num_particle);
extern void setPointPosition(float* position, Point* dev_points, unsigned int num_particle);
extern void getPositionPtr(float* position, Point* dev_points, unsigned int num_particle);
extern void getVelocityPtr(float* velocity, Point* dev_points, unsigned int num_particle);
extern void getPosAndVelPtr(float* position, float* velocity, Point* dev_points, unsigned int num_particle);

void MPMSnowComponent::reset()
{
    m_bInitialized = false;
}

void MPMSnowComponent::addInstance(
    ParticleModelConfig config,
    std::vector<float3> vel_start)
{
    auto cube = ModelHelper::create3DParticleModel(config);
    this->addInstance(
        cube,
        vel_start);
}

void MPMSnowComponent::addInstance(
    std::vector<float3> init_pos,
    std::vector<float3> vel_start)
{
    m_bInitialized = false;
    m_numParticles = 0;
    m_deviceGrid = nullptr;
    m_deviceParticles = nullptr;
    //if (init_pos.size() == vel_start.size())
    //{
    //    for (int i = 0; i < init_pos.size(); i++)
    //    {
    //        Point p(init_pos[i].x, init_pos[i].y, init_pos[i].z, vel_start[i].x, vel_start[i].y, vel_start[i].z);
    //        m_hParticles.push_back(p);
    //    }
    //} else
    {
        for (int i = 0; i < init_pos.size(); i++)
        {
            Point p(init_pos[i].x, init_pos[i].y, init_pos[i].z);
            m_hParticles.push_back(p);
        }
    }
    int numParticles = m_hParticles.size();
    printf("numParticles: %d\n", numParticles);
    initialize(numParticles);

}

void MPMSnowComponent::initialize(int numParticles)
{
    if (m_bInitialized)
    {
        std::cout << "Already initialized.\n";
        return;
    }

    m_numParticles       = numParticles;
    m_numGrids           = SNOW_GRID_SIZE * SNOW_GRID_SIZE * SNOW_GRID_SIZE;
    unsigned int memSize = sizeof(Point) * m_numParticles;
    cudaMalloc(( void** )&m_deviceParticles, memSize);
    cudaMemcpy(m_deviceParticles, &m_hParticles[0], memSize, cudaMemcpyHostToDevice);

    unsigned int gridMemSize = sizeof(Grid) * m_numGrids;
    cudaMalloc(( void** )&m_deviceGrid, gridMemSize);
    cudaMemset(m_deviceGrid, 0, gridMemSize);

    cudaMalloc(( void** )&m_devicePos, sizeof(float) * m_numParticles * 3);
    cudaMalloc(( void** )&m_deviceVel, sizeof(float) * m_numParticles * 3);
    cudaMalloc(( void** )&m_externelForce, sizeof(float) * m_numParticles * 3);
    cudaMemset(m_externelForce, 0, sizeof(float) * m_numParticles * 3);
    // copy data to device
    getPosAndVelPtr(m_devicePos, m_deviceVel, m_deviceParticles, m_numParticles);
    m_bInitialized = true;
}

MPMSnowComponent::~MPMSnowComponent()
{
    if (m_bInitialized)
    {
        cudaFree(m_deviceGrid);
        cudaFree(m_deviceParticles);
        cudaFree(m_devicePos);
        cudaFree(m_deviceVel);
    }
}
void MPMSnowComponent::deInit()
{
    if (m_bInitialized)
    {
        cudaFree(m_deviceGrid);
        cudaFree(m_deviceParticles);
        cudaFree(m_devicePos);
        cudaFree(m_deviceVel);
        cudaFree(m_externelForce);
    }
    m_bInitialized = false;
}

MPMSnowSolver::MPMSnowSolver(std::vector<float> world_boundary = std::vector<float>({0.0, 0.0, 0.0, 10.f, 10.f, 10.f}))
    : m_snow_particles(nullptr), m_is_init(false), m_cur_time(0)
{
    float word_size      = 30.f;
    int   n_grid         = SNOW_GRID_SIZE;
    m_params.alpha       = 0.99f;
    m_params.dt          = 4e-4;
    m_params.compression = 0.05f;
    m_params.stretch     = 0.0075f;
    m_params.young       = 2e5;
    m_params.poisson     = 0.2;
    m_params.hardening   = 5.f;
    m_params.gridSize    = make_int3(n_grid, n_grid, n_grid);

    m_params.stickyWalls   = false;
    m_params.frictionCoeff = 0.2f;
    m_params.lambda        = m_params.young * m_params.poisson / ((1 + m_params.poisson) * (1 - 2 * m_params.poisson));
    m_params.mu            = m_params.young / (2 * (1 + m_params.poisson));

    m_params.gravity    = make_float3(0, -9.8, 0);
    m_params.boxCorner1 = make_float3(world_boundary[0], world_boundary[1], world_boundary[2]);
    m_params.boxCorner2 = make_float3(world_boundary[3], world_boundary[4], world_boundary[5]);
    m_params.cellSize   = word_size / float(n_grid);
    SetupParam(m_params);

    m_unit_height  = m_params.cellSize;
    m_height_x_num = static_cast<int>((m_params.boxCorner2.x - m_params.boxCorner1.x) / m_unit_height);
    m_height_z_num = static_cast<int>((m_params.boxCorner2.z - m_params.boxCorner1.z) / m_unit_height);
    m_host_height.insert(m_host_height.end(), m_height_x_num * m_height_z_num, m_params.boxCorner1.y);
}

MPMSnowSolver::~MPMSnowSolver()
{
    cudaFree(m_device_height);
}

bool MPMSnowSolver::initialize()
{
    if (m_is_init)
        return true;
    if (m_snow_particles == nullptr)
    {
        std::cout << "ERROR: Must set snow particles first.\n";
        return false;
    }
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
    {
        std::cout << "ERROR: no snow component.\n";
        return false;
    }

    cudaMalloc(( void** )&m_device_height, sizeof(float) * m_height_x_num * m_height_z_num);
    cudaMemcpy(m_device_height, m_host_height.data(), sizeof(float) * m_height_x_num * m_height_z_num, cudaMemcpyHostToDevice);

    m_is_init = true;
    std::cout << "snow solver initialized successfully.\n";
    return true;
}

bool MPMSnowSolver::isInitialized() const
{
    return m_is_init;
}

bool MPMSnowSolver::reset()
{
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (m_snow_particles == nullptr)
    {
        std::cout << "    error: no snow particles attached.\n";
        return false;
    }
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    unsigned int      memSize        = sizeof(Point) * snow_component->m_numParticles;
    cudaMemcpy(snow_component->m_deviceParticles, &(snow_component->m_hParticles[0]), memSize, cudaMemcpyHostToDevice);

    unsigned int gridMemSize = sizeof(Grid) * snow_component->m_numGrids;
    cudaMalloc(( void** )&(snow_component->m_deviceGrid), gridMemSize);
    cudaMemset(snow_component->m_deviceGrid, 0, gridMemSize);
    return true;
}

bool MPMSnowSolver::step()
{
    if (m_snow_particles == nullptr)
        return false;
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return false;

    SetupParam(m_params);

    update(snow_component->m_deviceParticles,
           snow_component->m_deviceGrid,
           snow_component->m_externelForce,
           m_device_height,
           m_unit_height,
           m_height_x_num,
           m_height_z_num,
           snow_component->m_numParticles,
           snow_component->m_numGrids);
    if (m_config.m_showGUI)
    {
        getPosAndVelPtr(snow_component->m_devicePos,
                        snow_component->m_deviceVel,
                        snow_component->m_deviceParticles,
                        snow_component->m_numParticles);
    }
    return true;
}

bool MPMSnowSolver::run()
{
    if (!m_is_init)
    {
        std::cout << "Must initialized first.\n";
        return false;
    }
    if (m_snow_particles == nullptr)
        return false;

    int step_id = 0;
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        std::cout << "    step " << step_id << ": " << m_cur_time << " -> " << m_cur_time + dt << "\n";
        m_cur_time += dt;
        m_config.m_dt = dt;
        int substep   = 0;
        // record start time, calculate the time used for each step
        auto start = std::chrono::high_resolution_clock::now();
        while (substep < 10)
        {
            substep++;
            step();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "    step " << step_id << " finished in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms.\n";
        // cout frame rate
        std::cout << "    frame rate: " << 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " fps.\n";
        ++step_id;
        if (m_config.m_write2ply)
        {
            MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
            getPosAndVelPtr(snow_component->m_devicePos,
                            snow_component->m_deviceVel,
                            snow_component->m_deviceParticles,
                            snow_component->m_numParticles);
            writeToPly(step_id);
        }
    }
    return true;
}

bool MPMSnowSolver::isApplicable(const Object* object) const
{
    if (!object)
        return false;

    return object->hasComponent<MPMSnowComponent>();
}

bool MPMSnowSolver::attachObject(Object* object)
{
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<MPMSnowComponent>())
    {
        std::cout << "object attached as snow particle system.\n";
        m_snow_particles                 = object;
        MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
        m_hostParticles.resize(snow_component->m_numParticles * 3);
        // m_params.particleCorner1 = make_float3(snow_component->m_boundary[0], snow_component->m_boundary[1], snow_component->m_boundary[2]);
        // m_params.particleCorner2 = make_float3(snow_component->m_boundary[3], snow_component->m_boundary[4], snow_component->m_boundary[5]);
        // SetupParam(m_params);
    }
    initialize();
    return true;
}

bool MPMSnowSolver::detachObject(Object* object)
{
    if (!object)
        return false;

    if (m_snow_particles == object)
    {
        m_snow_particles = nullptr;
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void MPMSnowSolver::clearAttachment()
{
    m_snow_particles = nullptr;
}

void MPMSnowSolver::config(const SolverConfig& config)
{
    m_config = config;
}

float MPMSnowSolver::getYoungsModulus() const
{
    return m_params.young;
}

float MPMSnowSolver::getPoissonRatio() const
{
    return m_params.poisson;
}

float MPMSnowSolver::getHardeningCeoff() const
{
    return m_params.hardening;
}

float MPMSnowSolver::getCompressionCoeff() const
{
    return m_params.compression;
}

float MPMSnowSolver::getFrictionCoeff() const
{
    return m_params.frictionCoeff;
}

float MPMSnowSolver::getStretch() const
{
    return m_params.stretch;
}

bool MPMSnowSolver::getIfStick() const
{
    return m_params.stickyWalls;
}

bool MPMSnowSolver::getParticlePositionPtr(float* pos, unsigned int& numParticles)
{
    if (m_snow_particles == nullptr)
        return false;

    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return false;
    getPositionPtr(pos, snow_component->m_deviceParticles, snow_component->m_numParticles);
    numParticles = snow_component->m_numParticles;
    return true;
}

void MPMSnowSolver::setHeightField(const std::vector<float>& height, const float unit_height, const int height_x_num, const int height_z_num)
{
    m_host_height.clear();
    m_unit_height  = unit_height;
    m_height_x_num = height_x_num;
    m_height_z_num = height_z_num;
    m_host_height.insert(m_host_height.end(), height.begin(), height.end());
}

void MPMSnowSolver::setHeightField(float*& height, const float unit_height, const int height_x_num, const int height_z_num)
{
    m_unit_height   = unit_height;
    m_height_x_num  = height_x_num;
    m_height_z_num  = height_z_num;
    m_device_height = height;
    m_host_height.clear();
    m_host_height.resize(height_x_num * height_z_num);
    cudaMemcpy(m_host_height.data(), m_device_height, sizeof(float) * height_x_num * height_z_num, cudaMemcpyDeviceToHost);
}

void MPMSnowSolver::setYoungsModulus(const float& youngs_modulus)
{
    m_params.young = youngs_modulus;
}

void MPMSnowSolver::setGridBoundary(const float& x_min, const float& x_max, const float& y_min, const float& y_max, const float& z_min, const float& z_max)
{
    m_params.boxCorner1 = make_float3(x_min, y_min, z_min);
    m_params.boxCorner2 = make_float3(x_max, y_max, z_max);
    m_params.cellSize   = (x_max - x_min) / SNOW_GRID_SIZE;
    SetupParam(m_params);
}

void MPMSnowSolver::setPoissonRatio(const float& poisson_ratio)
{
    m_params.poisson = std::clamp(poisson_ratio, 0.f, 0.4999f);
}

void MPMSnowSolver::setHardeningCeoff(const float& hardening)
{
    m_params.hardening = std::clamp(hardening, 0.f, 20.f);
}

void MPMSnowSolver::setCompressionCoeff(const float& compress)
{
    m_params.compression = std::clamp(compress, 0.f, 0.1f);
}

void MPMSnowSolver::setFrictionCoeff(const float& frictionCoeff)
{
    m_params.frictionCoeff = std::clamp(frictionCoeff, 0.f, 1.f);
}

void MPMSnowSolver::setStretch(const float& stretch)
{
    m_params.stretch = std::clamp(stretch, 0.f, 0.001f);
}

void MPMSnowSolver::setStick(const bool& stick)
{
    m_params.stickyWalls = stick;
}

void MPMSnowSolver::setParticlePositionFromdev(float* pos, unsigned int numParticles)
{
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return;
    setPointPosition(pos, snow_component->m_deviceParticles, snow_component->m_numParticles);
}

void MPMSnowSolver::setParticle(const std::vector<Point>& pos)
{
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return;
    snow_component->m_hParticles   = pos;
    snow_component->m_numParticles = pos.size();
    snow_component->deInit();
    snow_component->initialize(snow_component->m_numParticles);
}

void MPMSnowSolver::setParticleVelocity(const std::vector<float>& vel)
{
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return;
    if (vel.size() != snow_component->m_numParticles * 3)
    {
        std::cerr << "Error: velocity size not match.\n" << std::endl;
        return;
    }
    for(int i = 0; i < snow_component->m_numParticles; i++)
    {
        snow_component->m_hParticles[i].m_velocity = make_float3(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
    }
    snow_component->deInit();
    snow_component->initialize(snow_component->m_numParticles);
}

void MPMSnowSolver::setParticlePosition(const std::vector<float>& pos)
{
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    if (snow_component == nullptr)
        return;
    if (pos.size() / 3 != 9)
    {
        std::cerr << "Error: position size not match.\n"
                  << std::endl;
        return;
    }
    snow_component->m_numParticles = pos.size() / 3;
    snow_component->m_hParticles.clear();
    snow_component->m_hParticles.resize(snow_component->m_numParticles);
    for (int i = 0; i < snow_component->m_numParticles; i++)
    {
        snow_component->m_hParticles[i].m_position = make_float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    snow_component->deInit();
    snow_component->initialize(snow_component->m_numParticles);
}

void MPMSnowSolver::setWorldBoundary(const float& x_min, const float& x_max, const float& y_min, const float& y_max, const float& z_min, const float& z_max)
{
    m_params.boxCorner1 = make_float3(0, 0, 0);
	m_params.boxCorner2 = make_float3(10, 10, 10);
    m_params.cellSize = (x_max - x_min) / SNOW_GRID_SIZE;
}

void MPMSnowSolver::writeToPly(const int& step_id)
{
    MPMSnowComponent* snow_component = m_snow_particles->getComponent<MPMSnowComponent>();
    cudaMemcpy(m_hostParticles.data(), snow_component->m_devicePos, sizeof(float) * snow_component->m_numParticles * 3, cudaMemcpyDeviceToHost);
    // write to ply file
    std::string filename = "./ply/snow_" + std::to_string(step_id) + ".ply";
    // write pos ti ply file
    std::ofstream outfile(filename);
    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << snow_component->m_numParticles << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property uchar red\n";
    outfile << "property uchar green\n";
    outfile << "property uchar blue\n";
    outfile << "end_header\n";
    for (int i = 0; i < snow_component->m_numParticles; i++)
    {
        float3 pos = make_float3(m_hostParticles[i * 3], m_hostParticles[i * 3 + 1], m_hostParticles[i * 3 + 2]);
        outfile << pos.x << " " << pos.y << " " << pos.z << " 255 255 255\n";
    }
    outfile.close();
}

}  // namespace Physika