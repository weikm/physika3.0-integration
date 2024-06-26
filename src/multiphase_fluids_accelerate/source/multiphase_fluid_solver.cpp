/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: implementation of MultiphaseFluidSolver class, a sample solver that demonstrates
 *               how to define a solver that needs objects of different kind to run
 * @version    : 1.0
 */
#pragma warning(disable : 26495)
#pragma warning(disable : 4244)
#pragma warning(disable : 4101)
#pragma warning(disable : 26451)
#pragma warning(disable : 4305)

#include "framework/object.hpp"
#include "multiphase_fluids_accelerate/interface/multiphase_fluid_solver.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

#include <vector_functions.hpp>

namespace Physika {

MultiphaseFluidSolver::MultiphaseFluidSolver()
    : Solver(), m_is_init(false), m_fluid(nullptr), m_gas(nullptr), m_boundary(nullptr), m_cur_time(0.0)
{
    m_position = 0x0;
    m_velocity = 0x0;
    m_type     = 0x0;

    m_num_particle     = 0;
    m_num_fluid        = 0;
    m_num_bound        = 0;
    m_num_rigid        = 0;
    m_num_bubble       = 0;
    m_max_num_particle = 0;
    m_test_index       = -1;

    m_unit_size    = 3 + 1;
    m_iteration    = 0;
    m_output_frame = 0;
}

MultiphaseFluidSolver::~MultiphaseFluidSolver()
{
    this->reset();
}

bool MultiphaseFluidSolver::initialize()
{
    std::cout << "MultiphaseFluidSolver::initialize() initializes the solver.\n";
    config(m_config);
    setParametersAll();
    AllocateParticleMemory(m_param[MAXNUMBER]);
    if (m_flag[HIGHRESOLUTION])
        EnableHighResolution();

    float3 rigid_center = make_float3(m_vec[RIGIDCENTER].x / m_param[SIMSCALE], (m_vec[RIGIDCENTER].y + m_vec[RIGIDDRIFT].y) / m_param[SIMSCALE], m_vec[RIGIDCENTER].z / m_param[SIMSCALE]);

    if (m_flag[LOADBOUND])
        InitialiseBoundary();
    if (m_flag[LOADRIGIDROTATE])
        InitialiseRotateRigid();
    if (m_flag[LOADRIGID])
        InitialiseRigid();

    if (m_flag[LOADGENERATEFORM])
        InitialisePreLoadParticles();
    else
        InitialisePositionPhase();

    SetKernels();
    SetTimeStep();
    RefineGridResolution();
    AllocateGrid();
    TransferParticleToSolver();
    m_fluid->getComponent<MultiphaseFluidComponent>()->pnum = m_num_particle;

    int3 grid_resolution = make_int3(m_vec_int[GRIDRESOLUTION].x, m_vec_int[GRIDRESOLUTION].y, m_vec_int[GRIDRESOLUTION].z);
    SetupParticlesCUDA(m_num_particle, grid_resolution, m_grid_number, m_param[PKERNELSELF], m_param[CONTAINERSIZE], m_mc_grid_resolution, m_mc_grid_number, m_max_num_particle);

    // coordinate transformation
    float3 boundary_min         = make_float3(m_vec[BOUNDARYMIN].x / m_param[SIMSCALE], m_vec[BOUNDARYMIN].y / m_param[SIMSCALE], m_vec[BOUNDARYMIN].z / m_param[SIMSCALE]);
    float3 boundary_max         = make_float3(m_vec[BOUNDARYMAX].x / m_param[SIMSCALE], m_vec[BOUNDARYMAX].y / m_param[SIMSCALE], m_vec[BOUNDARYMAX].z / m_param[SIMSCALE]);
    float3 grid_boundary_offset = make_float3(m_vec[GRIDBOUNDARYOFFSET].x / m_param[SIMSCALE], m_vec[GRIDBOUNDARYOFFSET].y / m_param[SIMSCALE], m_vec[GRIDBOUNDARYOFFSET].z / m_param[SIMSCALE]);
    float3 gravity              = make_float3(m_vec[GRAVITY].x, m_vec[GRAVITY].y, m_vec[GRAVITY].z);
    SetParametersCUDA(boundary_min, boundary_max, grid_boundary_offset, gravity, m_param[PMASS], m_param[PRESTDENSITY], m_param[PGASCONSTANT], m_param[PVISC], 0.5f, m_param[SIMSCALE], m_param[SMOOTHRADIUS], m_param[PARTICLERADIUS] / m_param[SIMSCALE], m_param[GRIDRADIUS], (!m_flag[IMPLICIT]), ( int )(m_param[TESTINDEX]), rigid_center, m_param[SMOOTHFACTOR], m_param[FACTORKN], m_param[FACTORKR], m_param[FACTORKS], ( int )m_param[NEIGHBORTHRESHOLD], m_param[SURFACETENSIONFACTOR], m_mc_grid_radius / m_param[SIMSCALE]);

    SetMFParametersCUDA(m_multi_phase + MFDENSITY * MAX_PHASE_NUMBER, m_multi_phase + MFRESTMASS * MAX_PHASE_NUMBER, m_multi_phase + MFVISCOSITY * MAX_PHASE_NUMBER, m_param[PHASENUMBER], m_param[DRIFTVELTAU], m_param[DRIFTVELSIGMA], m_flag[MISCIBLE], m_multi_phase_vec + MFCOLOR * MAX_PHASE_NUMBER);

    SetMcParametersCUDA(m_param[INTERPOLATERADIUS] * m_param[SMOOTHRADIUS], m_param[ANISOTROPICRADIUS], m_param[FACTORKN], m_param[FACTORKR], m_param[FACTORKS]);

    TransferToCUDA();

    // TransferParticleToSolver();//dont change anymore
    std::cout << "before compute test position"
              << m_position[123547].x << std::endl;
    std::cout << "m_total_time at config: " << m_config.m_total_time << std::endl;
    m_is_init = true;
    return true;
}

bool MultiphaseFluidSolver::isInitialized() const
{
    std::cout << "MultiphaseFluidSolver::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool MultiphaseFluidSolver::reset()
{
    std::cout << "MultiphaseFluidSolver::reset() sets the solver to newly constructed state.\n";
    m_is_init             = false;
    m_config.m_dt         = 0.0;
    m_config.m_total_time = 0.0;
    m_fluid               = nullptr;
    m_rigid               = nullptr;
    m_boundary            = nullptr;
    m_gas                 = nullptr;
    m_cur_time            = 0.0;
    return true;
}

bool MultiphaseFluidSolver::step()
{
    // std::cout << "MultiphaseFluidSolver::step() updates the solver state by a time step.\n";
    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_fluid /*|| !m_gas*/)
    {
        std::cout << "    error: gas or fluid not attached to the solver.\n";
        return false;
    }
    float* m_gpu_pos   = m_fluid->getComponent<MultiphaseFluidComponent>()->m_device_pos;
    float* m_gpu_color = m_fluid->getComponent<MultiphaseFluidComponent>()->m_color;
    // change:
    float time_step = m_param[TIMESTEP];
    tick(1);
    tick(0);
    InsertParticlesCUDA(m_num_particle);
    tock("InsertParticle", 0);

    tick(0);
    PrefixSumCellsCUDA(1);
    tock("PrefixSum", 0);

    tick(0);
    CountingSortFullCUDA(m_num_particle);
    tock("CountingSort", 0);

    tick(0);
    ComputeShapeCUDA(time_step, m_num_particle);
    tock("ComputeShape", 0);

    tick(0);
    ComputeAlphaCUDA(time_step, m_num_particle);
    tock("ComputeAlpha", 0);

    tick(0);
    ComputeForceCUDA(time_step, m_num_particle);
    tock("ComputeForce", 0);

    tick(0);
    AdvanceParticleCUDA(time_step, m_num_particle, m_iteration++);
    CopyToComponent(m_gpu_pos, m_gpu_color, m_num_particle);
    tock("AdvanceParticle", 0);
    tock("total", 1); /**/
    std::cout << "Sim Frame: " << m_iteration << std::endl;
    return true;
}

bool MultiphaseFluidSolver::run()
{
    std::cout << "MultiphaseFluidSolver::run() updates the solver till termination criteria are met.\n";

    if (!m_is_init)
    {
        std::cout << "    error: solver not initialized.\n";
        return false;
    }
    if (!m_fluid)
    {
        std::cout << "    error: gas or fluid not attached to the solver.\n";
        return false;
    }
    // Update till termination
    int step_id = 0;
    std::cout << "m_cur_time: " << m_cur_time << "m_total_time" << m_config.m_total_time << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while (m_cur_time < m_config.m_total_time)
    {
        double dt = (m_cur_time + m_config.m_dt <= m_config.m_total_time) ? m_config.m_dt : m_config.m_total_time - m_cur_time;
        // Do the step here
        // step();

        std::cout << "    step " << step_id << ": " << m_cur_time << " -> " << m_cur_time + dt << "\n\n";
        //std::cout << "test position" << m_position[123547].x << std::endl;
        m_cur_time += dt;
        ++step_id;
    }
    std::cout << "MultiphaseFluidSolver::run() applied to fluid " << m_fluid->id() << ".\n";
    auto    end     = std::chrono::high_resolution_clock::now();
    __int64 subtime = (end - start).count();
    double  fps     = subtime / step_id / 1000000000.0;
    std::cout << "run end " << std::endl;
    std::cout << "--------------------TEST-------------------------\n";
    std::cout << "solver finished,average fps: " << fps << "frame/second" << std::endl;
    std::cout << "--------------------TEST-------------------------\n";
    return true;
}

bool MultiphaseFluidSolver::isApplicable(const Object* object) const
{
    std::cout << "MultiphaseFluidSolver::isApplicable() checks if object has GasParticleComponent or FluidParticleComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<MultiphaseFluidComponent>();
}

bool MultiphaseFluidSolver::attachObject(Object* object)
{
    std::cout << "MultiphaseFluidSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "    error: object is not applicable.\n";
        return false;
    }

    if (object->hasComponent<MultiphaseFluidComponent>())
    {
        std::cout << "    object attached as fluid.\n";
        m_fluid = object;
    }
    /* if (object->hasComponent<GasParticleComponent>())
     {
         std::cout << "    object attached as gas.\n";
         m_gas = object;
     }*/

    return true;
}

bool MultiphaseFluidSolver::detachObject(Object* object)
{
    std::cout << "MultiphaseFluidSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_fluid == object)
    {
        m_fluid = nullptr;
        std::cout << "    Fluid detached.\n";
        return true;
    }
    return false;
}

void MultiphaseFluidSolver::clearAttachment()
{
    std::cout << "MultiphaseFluidSolver::clearAttachment() clears the target list of the solver.\n";
    m_gas      = nullptr;
    m_fluid    = nullptr;
    m_boundary = nullptr;
    m_rigid    = nullptr;
}

void MultiphaseFluidSolver::setParametersAll()
{
    const float simscale = 40.0f;

    // float param
    // -----------
    m_param[PSPACINGREALWORLD]     = 0.025f;             // 1		 //	1.0f / 40.0f
    m_param[PSPACINGGRAPHICSWORLD] = 1.0f;               // 2      // intervals between two particles
    m_param[PRESTDENSITY]          = 1000.0f;            // 3
    m_param[MAXNUMBER]             = 200 * 100 * 100;    // 4
    m_param[SIMSCALE]              = simscale;           // 5      // SCALEVALUE      graphics_pos = SCALEVALUE * world_pos
    m_param[PARTICLERADIUS]        = 0.5f;               // 6      // particle drawn radius
    m_param[TIMESTEP]              = 0.001f;             // 7
    m_param[GRIDRADIUS]            = 2.5f / simscale;    // 8		 // GRIDRADIUS
    m_param[MAXGRIDNUMBER]         = 200 * 200 * 200;    // 9
    m_param[MINNEIGHBORNUMBER]     = 0;                  // 10
    m_param[SMOOTHRADIUS]          = 2.0f / simscale;    // 11     // change this value "h" with GRIDRADIUS
    m_param[MAXNEIGHBORPARTICLE]   = 500;                // 12
    m_param[PMASS]                 = 0.015625f;          // 14     // MASS = RESTDENSITY * r * r * r
    m_param[PGASCONSTANT]          = 50.0f;              // 15
    m_param[PVISC]                 = 0.0f;               // 16
    m_param[FMASS]                 = 0.015625f * 0.01f;  // 17
    m_param[SMASS]                 = 0.015625f * 1.0f;   // 18
    m_param[TMASS]                 = 0.015625f * 0.5f;   // 19
    m_param[FDENSITY]              = 100.0f;             // 20
    m_param[SDENSITY]              = 1000.0f;            // 21
    m_param[TDENSITY]              = 500.0f;             // 22
    m_param[FVISC]                 = 0.05f;              // 23
    m_param[SVISC]                 = 9.0f;               // 24
    m_param[TVISC]                 = 5.0f;               // 25
    m_param[PHASENUMBER]           = MAX_PHASE_NUMBER;   // 26
    m_param[DRIFTVELTAU]           = 0.000001f * 2.0f;   // 27
    m_param[DRIFTVELSIGMA]         = 0.001f * 0.5f;      // 28
    m_param[MODE]                  = 1;                  // 29	 // 1 means muti-phase, change this value with _flag[MUTIPHASE]
    m_param[BOUNDOFFSET]           = 3.0f;               // 30
    m_param[BOUNDGAMMA]            = 0.7f;               // 31
    m_param[BOUNDBETA]             = 0.15f;              // 32
    m_param[BOUNDOMEGA]            = 0.5f;               // 33
    m_param[TESTINDEX]             = -1;                 // 34
    m_param[KD]                    = 3.0f;               // 35
    m_param[MAXLOADNUMBER]         = 80 * 100 * 100;     // 36
    m_param[POINTNUMBER]           = 89485;              // 37
    m_param[RIGIDSCALE]            = 1.0f;               // 38
    m_param[CONTAINERSIZE]         = 2;                  // 39
    m_param[SMOOTHFACTOR]          = 0.95f;              // 40
    m_param[FACTORKN]              = 0.5f;               // 41
    m_param[FACTORKR]              = 4.0f;               // 42
    m_param[FACTORKS]              = 1500.0f;            // 43
    m_param[NEIGHBORTHRESHOLD]     = 3.0f;               // 44
    m_param[SURFACETENSIONFACTOR]  = 0.0f;               // 45
    m_param[MCUBETHRESHOLD]        = 2.0f;               // 46
    m_param[MCUBEVOXEL]            = 0.25f;              // 47	// cell size = SIMSCALE * SMOOTHRADIUS * MCUBEVOXEL
    m_param[POINTNUMBERROTATE]     = 33609;              // 48
    m_param[RIGIDSCALEROTATE]      = 0.3f;               // 49	// final scale factor = RIGIDSCALEROTATE * RIGIDSCALE
    m_param[INTERPOLATERADIUS]     = 1.0f;               // 50
    m_param[ANISOTROPICRADIUS]     = 2.0f / simscale;    // 51
    m_param[GENERATEFRAMERATE]     = 5.0f;               // 52
    m_param[GENERATENUM]           = 4.0f;               // 53
    m_param[GENERATEPOSENUM]       = 1.0f;               // 54
    m_param[RIGIDROTATEOMEGA]      = 0.0f;               // 55
    m_param[REACTIONRATE]          = 0.0f;               // 56

    // vec param
    // ---------
    m_vec[BOUNDARYMIN].Set(-50.0, -10.0, -50.0);             // 1		 // Boundary in graphics world
    m_vec[BOUNDARYMAX].Set(50.0, 20.0, 50.0);                // 2		 // Boundary in graphics world
    m_vec[INITIALVOLUMEMIN].Set(-25.0, -5.0, -30.0);         // 3		 // Fluid bound in graphics world
    m_vec[INITIALVOLUMEMAX].Set(0.0, 15.0, 30.0);            // 4		 // Fluid bound in graphics world
    m_vec[GRIDBOUNDARYOFFSET].Set(5.0, 5.0, 5.0);            // 5
    m_vec[GRAVITY].Set(0.0, -9.8, 0.0);                      // 6
    m_vec[INITIALVOLUMEMINPHASE].Set(5.0, -5.0, -30.0);      // 7		 // Muti-fluid bound in graphics world
    m_vec[INITIALVOLUMEMAXPHASE].Set(30.0, 15.0, 30.0);      // 8		 // Muti-fluid bound in graphics world
    m_vec[RIGIDVOLUMEMIN].Set(-15.0, -18.0, -1.0);           // 9
    m_vec[RIGIDVOLUMEMAX].Set(-13.0, 15.0, 1.0);             // 10
    m_vec[INITIALVOLUMEMINPHASES].Set(-15.0, -27.0, -35.0);  // 11
    m_vec[INITIALVOLUMEMAXPHASES].Set(15.0, -5.0, 35.0);     // 12
    m_vec[RIGIDCENTER].Set(0.0f, 0.0f, 0.0f);                // 13
    m_vec[RIGIDDRIFT].Set(0.0f, 0.0f, 0.0f);                 // 14
    m_vec[RIGIDDRIFTROTATE].Set(35.0f, 10.0f, 0.0f);         // 15
    m_vec[RIGIDSCALE].Set(15.0f, 15.0f, 15.0f);              // 17
    m_vec[RIGIDSCALEROTATE].Set(0.04f, 0.24f, 0.04f);        // 18

    // bool param
    // ----------
    m_flag[LOADRIGID]         = false;  // 1
    m_flag[LOADBOUND]         = true;   // 2
    m_flag[MUTIPHASE]         = true;   // 3
    m_flag[IMPLICIT]          = false;  // 4
    m_flag[MISCIBLE]          = false;  // 5
    m_flag[LOADRIGIDROTATE]   = false;  // 6
    m_flag[HIGHRESOLUTION]    = true;   // 13
    m_flag[LOADGENERATEFORM]  = false;  // 19		// turn on this to generate particles
    m_flag[RIGIDDRIFT]        = false;  // 21
    m_flag[RENDERRIGID]       = false;  // 28
    m_flag[DISABELTRANSFER]   = false;  // 29

    // multiphase properties parameter
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + 0]   = 800.0f;            // 1 1
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + 1]   = 1000.0f;           // 1 2
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + 2]   = 1000.0f;           // 1 3
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + 3]   = 100.0f;            // 1 4
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + 4]   = 300.0f;            // 1 5
    m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + 0] = 4.5f;              // 2 1
    m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + 1] = 6.5f;              // 2 2
    m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + 2] = 6.0f;              // 2 3
    m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + 3] = 0.8f;              // 2 4
    m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + 4] = 2.1f;              // 2 5
    m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + 0]  = 0.015625f * 0.8f;  // 3 1
    m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + 1]  = 0.015625f * 1.0f;  // 3 2
    m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + 2]  = 0.015625f * 1.0f;  // 3 3
    m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + 3]  = 0.015625f * 0.1f;  // 3 4
    m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + 4]  = 0.015625f * 0.3f;  // 3 5

    // multiphase properties vector
    m_multi_phase_vec[MFCOLOR * MAX_PHASE_NUMBER + 0] = make_float3(0.1f, 0.1f, 0.8f);  // 1 1
    m_multi_phase_vec[MFCOLOR * MAX_PHASE_NUMBER + 1] = make_float3(0.1f, 0.8f, 0.1f);  // 1 2
    m_multi_phase_vec[MFCOLOR * MAX_PHASE_NUMBER + 2] = make_float3(1.0f, 0.0f, 0.0f);  // 1 3
    m_multi_phase_vec[MFCOLOR * MAX_PHASE_NUMBER + 3] = make_float3(1.0f, 1.0f, 1.0f);  // 1 4
    m_multi_phase_vec[MFCOLOR * MAX_PHASE_NUMBER + 4] = make_float3(0.0f, 0.1f, 1.0f);  // 1 5
}
void MultiphaseFluidSolver::setBoundaryParam(Vector3DF max_corner, Vector3DF min_corner)
{
    m_vec[BOUNDARYMAX] = max_corner;
    m_vec[BOUNDARYMIN] = min_corner;
}

Vector3DF MultiphaseFluidSolver::getBoundaryParamMax()
{
    return m_vec[BOUNDARYMAX];
}

Vector3DF MultiphaseFluidSolver::getBoundaryParamMin()
{
    return m_vec[BOUNDARYMIN];
}

void MultiphaseFluidSolver::setPhaseDensityParam(int phase, float density)
{
    m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + phase];
}

float MultiphaseFluidSolver::getPhaseDensityParam(int phase)
{
    return m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + phase];
}

void MultiphaseFluidSolver::setParameterFloat(int name, float value)
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return;
    }
    else
        m_param[name] = value;
}
float MultiphaseFluidSolver::getParameterFloat(int name) const
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return 0.0f;
    }
    else
        return m_param[name];
}
void MultiphaseFluidSolver::setParameterVec3F(int name, float x, float y, float z)
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return;
    }
    m_vec[name].Set(x, y, z);
}
Vector3DF MultiphaseFluidSolver::getParameterVec3F(int name) const
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return Vector3DF(0.0f, 0.0f, 0.0f);
    }
    return m_vec[name];
}
void MultiphaseFluidSolver::setParameterVec3I(int name, int x, int y, int z)
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return;
    }
    m_vec_int[name].Set(x, y, z);
}
Vector3DI MultiphaseFluidSolver::getParameterVec3I(int name) const
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return Vector3DI(0, 0, 0);
    }
    return m_vec_int[name];
}

void MultiphaseFluidSolver::setParameterMultiphaseFloat(int name, float value)
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return;
    }
    m_multi_phase[name] = value;
}
float MultiphaseFluidSolver::getParameterMultiphaseFloat(int name) const
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return 0.0f;
    }
    return m_multi_phase[name];
}
void MultiphaseFluidSolver::setParameterMultiphaseVec3F(int name, float x, float y, float z)
{
    if (name >= MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return;
    }
    m_multi_phase_vec[name] = make_float3(x, y, z);
}
float3 MultiphaseFluidSolver::getParameterMultiphaseVec3F(int name) const
{
    if (name > MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return m_multi_phase_vec[name];
}
void MultiphaseFluidSolver::config(const MultiphaseFluidSolver::SolverConfig& config)
{
    std::cout << "MultiphaseFluidSolver::config() setups the configuration of the solver.\n";
    m_config = config;
}
void MultiphaseFluidSolver::setParameterBool(int name, bool value) {
    if (name > MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return ;
    }
    m_flag[name] = value;
}
bool MultiphaseFluidSolver::getParameterBool(int name) const{
    if (name > MAX_PARAM_NUM || name < 0)
    {
        std::cout << "false name,need check" << std::endl;
        return false;
    }
    return m_flag[name];
}
void MultiphaseFluidSolver::TransferParticleToSolver()
{
    auto fluid_mpos  = m_fluid->getComponent<MultiphaseFluidComponent>()->m_pos;
    auto fluid_pnum  = m_fluid->getComponent<MultiphaseFluidComponent>()->load_num;
    auto fluid_alpha = m_fluid->getComponent<MultiphaseFluidComponent>()->m_alpha;
    // auto  fluid_mix_velocity = m_fluid->getComponent<FluidParticleComponent>()->m_mix_velocity;
    ////auto  sand_mpos = m_fluid->getComponent<GasParticleComponent>()->m_pos;
    ////auto  sand_pnum = m_fluid->getComponent<GasParticleComponent>()->pnum;
    ////auto  gas_mix_velocity = m_fluid->getComponent<GasParticleComponent>()->m_mix_velocity;
    int current_num = m_num_bound;
    // bool not_preload_p = !_flag[LOADGENERATEFORM];
    float inv_V = m_param[PRESTDENSITY] / m_param[PMASS];
    if (m_flag[HIGHRESOLUTION])
        inv_V *= 8.0f;
    int iter = 0;
    std::cout << "current_num: " << current_num << std::endl;
    std::cout << "fluid_load: " << fluid_pnum << std::endl;
    for (int idx = current_num; idx < current_num + fluid_pnum; idx++)
    {
        float rest_mass = 0.0f, rest_density = 0.0f;
        // position * 3
        m_position[idx].Set(fluid_mpos[iter * 3 + 0], fluid_mpos[iter * 3 + 1], fluid_mpos[iter * 3 + 2]);
        /*  if (idx%2000==0)
              std::cout << "idx: " << idx << "pos: " << fluid_mpos[iter * 3 + 0] << " " <<
              fluid_mpos[iter * 3 + 1] << " " << fluid_mpos[iter * 3 + 2] << " " <<std::endl;*/
        // velocity * 3
        m_external_force[idx].Set(0.0, 0.0, 0.0);
        m_mix_velocity[idx].Set(0, 0, 0);
        // radius
        m_particle_radius[idx] = 0.0125;

        // alpha
        for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
        {
            m_alpha_advanced[idx * MAX_PHASE_NUMBER + fcount] = fluid_alpha[iter * MAX_PHASE_NUMBER + fcount];
            rest_mass += fluid_alpha[iter * MAX_PHASE_NUMBER + fcount] * m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + fcount];
            rest_density += fluid_alpha[iter * MAX_PHASE_NUMBER + fcount] * m_multi_phase[MFDENSITY * MAX_PHASE_NUMBER + fcount];
        }
        // particle mass
        m_restmass[idx] = rest_mass;
        // rest_density
        m_mrest_density[idx] = rest_density;
        m_active[idx]        = true;
        if (iter < m_param[POINTNUMBER])
            m_type[idx] = RIGID;
        else
            m_type[idx] = FLUID;
        //_type[idx]   = FLUID;
        m_render[idx] = true;
        iter++;
        m_num_fluid++;
        // if (not_preload_p)
        m_num_particle++;
    }
    // sand
    // iter = 0;
    // for (int idx = current_num + fluid_pnum; idx < current_num + fluid_pnum + sand_pnum; idx++)
    //{
    //    // position * 3
    //    _position[idx].Set(sand_mpos[iter * 3 + 0] / _param[SIMSCALE], sand_mpos[iter * 3 + 1] / _param[SIMSCALE], sand_mpos[iter * 3 + 2] / _param[SIMSCALE]);
    //    // velocity * 3
    //    _mix_velocity[idx].Set(0, 0, 0);
    //    // radius
    //    _particle_radius[idx] = 0.00625;
    //    // particle mass
    //    _restmass[idx] = 0.00195312 * 1.2;
    //    // rest_density
    //    _mrest_density[idx] = _restmass[idx] * inv_V;
    //    // alpha
    //    for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
    //    {
    //        _alpha_advanced[idx * MAX_PHASE_NUMBER + fcount] = 1;
    //        if (fcount == 1)
    //        {
    //            _alpha_advanced[idx * MAX_PHASE_NUMBER + fcount] = 0;
    //        }
    //    }
    //    _active[idx] = true;
    //    _type[idx] = FLUID;
    //    _render[idx] = true;
    //    iter++;
    //    num_fluid++;
    //    if (not_preload_p)
    //        num_particle++;
    //}
    std::cout << "particle_num: " << m_num_particle << std::endl;
    // free(array_data);
    TransferToCUDA();
}
void MultiphaseFluidSolver::AllocateParticleMemory(int Number)
{
    int phasenum = ( int )m_param[PHASENUMBER];
    int unitsize = phasenum + 3 + 1;
    // Vector3DF objects
    m_position     = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_external_force     = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_velocity     = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_mix_velocity = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_acceleration = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_mforce       = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);
    m_center       = ( Vector3DF* )malloc(sizeof(Vector3DF) * Number);

    m_color = ( float3* )malloc(sizeof(float3) * Number);

    // float objects
    m_position_drawn       = ( float* )malloc(sizeof(float) * unitsize * Number);
    m_position_drawn_rigid = ( float* )malloc(sizeof(float) * unitsize * Number);
    m_density              = ( float* )malloc(sizeof(float) * Number);
    m_pressure             = ( float* )malloc(sizeof(float) * Number);
    m_mix_pressure         = ( float* )malloc(sizeof(float) * Number);
    m_alpha                = ( float* )malloc(sizeof(float) * Number * MAX_PHASE_NUMBER);
    m_alpha_advanced       = ( float* )malloc(sizeof(float) * Number * MAX_PHASE_NUMBER);
    m_restmass             = ( float* )malloc(sizeof(float) * Number);
    m_mrest_density        = ( float* )malloc(sizeof(float) * Number);
    m_eff_v                = ( float* )malloc(sizeof(float) * Number);
    m_smooth_raidus        = ( float* )malloc(sizeof(float) * Number);
    m_particle_radius      = ( float* )malloc(sizeof(float) * Number);
    m_delta_mass           = ( float* )malloc(sizeof(float) * Number);
    m_delta_mass_k         = ( float* )malloc(sizeof(float) * Number * MAX_PHASE_NUMBER);
    m_delta_alpha_k        = ( float* )malloc(sizeof(float) * Number * MAX_PHASE_NUMBER);
    m_rest_mass_k          = ( float* )malloc(sizeof(float) * Number * MAX_PHASE_NUMBER);
    m_surface_scalar_field = ( float* )malloc(sizeof(float) * Number);
    m_concentration        = ( float* )malloc(sizeof(float) * Number);

    // int objects
    m_type      = ( int* )malloc(sizeof(int) * Number);
    m_phase     = ( int* )malloc(sizeof(int) * Number);
    m_explosion = ( int* )malloc(sizeof(int) * Number);
    m_lock      = ( int* )malloc(sizeof(int) * Number);

    // bool objects
    m_active = ( bool* )malloc(sizeof(bool) * Number);
    m_render = ( bool* )malloc(sizeof(bool) * Number);
    m_rotate = ( bool* )malloc(sizeof(bool) * Number);
    m_mix    = ( bool* )malloc(sizeof(bool) * Number);

    // var parameters
    m_max_num_particle = Number;

    // grid parameters
    m_next_particle_index     = ( uint* )malloc(sizeof(uint) * Number);
    m_particle_grid_cellindex = ( uint* )malloc(sizeof(uint) * Number);

    // mc grid
    const int Max_Grid_Number = ( int )m_param[MAXGRIDNUMBER];
    m_mc_scalar_value_field   = ( float* )malloc(sizeof(float) * Max_Grid_Number);
    m_mc_color_field          = ( float* )malloc(3 * sizeof(float) * Max_Grid_Number);
    m_mc_grid_ver_idx         = ( int* )malloc(3 * sizeof(int) * Max_Grid_Number);

    if (m_flag[DISABELTRANSFER])
        memset(m_mix, false, sizeof(bool) * Number);

    // bubble
    m_idx_to_list_idx = ( int* )malloc(sizeof(int) * Number);
    m_bubble_list     = ( int* )malloc(sizeof(int) * MAX_BUBBLE_NUM);
    m_bubble_pos_list = ( float3* )malloc(sizeof(float) * MAX_BUBBLE_NUM * 3);
}
void MultiphaseFluidSolver::InitialiseBoundary()
{
    const float spacingRealWorldSize        = m_param[PSPACINGREALWORLD];
    const float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize * spacingRealWorldSize;
    const float mass                        = 2.0f * m_param[PRESTDENSITY] * particleVolumeRealWorldSize;

    const float     scale              = (1.0f / m_param[SIMSCALE]);
    const Vector3DF maxCorner          = m_vec[BOUNDARYMAX] * scale;
    const Vector3DF minCorner          = m_vec[BOUNDARYMIN] * scale;
    const Vector3DF maxCorner_graphics = m_vec[BOUNDARYMAX];
    const Vector3DF minCorner_graphics = m_vec[BOUNDARYMIN];
    const float     particleSpacing    = m_param[PSPACINGGRAPHICSWORLD];

    const float lengthX = maxCorner_graphics.x - minCorner_graphics.x;
    const float lengthY = maxCorner_graphics.y - minCorner_graphics.y;
    const float lengthZ = maxCorner_graphics.z - minCorner_graphics.z;

    const int numParticlesX = ceil(lengthX / particleSpacing);
    const int numParticlesY = ceil(lengthY / particleSpacing);
    const int numParticlesZ = ceil(lengthZ / particleSpacing);
    const int numParticles  = numParticlesX * numParticlesY * numParticlesZ;

    float tmpX, tmpY, tmpZ;
    if (numParticlesX % 2 == 0)
        tmpX = 0.0;
    else
        tmpX = 0.5;
    if (numParticlesY % 2 == 0)
        tmpY = 0.0;
    else
        tmpY = 0.5;
    if (numParticlesZ % 2 == 0)
        tmpZ = 0.0;
    else
        tmpZ = 0.5;

    int i = m_num_particle;
    for (int iy = 0; iy < numParticlesY; iy++)
    {
        const float y = minCorner.y + (iy /* + tmpY*/) * spacingRealWorldSize;
        for (int ix = 0; ix < numParticlesX; ix++)
        {
            const float x = minCorner.x + (ix /* + tmpX*/) * spacingRealWorldSize;
            for (int iz = 0; iz < numParticlesZ; iz++)
            {
                const float z = minCorner.z + (iz /* + tmpZ*/) * spacingRealWorldSize;
                if (iy == 0 || iy == numParticlesY - 1 || ix == 0 || ix == numParticlesX - 1 || iz == 0 || iz == numParticlesZ - 1)
                {
                    if (m_num_particle < m_max_num_particle)
                    {
                        m_position[i].Set(x, y, z);
                        m_external_force[i].Set(0.0, 0.0, 0.0);
                        m_velocity[i].Set(0.0, 0.0, 0.0);
                        m_mix_velocity[i].Set(0.0, 0.0, 0.0);
                        m_acceleration[i].Set(0.0, 0.0, 0.0);
                        m_density[i]                 = 2000.0f;
                        m_pressure[i]                = 0.0f;
                        m_restmass[i]                = mass;
                        m_mrest_density[i]           = 2000.0f;
                        m_particle_radius[i]         = m_param[PARTICLERADIUS] * scale;
                        m_smooth_raidus[i]           = m_param[SMOOTHRADIUS];
                        m_type[i]                    = BOUND;
                        m_active[i]                  = true;
                        m_particle_grid_cellindex[i] = GRID_UNDEF;
                        m_next_particle_index[i]     = GRID_UNDEF;
                        m_num_particle++;
                        m_num_bound++;
                    }
                    ++i;
                }
            }
        }
    }
    printf("Initialise boundary: %d\n", m_num_bound);
}

void MultiphaseFluidSolver::InitialiseRigid()
{
    int phasenum = ( int )m_param[PHASENUMBER];
    int unitsize = phasenum + 4;

    const float spacingRealWorldSize        = m_param[PSPACINGREALWORLD];
    const float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize * spacingRealWorldSize;
    const float mass                        = 2.0f * m_param[PRESTDENSITY] * particleVolumeRealWorldSize;

    const float     scale              = (1.0f / m_param[SIMSCALE]);
    const Vector3DF maxCorner          = m_vec[RIGIDVOLUMEMAX] * scale;
    const Vector3DF minCorner          = m_vec[RIGIDVOLUMEMIN] * scale;
    const Vector3DF maxCorner_graphics = m_vec[RIGIDVOLUMEMAX];
    const Vector3DF minCorner_graphics = m_vec[RIGIDVOLUMEMIN];
    const float     particleSpacing    = m_param[PSPACINGGRAPHICSWORLD];

    const float lengthX = maxCorner_graphics.x - minCorner_graphics.x;
    const float lengthY = maxCorner_graphics.y - minCorner_graphics.y;
    const float lengthZ = maxCorner_graphics.z - minCorner_graphics.z;

    const int numParticlesX = ceil(lengthX / particleSpacing);
    const int numParticlesY = ceil(lengthY / particleSpacing);
    const int numParticlesZ = ceil(lengthZ / particleSpacing);
    const int numParticles  = numParticlesX * numParticlesY * numParticlesZ;

    float tmpX, tmpY, tmpZ;
    if (numParticlesX % 2 == 0)
        tmpX = 0.0;
    else
        tmpX = 0.5;
    if (numParticlesY % 2 == 0)
        tmpY = 0.0;
    else
        tmpY = 0.5;
    if (numParticlesZ % 2 == 0)
        tmpZ = 0.0;
    else
        tmpZ = 0.5;

    int i = m_num_bound;

    float3 max_pos = make_float3(0.0f, 0.0f, 0.0f);
    float3 min_pos = make_float3(100.0f, 100.0f, 100.0f);

    const int load_num     = LoadPointCloud("data/mesh/Containor.txt", m_position_point_cloud);
    float3    offset       = make_float3(m_vec[RIGIDDRIFT].x / m_param[SIMSCALE], m_vec[RIGIDDRIFT].y / m_param[SIMSCALE], m_vec[RIGIDDRIFT].z / m_param[SIMSCALE]);
    float3    rigid_scale  = make_float3(m_vec[RIGIDSCALE].x, m_vec[RIGIDSCALE].y, m_vec[RIGIDSCALE].z);
    float3    rigid_center = make_float3(m_vec[RIGIDCENTER].x / m_param[SIMSCALE], m_vec[RIGIDCENTER].y / m_param[SIMSCALE], m_vec[RIGIDCENTER].z / m_param[SIMSCALE]);
    for (int index = 0; index < load_num; index++)
    {
        float x = m_position_point_cloud[3 * index + 1] * rigid_scale.x + offset.x;
        float y = m_position_point_cloud[3 * index + 2] * rigid_scale.y + offset.y;
        float z = m_position_point_cloud[3 * index + 0] * rigid_scale.z + offset.z;
        m_position[i].Set(x, y, z);
        m_center[i].Set(rigid_center.x, rigid_center.y, rigid_center.z);
        m_velocity[i].Set(0.0, 0.0, 0.0);
        m_mix_velocity[i].Set(0.0, 0.0, 0.0);
        m_acceleration[i].Set(0.0, 0.0, 0.0);
        m_density[i]         = 2000.0f;
        m_mix_pressure[i]    = 0.0f;
        m_pressure[i]        = 0.0f;
        m_restmass[i]        = mass;
        m_mrest_density[i]   = 1000.0f;
        m_particle_radius[i] = m_param[PARTICLERADIUS] * scale;
        m_smooth_raidus[i]   = m_param[SMOOTHRADIUS];
        // change:
        float preset_alpha[MAX_PHASE_NUMBER] = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
        for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
        {
            m_alpha_advanced[i * MAX_PHASE_NUMBER + fcount] = preset_alpha[fcount];
        }
        m_type[i]                    = RIGID;
        m_active[i]                  = true;
        m_render[i]                  = m_flag[RENDERRIGID];
        m_explosion[i]               = 1;
        m_particle_grid_cellindex[i] = GRID_UNDEF;
        m_next_particle_index[i]     = GRID_UNDEF;
        m_num_particle++;
        m_num_rigid++;
        i++;
        {
            if (max_pos.x < x)
                max_pos.x = x;
            if (max_pos.y < y)
                max_pos.y = y;
            if (max_pos.z < z)
                max_pos.z = z;
            if (min_pos.x > x)
                min_pos.x = x;
            if (min_pos.y > y)
                min_pos.y = y;
            if (min_pos.z > z)
                min_pos.z = z;
        }
    } /**/

    /*
    // load barrier
    float3 offset_rotate = make_float3(_vec[RIGIDDRIFTROTATE].x / _param[SIMSCALE], _vec[RIGIDDRIFTROTATE].y / _param[SIMSCALE], _vec[RIGIDDRIFTROTATE].z / _param[SIMSCALE]);
    float3 rigid_scale_rotate = make_float3(_vec[RIGIDSCALEROTATE].x, _vec[RIGIDSCALEROTATE].y, _vec[RIGIDSCALEROTATE].z);
    float* barrier;
    const int barrier_num = LoadPointCloud("data/mesh/Fan.txt", barrier);
    for (int index = 0; index < barrier_num; index++)
    {
        float x = -barrier[3 * index + 1] * rigid_scale_rotate.x + offset_rotate.x;
        float y = -barrier[3 * index + 2] * rigid_scale_rotate.y + offset_rotate.y;
        float z = barrier[3 * index + 0] * rigid_scale_rotate.z + offset_rotate.z;
        _position[i].Set(x, y, z);
        _center[i].Set(rigid_center.x + offset_rotate.x, rigid_center.y + offset_rotate.y, rigid_center.z + offset_rotate.z);
        _velocity[i].Set(0.0, 0.0, 0.0);
        _mix_velocity[i].Set(0.0, 0.0, 0.0);
        _acceleration[i].Set(0.0, 0.0, 0.0);
        _density[i] = 2000.0f;
        _mix_pressure[i] = 0.0f;
        _pressure[i] = 0.0f;
        _restmass[i] = mass;
        _mrest_density[i] = 1000.0f;
        _particle_radius[i] = _param[PARTICLERADIUS] * scale;
        _smooth_raidus[i] = _param[SMOOTHRADIUS];
        float preset_alpha[MAX_PHASE_NUMBER] = { 0.0f,0.0f,0.0f,0.0f,1.0f };
        for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
        {
            _alpha_advanced[i * MAX_PHASE_NUMBER + fcount] = preset_alpha[fcount];
        }
        _type[i] = RIGID;
        _active[i] = true;
        _render[i] = true;
        _rotate[i] = true;
        particle_grid_cellindex[i] = GRID_UNDEF;
        next_particle_index[i] = GRID_UNDEF;
        num_particle++;
        num_rigid++;
        i++;
    }
    free(barrier);*/

    m_position_drawn_rigid = ( float* )malloc(sizeof(float) * unitsize * m_num_rigid);
    printf("Initialise rigid: %d\nrigid max_x: %f, min_x: %f, max_y: %f, min_y: %f, max_z: %f, min_z: %f\n", m_num_rigid, max_pos.x * m_param[SIMSCALE], min_pos.x * m_param[SIMSCALE], max_pos.y * m_param[SIMSCALE], min_pos.y * m_param[SIMSCALE], max_pos.z * m_param[SIMSCALE], min_pos.z * m_param[SIMSCALE]);
}

void MultiphaseFluidSolver::InitialiseRotateRigid()
{
    int phasenum = ( int )m_param[PHASENUMBER];
    int unitsize = phasenum + 4;

    const float spacingRealWorldSize        = m_param[PSPACINGREALWORLD];
    const float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize * spacingRealWorldSize;
    const float mass                        = 2.0f * m_param[PRESTDENSITY] * particleVolumeRealWorldSize;

    const float     scale              = (1.0f / m_param[SIMSCALE]);
    const Vector3DF maxCorner          = m_vec[RIGIDVOLUMEMAX] * scale;
    const Vector3DF minCorner          = m_vec[RIGIDVOLUMEMIN] * scale;
    const Vector3DF maxCorner_graphics = m_vec[RIGIDVOLUMEMAX];
    const Vector3DF minCorner_graphics = m_vec[RIGIDVOLUMEMIN];
    const float     particleSpacing    = m_param[PSPACINGGRAPHICSWORLD];

    const float lengthX = maxCorner_graphics.x - minCorner_graphics.x;
    const float lengthY = maxCorner_graphics.y - minCorner_graphics.y;
    const float lengthZ = maxCorner_graphics.z - minCorner_graphics.z;

    const int numParticlesX = 6;
    const int numParticlesY = ceil(lengthY / particleSpacing);
    const int numParticlesZ = 2;
    const int numParticles  = numParticlesX * numParticlesY * numParticlesZ;

    float tmpX, tmpY, tmpZ;
    if (numParticlesX % 2 == 0)
        tmpX = 0.0;
    else
        tmpX = 0.5;
    if (numParticlesY % 2 == 0)
        tmpY = 0.0;
    else
        tmpY = 0.5;
    if (numParticlesZ % 2 == 0)
        tmpZ = 0.0;
    else
        tmpZ = 0.5;

    std::cout << std::endl
              << "rigid particle in each direction:	"
              << "X:" << numParticlesX << " "
              << "Y:" << numParticlesY << " "
              << "Z:" << numParticlesZ << std::endl;

    int i                  = m_num_particle;
    m_position_point_cloud = ( float* )malloc(sizeof(float) * 3 * m_param[POINTNUMBERROTATE]);
    LoadPointCloud("source/resources/mesh/finaldata_rotate.txt", m_position_point_cloud);
    float3 offset       = make_float3(m_vec[RIGIDDRIFTROTATE].x / m_param[SIMSCALE], m_vec[RIGIDDRIFTROTATE].y / m_param[SIMSCALE], m_vec[RIGIDDRIFTROTATE].z / m_param[SIMSCALE]);
    float3 rigid_center = make_float3(m_vec[RIGIDCENTER].x / m_param[SIMSCALE], (m_vec[RIGIDCENTER].y + m_vec[RIGIDDRIFT].y) / m_param[SIMSCALE], m_vec[RIGIDCENTER].z / m_param[SIMSCALE]);
    for (int index = 0; index < m_param[POINTNUMBERROTATE]; index++)
    {
        float x = -m_position_point_cloud[3 * index] * m_param[RIGIDSCALEROTATE] + offset.x;
        float y = m_position_point_cloud[3 * index + 2] * m_param[RIGIDSCALEROTATE] + offset.y;
        float z = m_position_point_cloud[3 * index + 1] * m_param[RIGIDSCALEROTATE] + offset.z;
        m_position[i].Set(x, y, z);
        m_center[i].Set(rigid_center.x, rigid_center.y, rigid_center.z);
        m_velocity[i].Set(0.0, 0.0, 0.0);
        m_mix_velocity[i].Set(0.0, 0.0, 0.0);
        m_acceleration[i].Set(0.0, 0.0, 0.0);
        m_density[i]                         = 2000.0f;
        m_mix_pressure[i]                    = 0.0f;
        m_pressure[i]                        = 0.0f;
        m_restmass[i]                        = mass;
        m_mrest_density[i]                   = 1000.0f;
        m_particle_radius[i]                 = m_param[PARTICLERADIUS] * scale;
        m_smooth_raidus[i]                   = m_param[SMOOTHRADIUS];
        float preset_alpha[MAX_PHASE_NUMBER] = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
        for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
        {
            m_alpha_advanced[i * MAX_PHASE_NUMBER + fcount] = preset_alpha[fcount];
        }
        m_type[i]                    = RIGID;
        m_rotate[i]                  = true;
        m_render[i]                  = true;
        m_active[i]                  = true;
        m_particle_grid_cellindex[i] = GRID_UNDEF;
        m_next_particle_index[i]     = GRID_UNDEF;
        m_num_particle++;
        m_num_rigid++;
        i++;
    }
    //_position_drawn_rigid = (float*)malloc(sizeof(float) * unitsize * num_rigid);
    // printf("num_rigid: %d\n", num_rigid);
}

void MultiphaseFluidSolver::InitialisePositionPhase()
{
    memset(m_render, true, sizeof(bool) * m_max_num_particle);
    m_num_particle -= m_num_fluid;
    m_num_fluid = 0;

    const float spacingRealWorldSize        = m_param[PSPACINGREALWORLD];
    const float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize * spacingRealWorldSize;
    const float mass                        = m_param[PRESTDENSITY] * particleVolumeRealWorldSize;

    const float     scale                    = (1.0f / m_param[SIMSCALE]);
    const Vector3DF maxCorner                = m_vec[INITIALVOLUMEMAX] * scale;
    const Vector3DF minCorner                = m_vec[INITIALVOLUMEMIN] * scale;
    const Vector3DF maxCorner_graphics       = m_vec[INITIALVOLUMEMAX];
    const Vector3DF minCorner_graphics       = m_vec[INITIALVOLUMEMIN];
    const Vector3DF maxCornerPhase           = m_vec[INITIALVOLUMEMAXPHASE] * scale;
    const Vector3DF minCornerPhase           = m_vec[INITIALVOLUMEMINPHASE] * scale;
    const Vector3DF maxCornerPhase_graphics  = m_vec[INITIALVOLUMEMAXPHASE];
    const Vector3DF minCornerPhase_graphics  = m_vec[INITIALVOLUMEMINPHASE];
    const Vector3DF maxCornerPhase2          = m_vec[INITIALVOLUMEMAXPHASES] * scale;
    const Vector3DF minCornerPhase2          = m_vec[INITIALVOLUMEMINPHASES] * scale;
    const Vector3DF maxCornerPhase2_graphics = m_vec[INITIALVOLUMEMAXPHASES];
    const Vector3DF minCornerPhase2_graphics = m_vec[INITIALVOLUMEMINPHASES];
    const float     particleSpacing          = m_param[PSPACINGGRAPHICSWORLD];

    // phase 1:
    const float lengthX = maxCorner_graphics.x - minCorner_graphics.x;
    const float lengthY = maxCorner_graphics.y - minCorner_graphics.y;
    const float lengthZ = maxCorner_graphics.z - minCorner_graphics.z;

    const int numParticlesX = ceil(lengthX / particleSpacing);
    const int numParticlesY = ceil(lengthY / particleSpacing);
    const int numParticlesZ = ceil(lengthZ / particleSpacing);
    const int numParticles  = numParticlesX * numParticlesY * numParticlesZ;

    // phase 2:
    const float lengthXPhase = maxCornerPhase_graphics.x - minCornerPhase_graphics.x;
    const float lengthYPhase = maxCornerPhase_graphics.y - minCornerPhase_graphics.y;
    const float lengthZPhase = maxCornerPhase_graphics.z - minCornerPhase_graphics.z;

    const int numParticlesXPhase = ceil(lengthXPhase / particleSpacing);
    const int numParticlesYPhase = ceil(lengthYPhase / particleSpacing);
    const int numParticlesZPhase = ceil(lengthZPhase / particleSpacing);
    const int numParticlesPhase  = numParticlesXPhase * numParticlesYPhase * numParticlesZPhase;

    // phase 3:
    const float lengthXPhase2 = maxCornerPhase2_graphics.x - minCornerPhase2_graphics.x;
    const float lengthYPhase2 = maxCornerPhase2_graphics.y - minCornerPhase2_graphics.y;
    const float lengthZPhase2 = maxCornerPhase2_graphics.z - minCornerPhase2_graphics.z;

    const int numParticlesXPhase2 = ceil(lengthXPhase2 / particleSpacing);
    const int numParticlesYPhase2 = ceil(lengthYPhase2 / particleSpacing);
    const int numParticlesZPhase2 = ceil(lengthZPhase2 / particleSpacing);
    const int numParticlesPhase2  = numParticlesXPhase2 * numParticlesYPhase2 * numParticlesZPhase2;

    float tmpX, tmpY, tmpZ;
    if (numParticlesX % 2 == 0)
        tmpX = 0.0;
    else
        tmpX = 0.5;
    if (numParticlesY % 2 == 0)
        tmpY = 0.0;
    else
        tmpY = 0.5;
    if (numParticlesZ % 2 == 0)
        tmpZ = 0.0;
    else
        tmpZ = 0.5;

    std::cout << std::endl
              << "Initialise particle:" << std::endl;
    std::cout << std::endl
              << "particle in each direction:	"
              << "X:" << numParticlesX << " "
              << "Y:" << numParticlesY << " "
              << "Z:" << numParticlesZ << std::endl;
    std::cout << std::endl
              << "phase particle in each direction:	"
              << "X:" << numParticlesXPhase << " "
              << "Y:" << numParticlesYPhase << " "
              << "Z:" << numParticlesZPhase << std::endl;

    int   i = m_num_particle;
    float x_s, y_s, z_s;
    float cos120 = -0.5f, sin120 = 0.866025f;
    float cos240 = -0.5f, sin240 = -0.866025f;
    float cos_s, sin_s;
    float dist;
}

void MultiphaseFluidSolver::InitialisePreLoadParticles()
{
    const int current_num = m_num_bound + m_num_rigid;
    m_num_particle        = m_param[MAXLOADNUMBER] + current_num;
    m_num_fluid           = 0;
    printf("load num: %d\n", m_num_particle);
    memset(m_active + current_num, false, sizeof(bool) * (m_max_num_particle - current_num));
    memset(m_render + current_num, false, sizeof(bool) * (m_max_num_particle - current_num));
    memset(m_rotate + current_num, false, sizeof(bool) * (m_max_num_particle - current_num));

    const float spacingRealWorldSize        = m_param[PSPACINGREALWORLD];
    const float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize * spacingRealWorldSize;
    const float mass                        = m_param[PRESTDENSITY] * particleVolumeRealWorldSize;

    const float     scale                    = (1.0f / m_param[SIMSCALE]);
    const Vector3DF maxCorner                = m_vec[INITIALVOLUMEMAX] * scale;
    const Vector3DF minCorner                = m_vec[INITIALVOLUMEMIN] * scale;
    const Vector3DF maxCorner_graphics       = m_vec[INITIALVOLUMEMAX];
    const Vector3DF minCorner_graphics       = m_vec[INITIALVOLUMEMIN];
    const Vector3DF maxCornerPhase           = m_vec[INITIALVOLUMEMAXPHASE] * scale;
    const Vector3DF minCornerPhase           = m_vec[INITIALVOLUMEMINPHASE] * scale;
    const Vector3DF maxCornerPhase_graphics  = m_vec[INITIALVOLUMEMAXPHASE];
    const Vector3DF minCornerPhase_graphics  = m_vec[INITIALVOLUMEMINPHASE];
    const Vector3DF maxCornerPhase2          = m_vec[INITIALVOLUMEMAXPHASES] * scale;
    const Vector3DF minCornerPhase2          = m_vec[INITIALVOLUMEMINPHASES] * scale;
    const Vector3DF maxCornerPhase2_graphics = m_vec[INITIALVOLUMEMAXPHASES];
    const Vector3DF minCornerPhase2_graphics = m_vec[INITIALVOLUMEMINPHASES];
    const float     particleSpacing          = m_param[PSPACINGGRAPHICSWORLD];

    // phase 1:
    const float lengthX = maxCorner_graphics.x - minCorner_graphics.x;
    const float lengthY = maxCorner_graphics.y - minCorner_graphics.y;
    const float lengthZ = maxCorner_graphics.z - minCorner_graphics.z;

    const int numParticlesX = ceil(lengthX / particleSpacing);
    const int numParticlesY = ceil(lengthY / particleSpacing);
    const int numParticlesZ = ceil(lengthZ / particleSpacing);
    const int numParticles  = numParticlesX * numParticlesY * numParticlesZ;

    // phase 2:
    const float lengthXPhase = maxCornerPhase_graphics.x - minCornerPhase_graphics.x;
    const float lengthYPhase = maxCornerPhase_graphics.y - minCornerPhase_graphics.y;
    const float lengthZPhase = maxCornerPhase_graphics.z - minCornerPhase_graphics.z;

    const int numParticlesXPhase = ceil(lengthXPhase / particleSpacing);
    const int numParticlesYPhase = ceil(lengthYPhase / particleSpacing);
    const int numParticlesZPhase = ceil(lengthZPhase / particleSpacing);
    const int numParticlesPhase  = numParticlesXPhase * numParticlesYPhase * numParticlesZPhase;

    // phase 3:
    const float lengthXPhase2 = maxCornerPhase2_graphics.x - minCornerPhase2_graphics.x;
    const float lengthYPhase2 = maxCornerPhase2_graphics.y - minCornerPhase2_graphics.y;
    const float lengthZPhase2 = maxCornerPhase2_graphics.z - minCornerPhase2_graphics.z;

    const int numParticlesXPhase2 = ceil(lengthXPhase2 / particleSpacing);
    const int numParticlesYPhase2 = ceil(lengthYPhase2 / particleSpacing);
    const int numParticlesZPhase2 = ceil(lengthZPhase2 / particleSpacing);
    const int numParticlesPhase2  = numParticlesXPhase2 * numParticlesYPhase2 * numParticlesZPhase2;

    float tmpX, tmpY, tmpZ;
    if (numParticlesX % 2 == 0)
        tmpX = 0.0;
    else
        tmpX = 0.5;
    if (numParticlesY % 2 == 0)
        tmpY = 0.0;
    else
        tmpY = 0.5;
    if (numParticlesZ % 2 == 0)
        tmpZ = 0.0;
    else
        tmpZ = 0.5;

    // std::cout << endl << "particle in each direction:	" << "X:" << numParticlesX << " " << "Y:" << numParticlesY << " " << "Z:" << numParticlesZ << endl;
    // std::cout << endl << "phase particle in each direction:	" << "X:" << numParticlesXPhase << " " << "Y:" << numParticlesYPhase << " " << "Z:" << numParticlesZPhase << endl;

    int i = m_num_fluid + m_num_bound + m_num_rigid;
}
void MultiphaseFluidSolver::TransferToCUDA()
{
    TransferDataToCUDA(( float* )m_position, ( float* )m_velocity, ( float* )m_acceleration, m_pressure, m_density, m_type, m_explosion, m_lock, m_active, m_render, m_rotate, ( float* )m_center, m_smooth_raidus, m_mc_grid_ver_idx, m_concentration, m_mix,(float*)m_external_force);
    if (m_flag[MUTIPHASE])
        TransferMFDataToCUDA(( float* )m_mforce, m_restmass, m_mrest_density, m_mix_pressure, m_alpha_advanced, m_phase, ( float* )m_mix_velocity, m_rest_mass_k);
}

void MultiphaseFluidSolver::TransferFromCUDA()
{
    TransferDataFromCUDA(( float* )m_position, ( float* )m_velocity, ( float* )m_acceleration, m_pressure, m_density, m_type, m_render, m_particle_radius, m_concentration, m_surface_scalar_field, ( float* )m_color, m_mc_scalar_value_field, m_mc_color_field);
    if (m_flag[MUTIPHASE])
        TransferMFDataFromCUDA(m_alpha_advanced, m_restmass, m_eff_v, m_delta_mass, m_delta_mass_k, m_delta_alpha_k, m_density, ( float* )m_mix_velocity);
    auto fluid_comp  = m_fluid->getComponent<MultiphaseFluidComponent>();
    auto positions   = fluid_comp->m_pos;
    fluid_comp->pnum = m_num_fluid;
    int iter         = 0;
    for (int i = m_num_bound; i < m_num_bound + m_num_fluid; i++)
    {
        positions[iter * 3 + 0] = m_position[i].x;
        positions[iter * 3 + 1] = m_position[i].y;
        positions[iter * 3 + 2] = m_position[i].z;
        iter++;
    }
    std::cout << "when transfer: " << positions[(123547 - m_num_bound) * 3] << std::endl;
    std::cout << "when transfer cuda: " << m_position[123547].x << std::endl;
    // if (_flag[BUBBLESIMULATION])
    //	TransferBubbleDataFromCUDA((float*)_position, _particle_radius, _type);
}
// void MultiphaseFluidSolver::CopyToComponent() {
//     auto particle_component=m_fluid->getComponent<FluidParticleComponent>();
//     //ProjectData(particle_component->m_device_pos, particle_component->m_color);
// }
//  load external model PointCloud data
int MultiphaseFluidSolver::LoadPointCloud(const char* pointdata, float*& load_array)
{
    std::ifstream readfile;

    readfile.open(pointdata, std::ios::in);
    if (!readfile.is_open())
    {
        std::cerr << "fail to load" << std::endl;
        ;
    }
    char buf[1024] = { 0 };
    readfile.getline(buf, sizeof(buf));

    int         count    = 0;
    std::string load_num = "";
    while (*(buf + count) != '\0')
    {
        if (*(buf + count) <= '9' && *(buf + count) >= '0')
            load_num += *(buf + count);
        count++;
    }
    const int num = std::atoi(load_num.c_str());
    load_array    = ( float* )malloc(num * sizeof(float) * 3);

    int i = 0;
    while (readfile >> buf)
    {
        // string Str = buf;
        float fp;
        sscanf_s(buf, "%f", &fp);
        load_array[i] = fp / m_param[SIMSCALE];
        i += 1;
    }
    return num;
}
void MultiphaseFluidSolver::SetKernels()
{
    m_param[PKERNELSELF] = kernelM4(0.0f, m_param[SMOOTHRADIUS]);

    float sr = m_param[SMOOTHRADIUS];
    for (int i = 0; i < m_lut_size; i++)
    {
        float dist         = sr * i / m_lut_size;
        m_lut_kernel_m4[i] = kernelM4(dist, sr);
    }
}

void MultiphaseFluidSolver::SetTimeStep()
{
    float maxParticleSpeed = 4.0f;
    float courantFactor    = 0.35f;
    if (m_flag[IMPLICIT])
        courantFactor = 0.6f;

    //_param[PGASCONSTANT] = 6000.0f;
    float speedOfSound  = sqrt(m_param[PGASCONSTANT]);
    float relevantSpeed = (speedOfSound > maxParticleSpeed) ? speedOfSound : maxParticleSpeed;
    m_time_step         = courantFactor * m_param[SMOOTHRADIUS] / relevantSpeed;
    //_param[TIMESTEP] = _time_step;
    printf("timestep: %f\n", m_time_step);
}
void MultiphaseFluidSolver::RefineGridResolution()
{
    Vector3DF Volume            = m_vec[BOUNDARYMAX] - m_vec[BOUNDARYMIN] + m_vec[GRIDBOUNDARYOFFSET] + m_vec[GRIDBOUNDARYOFFSET];
    m_vec_int[GRIDRESOLUTION].x = ( int )ceil(Volume.x / (m_param[GRIDRADIUS] * m_param[SIMSCALE]));
    m_vec_int[GRIDRESOLUTION].y = ( int )ceil(Volume.y / (m_param[GRIDRADIUS] * m_param[SIMSCALE]));
    m_vec_int[GRIDRESOLUTION].z = ( int )ceil(Volume.z / (m_param[GRIDRADIUS] * m_param[SIMSCALE]));
    m_grid_number               = m_vec_int[GRIDRESOLUTION].x * m_vec_int[GRIDRESOLUTION].y * m_vec_int[GRIDRESOLUTION].z;
    if (m_grid_number > m_param[MAXGRIDNUMBER])
    {
        std::cout << std::endl
                  << "---------------------------------------" << std::endl;
        std::cout << "Grid number:" << m_grid_number << " "
                  << "Too many grids!" << std::endl;
        std::cout << "Grid Resolution:" << m_vec_int[GRIDRESOLUTION].x << " " << m_vec_int[GRIDRESOLUTION].y << " " << m_vec_int[GRIDRESOLUTION].z << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        exit(-1);
    }
    m_grid_particle_table = ( uint* )malloc(sizeof(uint) * m_grid_number);
    m_num_particle_grid   = ( uint* )malloc(sizeof(uint) * m_grid_number);
    std::cout << "Grid number:" << m_grid_number << std::endl;

    m_grid_search_offset = ( int* )malloc(sizeof(int) * 27);
    int cell             = 0;
    for (int y = -1; y < 2; y++)
        for (int z = -1; z < 2; z++)
            for (int x = -1; x < 2; x++)
                m_grid_search_offset[cell++] = x + m_vec_int[GRIDRESOLUTION].x * (y + m_vec_int[GRIDRESOLUTION].y * z);
    // change:
    /**/
    // mc grid initializes
    m_mc_grid_radius       = m_param[SMOOTHRADIUS] * m_param[MCUBEVOXEL] * m_param[SIMSCALE];
    m_mc_grid_resolution.x = 1;
    m_mc_grid_resolution.y = 1;
    m_mc_grid_resolution.z = 1;
    m_mc_grid_min          = make_float3(m_vec[BOUNDARYMIN].x - m_vec[GRIDBOUNDARYOFFSET].x, m_vec[BOUNDARYMIN].y - m_vec[GRIDBOUNDARYOFFSET].y, m_vec[BOUNDARYMIN].z - m_vec[GRIDBOUNDARYOFFSET].z);
    m_mc_grid_max          = make_float3(m_vec[BOUNDARYMAX].x + m_vec[GRIDBOUNDARYOFFSET].x, m_vec[BOUNDARYMAX].y + m_vec[GRIDBOUNDARYOFFSET].y, m_vec[BOUNDARYMAX].z + m_vec[GRIDBOUNDARYOFFSET].z);
    m_mc_grid_number       = m_mc_grid_resolution.x * m_mc_grid_resolution.y * m_mc_grid_resolution.z;
    m_mc_grid_ver_number   = (m_mc_grid_resolution.x + 1) * (m_mc_grid_resolution.y + 1) * (m_mc_grid_resolution.z + 1);

    if (m_mc_grid_number > m_param[MAXGRIDNUMBER])
    {
        std::cout << std::endl
                  << "---------------------------------------" << std::endl;
        std::cout << "McGrid number:" << m_mc_grid_number << " "
                  << "Too many grids!" << std::endl;
        std::cout << "McGrid Resolution:" << m_mc_grid_resolution.x << " " << m_mc_grid_resolution.y << " " << m_mc_grid_resolution.z << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        exit(-1);
    }

    for (int i = 0; i < m_mc_grid_ver_number; i++)
        m_mc_grid_ver_idx[i] = i;
    std::cout << "MC Grid number:" << m_mc_grid_number << std::endl;
}
void MultiphaseFluidSolver::AllocateGrid()
{
    memset(m_grid_particle_table, GRID_UNDEF, sizeof(uint) * m_grid_number);
    memset(m_num_particle_grid, 0, sizeof(uint) * m_grid_number);

    const int grid_res_x = m_vec_int[GRIDRESOLUTION].x;
    const int grid_res_y = m_vec_int[GRIDRESOLUTION].y;
    const int grid_res_z = m_vec_int[GRIDRESOLUTION].z;

    for (int i = 0; i < m_num_particle; i++)
    {
        Vector3DF pos       = m_position[i];
        Vector3DI Cell      = GetCell(pos);
        const int CellIndex = Cell.x + m_vec_int[GRIDRESOLUTION].x * (Cell.y + m_vec_int[GRIDRESOLUTION].y * Cell.z);
        if (Cell.x >= 0 && Cell.x < grid_res_x && Cell.y >= 0 && Cell.y < grid_res_y && Cell.z >= 0 && Cell.z < grid_res_z)
        {
            m_particle_grid_cellindex[i]     = CellIndex;
            m_next_particle_index[i]         = m_grid_particle_table[CellIndex];
            m_grid_particle_table[CellIndex] = i;
            m_num_particle_grid[CellIndex] += 1;
        }
    }
}
void MultiphaseFluidSolver::EnableHighResolution()
{
    m_param[PSPACINGREALWORLD] *= 0.5f;
    m_param[PSPACINGGRAPHICSWORLD] *= 0.5f;
    m_param[PARTICLERADIUS] *= 0.5f;
    m_param[SMOOTHRADIUS] *= 0.5f;
    m_param[GRIDRADIUS] *= 0.5f;
    m_param[SURFACETENSIONFACTOR] *= 0.125f;
    m_param[PVISC] *= 0.5f;
    for (int fcount = 0; fcount < MAX_PHASE_NUMBER; fcount++)
    {
        m_multi_phase[MFRESTMASS * MAX_PHASE_NUMBER + fcount] *= 0.125f;
        m_multi_phase[MFVISCOSITY * MAX_PHASE_NUMBER + fcount] *= 0.25f;
    }
}
float MultiphaseFluidSolver::kernelM4(float dist, float sr)
{
    float s = dist / sr;
    float result;
    float factor = 2.546479089470325472f / (sr * sr * sr);
    if (dist < 0.0f || dist >= sr)
        return 0.0f;
    else
    {
        if (s < 0.5f)
        {
            result = 1.0f - 6.0 * s * s + 6.0f * s * s * s;
        }
        else
        {
            float tmp = 1.0f - s;
            result    = 2.0 * tmp * tmp * tmp;
        }
    }
    return factor * result;
}
Vector3DI MultiphaseFluidSolver::GetCell(const Vector3DF& position)
{
    Vector3DI Cell;

    const float pos_x = position.x - (m_vec[BOUNDARYMIN].x + m_vec[GRIDBOUNDARYOFFSET].x) / m_param[SIMSCALE];
    const float pos_y = position.y - (m_vec[BOUNDARYMIN].y + m_vec[GRIDBOUNDARYOFFSET].y) / m_param[SIMSCALE];
    const float pos_z = position.z - (m_vec[BOUNDARYMIN].z + m_vec[GRIDBOUNDARYOFFSET].z) / m_param[SIMSCALE];

    float r = m_param[GRIDRADIUS];
    Cell.x  = ( int )ceil(pos_x / r);
    Cell.y  = ( int )ceil(pos_y / r);
    Cell.z  = ( int )ceil(pos_z / r);
    return Cell;
}

inline void MultiphaseFluidSolver::tick(int label)
{
    m_last_time_point[label] = Clock::now();
}
inline void MultiphaseFluidSolver::tock(const std::string& info, int label)
{
    std::cout << "[" << info << "] "
              << (1000 / (static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_last_time_point[label]).count()) * 1e-6)) << " FPS"
              << std::endl;
}
}  // namespace Physika