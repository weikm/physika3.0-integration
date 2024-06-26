/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-10-18
 * @description: declaration of DummyCudaSolver class, a sample solver that demonstrates
 *               the use of cuda in solver developement
 * @version    : 1.0
 */

#pragma once

#include <vector>
#ifdef BUILD_WITH_CUDA
#include <thrust/device_vector.h>
#endif

#include "framework/solver.hpp"

/**
 * Known issue: nvcc can't compile entt code
 * Solution:
 * Place CUDA code and CPU code in separate files to enforce that framework/world.hpp and
 * framework/object.hpp are not included in .cu files.
 *
 * dummy_cuda_solver.cpp, dummy_cuda_solver.cu, cuda_kernels.cu serve as a sample.
 *
 */

namespace Physika {

struct ThrustComponent
{
    std::vector<double> m_host_pos;
    std::vector<double> m_host_vel;
#ifdef BUILD_WITH_CUDA
    thrust::device_vector<double> m_device_pos;
    thrust::device_vector<double> m_device_vel;
#endif

    /**
     * PLEASE DO READ THIS!!!
     *
     * Provide custom defined constructor, destructor, copy constructor, assign operator so that
     * compilers won't provide default ones.
     *
     * Compiler provided ones will lead to build/run-time errors that are not trivial to find out the reason.
     * For instance:
     * g++/msvc will generate default constructor for ThrustComponent while compiling dummy_cuda_solver.cpp,
     * undefined symbols error occurs if BUILD_WITH_CUDA is on, because g++/msvc cannot handle device code
     * of thrust.
     * To resolve issue, provide host only definitions in dummy_cuda_solver.cpp for the case when BUILD_WITH_CUDA is off,
     * but provide host&&device definitions in dummy_cuda_solver.cu for the case when BUILD_WITH_CUDA is on.
     *
     */
    ThrustComponent();
    ~ThrustComponent();
    ThrustComponent(const ThrustComponent&);
    ThrustComponent& operator=(const ThrustComponent&);
    bool             operator==(const ThrustComponent& comp) const;

    void reset();

    void resize(size_t size, double value);

#ifdef BUILD_WITH_CUDA
    void copyHostToDevice();
    void copyDeviceToHost();
#endif

    bool checkDataSizeConsistency();
};

/**
 * @brief DummyCudaSolver simulates 1d free fall due to gravity
 *
 */
class DummyCudaSolver : public Solver
{
public:
    struct SolverConfig
    {
        double m_dt;
        double m_total_time;
        double m_gravity;
        bool   m_use_gpu;
    };

    DummyCudaSolver();
    ~DummyCudaSolver();

    bool initialize() override;

    bool isInitialized() const override;

    bool reset() override;

    bool step() override;

    bool run() override;

    bool isApplicable(const Object* object) const override;

    bool attachObject(Object* object) override;

    bool detachObject(Object* object) override;

    void clearAttachment() override;

    void config(const SolverConfig& config);

private:
    bool stepCPU(double dt);
    bool stepGPU(double dt);

#ifdef BUILD_WITH_CUDA
    void stepKernelWrapper(int blocks_per_grid, int threads_per_block, double dt, double gravity, ThrustComponent* comp);
#endif

private:
    bool         m_is_init;
    SolverConfig m_config;
    Object*      m_object;
    double       m_cur_time;
};

}  // namespace Physika