/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: declaration of DummySFISolver class, a sample solver that demonstrates
 *               how to define a solver that needs objects of different kind to run
 * @version    : 1.0
 */

#pragma once

#include "framework/solver.hpp"

namespace Physika {

/**
 * Dummy components to identify if an object is solid or fluid
 */
struct SolidComponent
{
    void reset() {}  // physika requires every component type to have a reset() method
    // data entries here
    double m_pos;
};

struct FluidComponent
{
    void reset() {}  // physika requires every component type to have a reset() method
    // data entries here
    double m_pressure;
};

class DummySFISolver : public Solver
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

    DummySFISolver();
    ~DummySFISolver();

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
     * @brief a sample method to demonstrate how configuration can be set to solvers
     *        The configuration will be used in step()/run() methods
     */
    void config(const SolverConfig& config);

private:
    bool         m_is_init;
    SolverConfig m_config;
    Object*      m_fluid;
    Object*      m_solid;
    double       m_cur_time;
};

}  // namespace Physika