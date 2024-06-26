/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: declaration of SingleObjSolver class, a sample solver that demonstrates
 *               how to define a solver that applies to single object
 * @version    : 1.0
 */

#pragma once

#include "framework/solver.hpp"

namespace Physika {

/**
 * a dummy component to demonstrate how solvers use component type(s) to determine which object
 * is applicable.
 */
struct DummyComponent
{
    void reset() {}  // physika requires every component type to have a reset() method
    // data entries here
    double m_value;
};

/**
 * SingleObjSolver is a sample of typical solver whose subject is a single object
 * The solver is a dummy dynamic simulation solver, it is applicable to objects that
 * have the DummyComponent component and the solver does nothing but print stuff.
 *
 */
class SingleObjSolver : public Solver
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

public:
    SingleObjSolver();
    ~SingleObjSolver();

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
    Object*      m_object;
    double       m_cur_time;
};

}  // namespace Physika