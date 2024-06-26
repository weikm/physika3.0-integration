/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-09
 * @description: declaration of Solver class, abstract base class of all solvers
 * @version    : 1.0
 */

#pragma once

#include <cstdint>

namespace Physika {

class Object;

/**
 * Solver is the base class of all solvers, and provides common APIs.
 *
 * Unlike Object, Solver can be added to multiple scenes.
 * Note that Solver is not aware of the existence of Scenes, it's the developers' obligation
 * to make sure things don't get messed up across scenes if a solver is concurrently attached to objects
 * in multiple scenes.
 */
class Solver
{
public:
    Solver()          = default;
    virtual ~Solver() = default;

    /**
     * @brief    get the unique id of the solver, which won't change once the solver is created
     *
     * @return   the id of the solver
     */
    std::uint64_t id() const;

    /**
     * @brief initialize the solver to get it ready for execution.
     *        The behavior of duplicate calls is up to the developers of subclasses, yet it is
     *        recommended that duplicate calls should be ignored to avoid redundant computation.
     *
     * @return  true if initialization succeeds, otherwise return false
     *
     */
    virtual bool initialize() = 0;

    /**
     * @brief get the initialization state of the solver.
     *
     * @return   true if solver has been properly initialized, otherwise return false
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief reset the solver to newly initialized state
     *
     * @return    true if reset succeeds, otherwise return false
     */
    virtual bool reset() = 0;

    /**
     * @brief step the solver ahead through a prescribed time step. The step size is set via other methods
     *        declared by subclasses, e.g., setTimeStep(double dt).
     *        The method is designed for time-dependent solvers, typically used in an execution loop.
     *        For time-independent solvers, this method should behave identical to the run() method.
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    virtual bool step() = 0;

    /**
     * @brief run the solver till termination condition is met. Termination conditions are set via other methods
     *        declared by subclasses, e.g., a time range, a tolerance, etc.
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    virtual bool run() = 0;

    /**
     * @brief check whether the solver is applicable to given object
     *        If the object contains all Components that the solver requires, it is defined as applicable.
     *        Note that the solver is applicable to the object doesn't mean that it will be applied to the
     *        object. It's the developer's job to bind the object to the solver via the solver's APIs.
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    virtual bool isApplicable(const Object* object) const = 0;

    /**
     * @brief attach an object to the solver
     *        The solver will be applied to the object (if applicable) when step()/run() is called
     *        Multiple objects can be attached to the solver via multiple function call
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully attached
     *            false if error occurs, e.g., object is nullptr, object has been attached before, etc.
     */
    virtual bool attachObject(Object* object) = 0;

    /**
     * @brief detach an object to the solver
     *        The solver will not be applied to the object when step()/run() is called
     *        Multiple objects can be dettached to the solver via multiple function call
     *
     * @param[in] object pointer to the object
     *
     * @return    true if the object is successfully detached
     *            false if error occurs, e.g., object is nullptr, object has not been attached before, etc.
     */
    virtual bool detachObject(Object* object) = 0;

    /**
     * @brief clear all object attachments
     */
    virtual void clearAttachment() = 0;
};

}  // namespace Physika