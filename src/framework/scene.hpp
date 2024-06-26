/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-09
 * @description: declaration of Scene class, which is basically the container of all simulation subjects
 *               in one setup
 * @version    : 1.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace Physika {

class Object;
class Solver;

/**
 * Scene is a self-contained unit for simulation.
 * Objects and Solvers are created and assigned to Scene by World.
 * Scene does NOT take ownership of the objects/solvers assigned to it.
 * Objects are automatically attached to Solvers if applicable.
 *
 * Scene accepts dangling solvers (solvers not created by world), but it is not preferred.
 *
 * Scene provides 2 views of its content:
 * 1. Object-oriented view: it's all about the data of the objects.
 * 2. Computation-oriented view: it's all about the computation graph of the solvers.
 */
class Scene final
{
public:
    ~Scene();

    std::uint64_t id() const;

    /**
     * @brief reset the state of all contents in the scene
     *        The objects and solvers are reset as well, yet not removed
     *
     * @return    true if scene is successfully reset, otherwise return false
     */
    bool reset();

    //////////////////// Object Management ////////////////////

    /**
     * @brief add an object into the scene
     *        an object already in scene will not be added redundantly
     *
     * @param[in] object    pointer to the object to be added
     *
     * @return    true if the object is successfully added
     *            false if error occurs, e.g., the pointer is nullptr, the object is in another scene, etc.
     */
    bool addObject(Object* object);

    /**
     * @brief add an object with given id into the scene
     *        an object already in scene will not be added redundantly
     *
     * @param[in] id    id of the object to be added
     *
     * @return     true if the object is successfully added
     *             false if error occurs, e.g., object with given id not found in world, the object is in another scene, etc.
     */
    bool addObjectById(std::uint64_t id);

    /**
     * @brief remove an object from the scene
     *
     * @param[in] object    pointer to the object to be removed
     *
     * @return    true if the object is successfully removed
     *            false if error occurs, e.g., the pointer is nullptr, the object is not in the scene, etc.
     */
    bool removeObject(Object* object);

    /**
     * @brief remove an object with given id from the scene
     *
     * @param[in] id    id of the object to be removed
     *
     * @return    true if the object is successfully removed
     *            false if error occurs, e.g., object with given id not found in scene, etc.
     */
    bool removeObjectById(std::uint64_t id);

    /**
     * @brief remove all objects from the scene
     *
     */
    void removeAllObjects();

    /**
     * @brief get the number of objects in scene
     *
     * @return    the number of objects in scene
     */
    std::uint64_t objectNum() const;

    /**
     * @brief find the object with given id in the scene
     *
     * @param[in] obj_id    id of the object to fetch
     *
     * @return    pointer to the object with given id, return nullptr if not found in scene
     */
    const Object* getObjectById(std::uint64_t obj_id) const;
    Object*       getObjectById(std::uint64_t obj_id);

    //////////////////// Solver Management ////////////////////

    /**
     * @brief add a solver into the scene
     *        dangling solver can be added as well
     *        a solver already in scene will not be added redundantly
     *
     * @param[in] solver    pointer to the solver to be added
     *
     * @return    true if the solver is successfully added
     *            false if error occurs, e.g., the pointer is nullptr, etc.
     */
    bool addSolver(Solver* solver);

    /**
     * @brief add a solver with given id into the scene
     *        a solver already in scene will not be added redundantly
     *
     * @param[in] id    id of the solver to be added
     *
     * @return    true if the solver is successfully added
     *            false if error occurs, e.g., the solver with given id is not found in the world, etc.
     */
    bool addSolverById(std::uint64_t id);

    /**
     * @brief remove a solver from the scene
     *
     * @param[in] solver    pointer to the solver to be removed
     *
     * @return    true if the solver is successfully removed
     *            false if error occurs, e.g., the pointer is nullptr, the solver is not found in scene, etc.
     */
    bool removeSolver(Solver* solver);

    /**
     * @brief remove a solver with given id from the scene
     *
     * @param[in] id    id of the solver to be removed
     *
     * @return    true if the solver is successfully removed
     *            false if error occurs, e.g., the solver is not found in the scene, etc.
     */
    bool removeSolverById(std::uint64_t id);

    /**
     * @brief remove all solvers in the scene
     */
    void removeAllSolvers();

    /**
     * @brief get the total number of solvers in the scene
     *
     * @return    the number of solvers in the scene
     */
    std::uint64_t solverNum() const;

    /**
     * @brief get the solver with given id
     *
     * @param[in] solver_id    id of the solver to be found
     *
     * @return    pointer of the solver with given id, return nullptr if not found in the scene
     */
    const Solver* getSolverById(std::uint64_t solver_id) const;
    Solver*       getSolverById(std::uint64_t solver_id);

    //////////////////// Update Methods ////////////////////

    /**
     * @brief run the step() methods of all solvers in the scene
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool step();

    /**
     * @brief run the run() methods of all solvers in the scene
     *
     * @return    true if procedure successfully completes, otherwise return false
     */
    bool run();

private:
    Scene();  // make constructor private to force creation through World API

    // disable copy&&move
    Scene(const Scene&)            = delete;
    Scene(Scene&&)                 = delete;
    Scene& operator=(const Scene&) = delete;
    Scene& operator=(Scene&&)      = delete;

    void setId(std::uint64_t id);

    bool updateSolverAttachInfo();  // update the attachment between objects and solvers

private:
    friend class World;
    std::uint64_t        m_id;                    //!< unique id of the scene
    std::vector<Object*> m_objects;               //!< objects in the scene
    std::vector<Solver*> m_solvers;               //!< solvers in the scene
    bool                 m_attach_info_outdated;  //!< the attachment info of solvers need update
};

}  // namespace Physika