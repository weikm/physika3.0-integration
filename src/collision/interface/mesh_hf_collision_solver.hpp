/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: solver for mesh-hf collision
 * @version    : 1.0
 */

#pragma once

#include <vector>
#include <assert.h>

#include "framework/solver.hpp"
#include "collision/internal/trimesh.hpp"
#include "collision/internal/height_field.hpp"
#include "collision/internal/collision_vec3.hpp"

namespace Physika {
struct MeshHeightFieldCollisionComponent
{
    MeshHeightFieldCollisionComponent() {}
    ~MeshHeightFieldCollisionComponent() {}
    // input
    triMesh*       m_trimesh;
    heightField1d* m_heightField;

    // output
    int*   m_collision_id;      // the id of collision point
    vec3f* m_collision_normal;  // the normal of collision point
    int    m_counter;           // the num of collisions
};

/**
 * MeshHeightFieldCollisionSolver is a sample solver for collision detect
 * one object should have CollidableTriangleMeshComponent and another object
 * should have collidableHeightFieldComponent, now it can only use gpu to detect
 */
class MeshHeightFieldCollisionSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt;
        float m_total_time;
    };

public:
    MeshHeightFieldCollisionSolver();
    ~MeshHeightFieldCollisionSolver();

    /**
     * @brief initialize the solver to get it ready for execution.
     *
     * @return  true if initialization succeeds, otherwise return false
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
     * @brief run the solver in a time step, in this solver, step() is the same as run()
     *
     * @return    true if reset succeeds, otherwise return false
     */
    bool step() override;

    /**
     * @brief run the solver to get the collision results
     *
     * @return true if procedure successfully completes, otherwise return false
     */
    bool run() override;

    /**
     * @brief check whether the solver is applicable to given object
     *        in this case, the solver is applicable to objects that
     *        have the CollidableTriangleMesh component and CollidableHeightField component
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    bool isApplicable(const Object* object) const override;

    /**
     * @brief attach an object to the solver, in this stage we not have exception handling mechanism
     *        so in this solver, can only attach two object, one should have
     *        trianglemeshComponent and another should have collidableHeightFiledComponent
     *
     * @param[in] object    the object to attach
     *
     * @return    true if the object is successfully attached, otherwise return false
     */
    bool attachObject(Object* object) override;

    /**
     * @brief detach an object from the solver
     *
     * @param[in] object    the object to detach
     *
     * @return    true if the object is successfully detached, otherwise return false
     */
    bool detachObject(Object* object) override;

    /**
     * @brief clear the attachment of the solver
     */
    void clearAttachment() override;


    int getMaxCollision()
    {
        return m_maxCollision;
    }

    void setMaxCollision(int maxCollision)
    {
        this->m_maxCollision = maxCollision;
    }

private:
    bool       m_is_init;       // the flag of initialization
    float         m_cur_time;   // current time
    SolverConfig m_config;
    int  m_maxCollision;  // the max num of collisions
    Object* m_mesh_heightField_object;  // mesh heightField object
};
}  // namespace Physika
