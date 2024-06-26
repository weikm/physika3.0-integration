/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: solver for mesh-ray collision
 * @version    : 1.0
 */
#pragma once

#include <vector>

#include "framework/solver.hpp"
#include "collision/internal/collision_vec3.hpp"
#include "collision/interface/collidable_ray.hpp"
#include "collision/interface/collidable_trianglemesh.hpp"

namespace Physika {
// the result of collision between mesh and ray
struct MeshRayCollisionComponent
{
    MeshRayCollisionComponent() {}
    ~MeshRayCollisionComponent() {}

    // input
    triMesh* m_mesh;
    Ray*     m_ray;

    vec3f m_lcsXDir;    //!< x direction of local coordinate system
    vec3f m_lcsYDir;    //!< y direction of local coordinate system
    vec3f m_lcsZDir;    //!< z direction of local coordinate system
    vec3f m_lcsOrigin;  //!< origin of local coordinate system
    bool  m_transformToLCS;  //!< whether transform the result into local coordinate system, now is just used for HNU

    // output
    vec3f m_intersectPoint;      //!< the first intersect point of ray and mesh
    vec3f m_lcsIntersectPoint;   //!< the first intersect point of ray and mesh in local coordinate system
    vec3f m_intersectNormal;     //!< the normal of the intersect point
    vec3f m_lcsIntersectNormal;  //!< the normal of the intersect point in local coordinate system
    vec3f m_lcsRayDir;           //!< the direction of ray in local coordinate system
    vec3f m_rayDir;              //!< the direction of ray
};

/**
 * MeshRayCollisionSolver is a sample solver for collision detect
 * it detect collision between two object, one should have CollidableTriangleMeshComponent
 * and other should have collidableRayComponent. now it can only use cpu to detect collision
 */
class MeshRayCollisionSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt;
        float m_total_time;
    };

public:
    MeshRayCollisionSolver();
    MeshRayCollisionSolver(const vec3f lcsXDir, const vec3f lcsYDir, const vec3f lcsZDir, const vec3f lcsOrigin, const bool transformToLCS);
    ~MeshRayCollisionSolver();

    /**
     * @brief initialize the solver to get it ready for execution.
     *
     * @return  true if initialization succeeds, otherwise return false
     *
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
     *        have the CollidableTriangleMesh component and collidableRay component
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    bool isApplicable(const Object* object) const override;

    /**
     * @brief attach an object to the solver, in this stage we not have exception handling mechanism
     *        so in this solver, can only attach two object, one should have
     *        trianglemeshComponent and another should have collidablerayComponent
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

    /**
     * @brief get the collision result of mesh and ray
     *
     * @return true if collide
     */
    bool doCollision();

    /**
     * @brief transform the position of ray mesh intersection point from world space to local space
     *
     * @return true if procedure successfully completes, otherwise return false
     */
    bool transform();

    /**
     * @brief get the result of mesh ray collision, in this solver, it returns the posiition of the
     *        first intersection point of ray and mesh and the direction of ray, if the m_transformToLCS
     *        is true, it returns the position of the first intersection point of ray and mesh and ray
     *        direction in local
     *
     */
    void getResult(vec3f& intersectPoint, vec3f& rayDir) const;

    /**
     * @brief set the lcs of the mesh component if need to get the local coordinate of result
     *
     */
    void setLCS(const vec3f lcsXDir, const vec3f lcsYDir, const vec3f lcsZDir, const vec3f lcsOrigin, const bool transformToLCS);

private:
    
    bool         m_is_init;         // the flag of initialization
    float        m_cur_time;        // current time
    
    SolverConfig m_config;
    // input
    Object*  m_mesh_ray_object;  //!< collision object
};
}  // namespace Physika