#pragma once
#include "collision/interface/collidable_sdf.hpp"
#include "collision/interface/collidable_points.hpp"
#include "collision/internal/collision_aabb.hpp"
#include "framework/solver.hpp"
#include "framework/object.hpp"
namespace Physika {

// the result of collision between points and sdf
struct SDFPointsCollisionComponent {
    SDFPointsCollisionComponent() {}
    ~SDFPointsCollisionComponent() {}

    //input
    vec3f    m_translation;  // the translation of object
    matrix3f m_rotation;     // the rotation of object

    vec3f m_velocity;          // the velocity of object
    vec3f m_angular_velocity;  // the angular velocity of object

    DistanceField3D* m_sdf;  // the signed distance field of object

    vec3f* m_pos;     // the position of points
    int    m_num;     // the total num of points
    float  m_radius;  // the radius of points

    //output
    int    m_num_collisions;      // the num of collisions
    int*   m_collision_id;        // the id of collide particle
    vec3f* m_collision_normal;    // the normal of collision
    float* m_collision_distance;  // the distance of penetration
};

/**
 * SDFPointsCollisionSolver is a sample solver to detect the collision 
 * between two object, one should have collidable sdf component and
 * the other should have collidable points component
 * for example a car's sdf and the sand particles
 */ 
class SDFPointsCollisionSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt;
        float m_total_time;
    };

public:
    SDFPointsCollisionSolver();
    ~SDFPointsCollisionSolver();

    /**
     * @brief initialize the solver to get it ready for execution.
     *        The behavior of duplicate calls is up to the developers of subclasses, yet it is
     *        recommended that duplicate calls should be ignored to avoid redundant computation.
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
     *        have the CollidableSDF component and CollidablePoints component
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    bool isApplicable(const Object* object) const override;

    /**
     * @brief attach an object to the solver, in this stage we not have exception handling mechanism
     *        so in this solver, can only attach two object, one should have
     *        collidableSDFComponent and another should have collidablePointsComponent
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
     * @brief get the collision result of mesh and height field
     *
     * @return true if collide
     */
    bool doCollision();

    //bool doCollisionCPU();

    void getResult(std::vector<vec3f>& res);

private:
    bool m_is_init;               //the flag of initialization 
    float m_cur_time;              // current time
    int          m_maxCollision;          // the max num of collision
    SolverConfig m_config;
    // input
    Object* m_sdf_points_object;  //!< collision object 

};
}  // namespace Physika