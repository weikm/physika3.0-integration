/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: solver for particles collision
 * @version    : 1.0
 */
#include "framework/solver.hpp"
#include "collision/interface/collidable_points.hpp"
#include "collision/internal/collision_pair.hpp"

namespace Physika {
// The results of collision between particles
struct ParticlesCollisionComponent
{
    ParticlesCollisionComponent() {}
    ~ParticlesCollisionComponent() {}

    //input
    vec3f* m_pos;     // the position of points
    int    m_num;     // the total num of points
    float  m_radius;  // the radius of points

    //output
    std::vector<id_pair> m_pairs;   //the id of contact pairs.
};

/**
 * SDFPointsCollisionDetect is a solver to detect the collision
 * between particles
 */
class ParticlesCollisionSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt;
        float m_total_time;
    };

public:
    ParticlesCollisionSolver();

    ~ParticlesCollisionSolver();

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
     * @return true if run successfully
     */
    bool doCollision();

    void getResult(std::vector<id_pair>& pairs)
    {
        pairs = m_particles->getComponent<ParticlesCollisionComponent>()->m_pairs;
    }

    int getMaxCollision()
    {
        return m_maxCollision;
    }

    void setMaxCollision(int maxCollision)
    {
        this->m_maxCollision = maxCollision;
    }

private:
    bool    m_is_init;    // the flag of initialization
    float   m_cur_time;   // current time
    int          m_maxCollision;  // the max num of collision
    SolverConfig m_config;
    Object* m_particles;  //!< particles

};
}  // namespace Physika