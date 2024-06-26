#pragma once
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/collision_pair.hpp"
#include <vector>
namespace Physika {
/**
 * SDFPointsCollisionDetect is a solver to detect the collision
 * between particles
 */
class ParticlesCollisonDetect
{
public:
    ParticlesCollisonDetect();

    /**
     * @brief construct for collision detection between two meshes
     * @param[in] points  points to do collision detection
     * @param[in] num     the num of points
     * @param[in] radius  the radius of points
     */
    ParticlesCollisonDetect(std::vector<vec3f> points, int num, float radius);

    ~ParticlesCollisonDetect();

    /**
     * @brief       set the points to do collision detect
     * @param[in]   the position of points
     */
    void setPoints(std::vector<vec3f> points);

    /**
     * @brief       set the radius of collision points
     * @param[int]  the radius of points
     */
    void setRadius(float radius);

    /**
     * @brief       get the collision result
     * @param[out]  the id of collision pairs
     */
    void getCollisionPairs(std::vector<id_pair>& pairs) const;

    /**
     * @brief do collision detect
     */
    bool execute();

private:
    // input
    std::vector<vec3f> m_points;  //!< collision points
    float  m_radius;      //!< collision radius
    int    m_num_points;  //!< the num of points

    // output
    std::vector<id_pair> m_pairs;  //!< collision pairs

};
}  // namespace Physika
