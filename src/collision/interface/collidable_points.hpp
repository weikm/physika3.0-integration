/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collideable ray component in Physika.
 * @version    : 1.0
 */

#pragma once
#include <vector>

#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/collision_pair.hpp"
#include "framework/object.hpp"

/**
 * a component to add collidable points attribute to points
 */
namespace Physika {
struct CollidablePointsComponent
{
    // TO Do: finish reset function
    void reset()
    {
        // m_pos.clear();
    }
    vec3f* m_pos;     // the position of points
    int    m_num;     // the total num of points
    float  m_radius;  // the radius of points

};
}  // namespace Physika