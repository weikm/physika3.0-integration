/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-05
 * @description: util for ccd
 * @version    : 1.0
 */
#pragma once
#include "collision/internal/collision_vec3.hpp"

namespace Physika {
class CollisionUtil
{
public:
    /**
     * @brief ccd test for edge, return the collision time, if no collision occurs return -1;
     * 
     * @param[in] ta0       the start point of edge 0 at t0
     * @param[in] tb0       the end point of edge 0 at t0
     * @param[in] tc0       the start point of edge 1 at t0
     * @param[in] td0       the end point of edge 1 at t0
     * @param[in] ta1       the start point of edge 0 at t1
     * @param[in] tb1       the end point of edge 0 at t1
     * @param[in] tc1       the start point of edge 1 at t1
     * @param[in] td1       the end point of edge 1 at t1
     * @param[out] qi       the intersection point
     */
    static double
    Intersect_EE(const vec3f& ta0, const vec3f& tb0, const vec3f& tc0, const vec3f& td0, const vec3f& ta1, const vec3f& tb1, const vec3f& tc1, const vec3f& td1, vec3f& qi);

    /**
     * @brief ccd test for vertex and face, return the collision time, if no collision occurs return -1; 
     *
     * @param[in] ta0       the start point of edge 0 at t0
     * @param[in] tb0       the end point of edge 0 at t0
     * @param[in] tc0       the start point of edge 1 at t0
     * @param[in] ta1       the start point of edge 0 at t1
     * @param[in] tb1       the end point of edge 0 at t1
     * @param[in] tc1       the start point of edge 1 at t1
     * @param[in] q0        the vertex at t0
     * @param[in[ q1        the vertex at t1
     * @param[out] qi       the intersection point
     * @param[out] baryc    the barycentric coordinate of the intersection point
     */
    static double
    Intersect_VF(const vec3f& ta0, const vec3f& tb0, const vec3f& tc0, const vec3f& ta1, const vec3f& tb1, const vec3f& tc1, const vec3f& q0, const vec3f& q1, vec3f& qi, vec3f& baryc);
};
}

