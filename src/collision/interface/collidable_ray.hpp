/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collideable ray component in Physika.
 * @version    : 1.0
 */

#pragma once
#include "collision/internal/ray.hpp"


/**
 * a component to add collidable ray attribute to an object
 * for exapmle: the tradjectory of a bullet
 */
namespace Physika
{
    struct CollidableRayComponent
    {
        void reset()
        {
            m_ray = nullptr;
        }
        Ray* m_ray;
    };
}