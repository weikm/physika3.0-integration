/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collideable height field(HF) component in Physika.
 * @version    : 1.0
 */
#pragma once
#include <vector>

#include "collision/internal/height_field.hpp"
#include "collision/internal/collision_vec3.hpp"
/**
 * a component to add collidable HF attribute to an object
 */
namespace Physika {
struct CollidableHeightFieldComponent
{
    CollidableHeightFieldComponent()
    {
        m_land = new heightField1d;
    }
    void reset()
    {
    }

    heightField1d* m_land;  // height of land
};
}  // namespace Physika