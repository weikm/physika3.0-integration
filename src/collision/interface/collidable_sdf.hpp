/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collideable sigen distance field(SDF) component in Physika.
 * @version    : 1.0
 */
#pragma once
#include <vector>
#include "collision/internal/collision_mat3f.hpp"
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/distance_field3d.hpp"
/**
 * a component to add collidable SDF attribute to an object
 * in this version, we can just add this component to a object
 * which has a triangle mesh component
 */
namespace Physika {
struct CollidableSDFComponent
{
    //a temporary Constructor function

    CollidableSDFComponent() {
        vec3f                            lo(0.0f, 0.0f, 0.0f);
        vec3f                            hi(1.0f, 1.0f, 1.0f);
        m_sdf = new DistanceField3D;
        m_sdf->setSpace(lo - vec3f(0.025f, 0.025f, 0.025f), hi + vec3f(0.025f, 0.025f, 0.025f), 105, 105, 105);
        m_sdf->loadBox(lo, hi, true);
    }
    ~CollidableSDFComponent() {}

    //TO DO : reset function
    void reset() 
    {
    }

    vec3f getTranslation()
    {
        return m_translation;
    }
    matrix3f getRotationMatrix()
    {
        return m_rotation;
    }
    
    vec3f  m_translation;         // the translation of object
    matrix3f m_rotation;          // the rotation of object
    
    vec3f m_velocity;             // the velocity of object
    vec3f m_angular_velocity;     // the angular velocity of object
    
    DistanceField3D* m_sdf;       // the signed distance field of object

};
}  // namespace Physika