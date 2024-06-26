#pragma once
#include "collision/internal/collision_vec3.hpp"
namespace Physika {
struct Ray
{
    Ray(vec3f origin, vec3f dir) 
    {
        m_origin = origin;
		m_dir = dir;
    }
    /**
     * @brief get the point on the ray
     *
     * @param[in] origin origin of the ray
     * @param[in] dir direction of the ray
     *
     */
    vec3f getPoint(double t) const
    {
        return m_origin + t * m_dir;
    }

    vec3f m_origin;  // origin of the ray
    vec3f m_dir;     // direction of the ray
};

}  // namespace Physika