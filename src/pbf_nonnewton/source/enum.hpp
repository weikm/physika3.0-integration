//@author        : Long Shen
//@date          : 2023/10/8
//@description   :
//@version       : 1.0

#ifndef PHYSIKA_ENUM_HPP
#define PHYSIKA_ENUM_HPP

#include <iostream>

namespace Physika {

/**
 * @brief the simMaterial of particles
 */
enum SimMaterial : uint32_t
{
    FLUID,
    BOUND,
    RIGID
};
}

#endif  // PHYSIKA_ENUM_HPP
