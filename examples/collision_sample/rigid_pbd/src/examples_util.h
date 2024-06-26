#ifndef RAW_PHYSICS_EXAMPLES_EXAMPLES_UTIL_H
#define RAW_PHYSICS_EXAMPLES_EXAMPLES_UTIL_H

#include "./render/mesh.h"
#include "./physics/collider.h"

Collider* examples_util_create_single_convex_hull_collider_array(Vertex* vertices, u32* indices, vec3 scale);
Collider examples_util_create_convex_hull_collider(Vertex* vertices, u32* indices, vec3 scale);

#endif