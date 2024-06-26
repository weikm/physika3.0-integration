#include "examples_util.h"
#include <light_array.h>
#include "./render/obj.h"

Collider* examples_util_create_single_convex_hull_collider_array(Vertex* vertices, u32* indices, vec3 scale) {
	vec3* vertices_positions = array_new(vec3);
	for (u32 i = 0; i < array_length(vertices); ++i) {
		vec3 position = (vec3) {
			(r64)vertices[i].position.x,
			(r64)vertices[i].position.y,
			(r64)vertices[i].position.z
		};
		position.x *= scale.x;
		position.y *= scale.y;
		position.z *= scale.z;
		array_push(vertices_positions, position);
	}
	Collider collider = collider_convex_hull_create(vertices_positions, indices);
	array_free(vertices_positions);

	Collider* colliders = array_new(Collider);
	array_push(colliders, collider);
	return colliders;
}

Collider* examples_util_create_sphere_convex_hull_array(r32 radius) {
	Collider collider = collider_sphere_create(radius);
	Collider* colliders = array_new(Collider);
	array_push(colliders, collider);

	return colliders;
}

Collider examples_util_create_convex_hull_collider(Vertex* vertices, u32* indices, vec3 scale) {
	vec3* vertices_positions = array_new(vec3);
	for (u32 i = 0; i < array_length(vertices); ++i) {
		vec3 position = (vec3) {
			(r64)vertices[i].position.x,
			(r64)vertices[i].position.y,
			(r64)vertices[i].position.z
		};
		position.x *= scale.x;
		position.y *= scale.y;
		position.z *= scale.z;
		array_push(vertices_positions, position);
	}
	Collider collider = collider_convex_hull_create(vertices_positions, indices);
	array_free(vertices_positions);
	return collider;
}

