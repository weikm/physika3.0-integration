#include "spot_storm.h"

#include <light_array.h>
#include <stdio.h>
#include <math.h>
#include "entity.h"
#include "util.h"
#include "examples_util.h"
#include "./render/obj.h"
#include "./physics/pbd.h"

static Collider* create_spot_colliders(vec3 scale, Vertex** hulls_vertices, u32** hulls_indices) {
	Collider* spot_colliders = array_new(Collider);
	Collider collider;

	for (u32 i = 0; i < array_length(hulls_vertices); ++i) {
		collider = examples_util_create_convex_hull_collider(hulls_vertices[i], hulls_indices[i], scale);
		array_push(spot_colliders, collider);
	}

	return spot_colliders;
}

static Quaternion generate_random_quaternion() {
	r64 x = rand() / (r64)RAND_MAX;
	r64 y = rand() / (r64)RAND_MAX;
	r64 z = rand() / (r64)RAND_MAX;
	r64 angle = rand() / (r64)RAND_MAX;
	angle = -180.0 + angle * 360.0;
	return quaternion_new((vec3) {x, y, z}, angle);
}

int ex_spot_storm_init() {
	entity_module_init();

	Vertex* cube_vertices;
	u32* cube_indices;
	obj_parse("../../../media/cube.obj", &cube_vertices, &cube_indices);
	Mesh cube_mesh = graphics_mesh_create(cube_vertices, cube_indices);

	vec3 floor_scale = (vec3){50.0, 1.0, 50.0};
	Collider* floor_colliders = examples_util_create_single_convex_hull_collider_array(cube_vertices, cube_indices, floor_scale);
	entity_create_fixed(cube_mesh, (vec3){0.0, -2.0, 0.0}, quaternion_new((vec3){0.0, 1.0, 0.0}, 0.0),
		floor_scale, (vec4){1.0, 1.0, 1.0, 1.0}, floor_colliders, 0.5, 0.5, 0.0);
	
	Vertex* spot_vertices;
	u32* spot_indices;
	obj_parse("../../../media/spot/spot.obj", &spot_vertices, &spot_indices);
	Mesh spot_mesh = graphics_mesh_create(spot_vertices, spot_indices);
	vec3 spot_scale = (vec3){2.0, 2.0, 2.0};

	Vertex** hulls_vertices = array_new(Vertex*);
	u32** hulls_indices = array_new(u32*);
	Vertex* hull_vertices;
	u32* hull_indices;
	obj_parse("../../../media/spot/spot-hull-1.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-2.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-3.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-4.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-5.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-6.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-7.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-8.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-9.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-10.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);
	obj_parse("../../../media/spot/spot-hull-11.obj", &hull_vertices, &hull_indices);
	array_push(hulls_vertices, hull_vertices);
	array_push(hulls_indices, hull_indices);

	const u32 N = 3;
	r64 y = 2.0;
	r64 gap = 3.5;
	for (u32 i = 0; i < N; ++i) {
		y += gap;

		r64 x = -2.0 * (N / 2.0);
		for (u32 j = 0; j < N; ++j) {
			x += gap;

			r64 z = -2.0 * (N / 2.0);
			for (u32 k = 0; k < N; ++k) {
				z += gap;

				Collider* spot_colliders = create_spot_colliders(spot_scale, hulls_vertices, hulls_indices);
				entity_create(spot_mesh, (vec3){x, y, z}, generate_random_quaternion(),
					spot_scale, util_pallete(i + j + k), 1.0, spot_colliders, 0.8, 0.8, 0.0);
			}
		}
	}

	array_free(cube_vertices);
	array_free(cube_indices);
	array_free(spot_vertices);
	array_free(spot_indices);
	for (u32 i = 0; i < array_length(hulls_vertices); ++i) {
		array_free(hulls_vertices[i]);
		array_free(hulls_indices[i]);
	}
	array_free(hulls_vertices);
	array_free(hulls_indices);

	return 0;
}

void ex_spot_storm_destroy()
{
	Entity** entities = entity_get_all();
	for (u32 i = 0; i < array_length(entities); ++i) {
		Entity* e = entities[i];
		colliders_destroy(e->colliders);
		array_free(e->colliders);
		mesh_destroy(&e->mesh);
		entity_destroy(e);
	}
	array_free(entities);
	entity_module_destroy();
}

void ex_spot_storm_update(r64 delta_time)
{
	//printf("(Quaternion){%f, %f, %f, %f}\n", camera.rotation.x, camera.rotation.y, camera.rotation.z, camera.rotation.w);
	//printf("(Quaternion){%f, %f, %f, %f}\n", camera.yrotation.x, camera.yrotation.y, camera.yrotation.z, camera.yrotation.w);
	//printf("(vec3){%f, %f, %f}\n", camera.position.x, camera.position.y, camera.position.z);
	//printf("ex_spot_storm_update\n");

	Entity** entities = entity_get_all();
	for (u32 i = 0; i < array_length(entities); ++i) {
		Entity* e = entities[i];
		colliders_update(e->colliders, e->world_position, &e->world_rotation);
	}

	const r64 GRAVITY = 10.0;
	for (u32 i = 0; i < array_length(entities); ++i) {
		entity_add_force(entities[i], (vec3){0.0, 0.0, 0.0}, (vec3){0.0, -GRAVITY * 1.0 / entities[i]->inverse_mass, 0.0}, false);
	}

	pbd_simulate(delta_time, entities, 1, 1, true);

	for (u32 i = 0; i < array_length(entities); ++i) {
		entity_clear_forces(entities[i]);
	}
	array_free(entities);
}

extern void render_mesh(Mesh& mesh, r32 *mdlMtrx);



void util_matrix_to_r32_array2(const mat4* m, r32 out[16]) {
	out[0] = (r32)m->data[0][0];
	out[1] = (r32)m->data[1][0];
	out[2] = (r32)m->data[2][0];
	out[3] = (r32)m->data[3][0];
	out[4] = (r32)m->data[0][1];
	out[5] = (r32)m->data[1][1];
	out[6] = (r32)m->data[2][1];
	out[7] = (r32)m->data[3][1];
	out[8] = (r32)m->data[0][2];
	out[9] = (r32)m->data[1][2];
	out[10] = (r32)m->data[2][2];
	out[11] = (r32)m->data[3][2];
	out[12] = (r32)m->data[0][3];
	out[13] = (r32)m->data[1][3];
	out[14] = (r32)m->data[2][3];
	out[15] = (r32)m->data[3][3];
}

void ex_spot_storm_render()
{
	Entity** entities = entity_get_all();
	for (u32 i = 0; i < array_length(entities); ++i) {
		Entity* ent = entities[i];
		mat4 model_matrix = entity_get_model_matrix(ent);

		r32 model[16];
		util_matrix_to_r32_array2(&model_matrix, model);

		render_mesh(ent->mesh, model);
	}

	array_free(entities);
}

