#pragma once
#include <gm.h>

#pragma pack(push, 1)
typedef struct {
	fvec3 position;
	fvec3 normal;
	fvec2 texture_coordinates;
} Vertex;
#pragma pack(pop)

typedef struct {
	r32* vdata;
	u32* idata;
	int dl;

	u32 num_indices;
} Mesh;

Mesh graphics_mesh_create(Vertex* vertices, u32* indices);
void mesh_destroy(Mesh* mesh);
void render_mesh(Mesh& mesh, r32* mat);
