#pragma once

typedef struct _kmesh{
	int _num_vtx;
	int _num_tri;
	g_tri3f* _tris;
	REAL3* _vtxs;
	int _num_bvh_nodes;
	g_bvh_node* _bvh_nodes;
} g_mesh;

#if 0
typedef struct _rigid {
	g_matrix3f *_rot;
	REAL3 _off;
	int _mesh;
} g_rigid;
#endif

