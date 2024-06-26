#pragma once

typedef struct __align__(16) _bvh_node {
	g_box _box;
	int _child; // >=0 leaf with tri_id, <0 left & right
	int _parent;

	CU_FORCEINLINE  __host__ __device__ g_box& box() { return _box; }
	CU_FORCEINLINE  __host__ __device__ _bvh_node* left() { return this - _child; }
	CU_FORCEINLINE  __host__ __device__ _bvh_node* right() { return this - _child + 1; }
	CU_FORCEINLINE  __host__ __device__ int triID() { return _child; }
	CU_FORCEINLINE  __host__ __device__ bool isLeaf() { return _child >= 0; }
	CU_FORCEINLINE  __host__ __device__ int parentID() { return _parent; }
} g_bvh_node;
