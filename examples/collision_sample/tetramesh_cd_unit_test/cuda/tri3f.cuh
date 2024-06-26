#pragma once

typedef unsigned int uint;

typedef struct _tri3f {
	uint3 _ids;

	CU_FORCEINLINE __device__ __host__ uint id0() const { return _ids.x; }
	CU_FORCEINLINE __device__ __host__ uint id1() const { return _ids.y; }
	CU_FORCEINLINE __device__ __host__ uint id2() const { return _ids.z; }
	CU_FORCEINLINE __device__ __host__ uint id(int i) const { return (i == 0 ? id0() : ((i == 1) ? id1() : id2())); }
} g_tri3f;
