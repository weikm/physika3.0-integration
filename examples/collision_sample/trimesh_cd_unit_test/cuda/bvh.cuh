#pragma once

typedef struct __align__(16) _bvh_node {
	g_box _box;
	int _child; // >=0 leaf with tri_id, <0 left & right

	CU_FORCEINLINE  __host__ __device__ const g_box& box() const { return _box; }
	CU_FORCEINLINE  __host__ __device__ const _bvh_node* left() const { return this - _child; }
	CU_FORCEINLINE  __host__ __device__ const _bvh_node* right() const { return this - _child + 1; }
	CU_FORCEINLINE  __host__ __device__ int triID() const { return _child; }
	CU_FORCEINLINE  __host__ __device__ bool isLeaf() const { return _child >= 0; }
} g_bvh_node;


typedef struct __align__(16) _qbvh_node {
	//12 bytes
	unsigned short int	bxMin[3];
	unsigned short int	bxMax[3];
	//4 bytes
	int _child; // >=0 leaf with tri_id, <0 left & right

	CU_FORCEINLINE  __host__ __device__ const _qbvh_node* left() const { return this - _child; }
	CU_FORCEINLINE  __host__ __device__ const _qbvh_node* right() const { return this - _child + 1; }
	CU_FORCEINLINE  __host__ __device__ int triID() const { return _child; }
	CU_FORCEINLINE  __host__ __device__ bool isLeaf() const { return _child >= 0; }

	CU_FORCEINLINE  __host__ __device__ bool testQuantizedBoxOverlapp(
		const unsigned short* quantizedMin, const unsigned short* quantizedMax) const
	{
		if (bxMin[0] > quantizedMax[0] ||
			bxMax[0] < quantizedMin[0] ||
			bxMin[1] > quantizedMax[1] ||
			bxMax[1] < quantizedMin[1] ||
			bxMin[2] > quantizedMax[2] ||
			bxMax[2] < quantizedMin[2])
		{
			return false;
		}
		return true;
	}

} g_qbvh_node;


typedef struct __align__(16) _sbvh_node {
	//12 bytes
	unsigned short int	bxMin[3];
	unsigned short int	bxMax[3];
	//4 bytes
	int _child; // >=0 leaf with tri_id, <0 left & right

	CU_FORCEINLINE  __host__ __device__ const _sbvh_node* left(_sbvh_node * upperPtr, _sbvh_node * lowerPtr, int upperNum) const
	{
		if ((this - upperPtr) < upperNum) {
			const _sbvh_node* r = this - _child;
			if ((r - upperPtr) < upperNum)
				return r;
			else
				return lowerPtr + (r - upperPtr) - upperNum;
		}
		else
			return this - _child;
	}

	CU_FORCEINLINE  __host__ __device__ const _sbvh_node* right(_sbvh_node* upperPtr, _sbvh_node* lowerPtr, int upperNum) const
	{
		if ((this - upperPtr) < upperNum) {
			const _sbvh_node* r = this - _child + 1;
			if ((r - upperPtr) < upperNum)
				return r;
			else
				return lowerPtr + (r - upperPtr) - upperNum;
		}
		else
			return this - _child + 1;
	}

	CU_FORCEINLINE  __host__ __device__ int triID() const { return _child; }
	CU_FORCEINLINE  __host__ __device__ bool isLeaf() const { return _child >= 0; }

	CU_FORCEINLINE  __host__ __device__ bool testQuantizedBoxOverlapp(
		const unsigned short* quantizedMin, const unsigned short* quantizedMax) const
	{
		if (bxMin[0] > quantizedMax[0] ||
			bxMax[0] < quantizedMin[0] ||
			bxMin[1] > quantizedMax[1] ||
			bxMax[1] < quantizedMin[1] ||
			bxMin[2] > quantizedMax[2] ||
			bxMax[2] < quantizedMin[2])
		{
			return false;
		}
		return true;
	}

} g_sbvh_node;

