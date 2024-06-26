#pragma once

typedef struct __align__(16) _matrix3f {
	REAL _data[9];

#if 0
	CU_FORCEINLINE void print() const{
		for  (int i=0; i<9; i++)
			if (i == 8)
				printf("%lf\n", _data[i]);
			else
				printf("%lf, ", _data[i]);
	}

	CU_FORCEINLINE __host__ __device__ REAL operator()(size_t row, size_t col) const {
		assert(row < 3 && col < 3);
		return _data[col * 3 + row];
	}
#endif

	CU_FORCEINLINE __host__ __device__ REAL3 operator*(const REAL3 & rhs) const {
		return make_REAL3(
			_data[0 + 0 * 3] * rhs.x + _data[0 + 1 * 3] * rhs.y + _data[0 + 2 * 3] * rhs.z,
			_data[1 + 0 * 3] * rhs.x + _data[1 + 1 * 3] * rhs.y + _data[1 + 2 * 3] * rhs.z,
			_data[2 + 0 * 3] * rhs.x + _data[2 + 1 * 3] * rhs.y + _data[2 + 2 * 3] * rhs.z);
	}
} g_matrix3f;

#if 0
//! Multiply row vector by matrix, v^T * M
CU_FORCEINLINE __host__ __device__  REAL3 operator*(const REAL3& lhs, const g_matrix3f& rhs) {
	return make_REAL3(
		lhs.x * rhs(0, 0) + lhs.y * rhs(1, 0) + lhs.z * rhs(2, 0),
		lhs.x * rhs(0, 1) + lhs.y * rhs(1, 1) + lhs.z * rhs(2, 1),
		lhs.x * rhs(0, 2) + lhs.y * rhs(1, 2) + lhs.z * rhs(2, 2)
	);
}
#endif
