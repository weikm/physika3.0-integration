#pragma once

#include "collision/internal/collision_def.cuh"
typedef struct __align__(16) _matrix3f
{
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

    CU_FORCEINLINE __host__ __device__ _matrix3f()
    {
        for (int i = 0; i < 9; i++)
            _data[i] = 0;
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f(const _matrix3f& other)
    {
        for (int i = 0; i < 9; i++)
            _data[i] = other._data[i];
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f(REAL entry00, REAL entry01, REAL entry02, REAL entry10, REAL entry11, REAL entry12, REAL entry20, REAL entry21, REAL entry22)
    {
        _data[0] = entry00, _data[3] = entry01, _data[6] = entry02;
        _data[1] = entry10, _data[4] = entry11, _data[7] = entry12;
        _data[2] = entry20, _data[5] = entry21, _data[8] = entry22;
    }

    CU_FORCEINLINE __host__ __device__ REAL3 operator*(const REAL3& rhs) const
    {
        return make_REAL3(
            _data[0 + 0 * 3] * rhs.x + _data[0 + 1 * 3] * rhs.y + _data[0 + 2 * 3] * rhs.z,
            _data[1 + 0 * 3] * rhs.x + _data[1 + 1 * 3] * rhs.y + _data[1 + 2 * 3] * rhs.z,
            _data[2 + 0 * 3] * rhs.x + _data[2 + 1 * 3] * rhs.y + _data[2 + 2 * 3] * rhs.z);
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f getTranspose() const
    {
        return _matrix3f(
            _data[0], _data[1], _data[2], _data[3], _data[4], _data[5], _data[6], _data[7], _data[8]);
    }

    // Default assignment operator is fine.
    CU_FORCEINLINE __host__ __device__ REAL operator()(size_t row, size_t col) const
    {
        assert(row < 3 && col < 3);
        return _data[col * 3 + row];
    }

    CU_FORCEINLINE __host__ __device__ REAL& operator()(size_t row, size_t col)
    {
        assert(row < 3 && col < 3);
        return _data[col * 3 + row];
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f operator*(const _matrix3f& rhs) const
    {
        _matrix3f result;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                REAL val = 0;
                for (int i = 0; i < 3; ++i)
                {
                    val += operator()(r, i) * rhs(i, c);
                }
                result(r, c) = val;
            }
        }
        return result;
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f getInverse() const
    {
        _matrix3f result(
            operator()(1, 1) * operator()(2, 2) - operator()(1, 2) * operator()(2, 1),
            operator()(0, 2) * operator()(2, 1) - operator()(0, 1) * operator()(2, 2),
            operator()(0, 1) * operator()(1, 2) - operator()(0, 2) * operator()(1, 1),

            operator()(1, 2) * operator()(2, 0) - operator()(1, 0) * operator()(2, 2),
            operator()(0, 0) * operator()(2, 2) - operator()(0, 2) * operator()(2, 0),
            operator()(0, 2) * operator()(1, 0) - operator()(0, 0) * operator()(1, 2),

            operator()(1, 0) * operator()(2, 1) - operator()(1, 1) * operator()(2, 0),
            operator()(0, 1) * operator()(2, 0) - operator()(0, 0) * operator()(2, 1),
            operator()(0, 0) * operator()(1, 1) - operator()(0, 1) * operator()(1, 0));

        REAL det =
        operator()(0, 0) * result(0, 0) +
        operator()(0, 1) * result(1, 0) +
        operator()(0, 2) * result(2, 0);

        assert(!is_equal2(det, 0));

        REAL invDet = 1.0f / det;
        for (int i = 0; i < 9; ++i)
            result._data[i] *= invDet;

        return result;
    }

    CU_FORCEINLINE __host__ __device__ _matrix3f& operator*=(const _matrix3f& rhs)
    {
        return operator=(operator*(rhs));
    }

    CU_FORCEINLINE __host__ __device__ REAL3 operator()(size_t row) const
    {
        return getRow(row);
    }

    CU_FORCEINLINE __host__ __device__ REAL3 getRow(int i) const
    {
        assert(0 <= i && i < 3);
        return make_REAL3(_data[i], _data[3 + i], _data[6 + i]);
    }
}
g_matrix3f;

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
