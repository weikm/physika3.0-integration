#pragma once
#include "collision/internal/collision_def.cuh"
#include "collision/internal/collision_vec3.cuh"
#include "collision/internal/collision_mat3f.cuh"


typedef struct __align__(16) _transf
{
    REAL3      _off;
    g_matrix3f _trf;

    /**@brief Return the basis matrix for the rotation */
    CU_FORCEINLINE __host__ __device__ g_matrix3f& getBasis()
    {
        return _trf;
    }
    /**@brief Return the basis matrix3f for the rotation.	 */
    CU_FORCEINLINE __host__ __device__ const g_matrix3f& getBasis() const
    {
        return _trf;
    }
    /**@brief Return the origin vector translation */
    CU_FORCEINLINE __host__ __device__ REAL3& getOrigin()
    {
        return _off;
    }
    /**@brief Return the origin vector translation */
    CU_FORCEINLINE __host__ __device__ const REAL3& getOrigin() const
    {
        return _off;
    }

    CU_FORCEINLINE __host__ __device__ _transf(g_matrix3f & trf, REAL3 off)
        : _trf(trf), _off(off)
    {
    }

    CU_FORCEINLINE __host__ __device__ REAL3 apply(REAL3 v)
    {
        return _trf * v + _off;
    }

    CU_FORCEINLINE __host__ __device__ _transf inverse() const
    {
        g_matrix3f inv = _trf.getTranspose();
        return _transf(inv, inv * -_off);
    }

    CU_FORCEINLINE __host__ __device__ _transf operator*(const _transf& t) const
    {
        return _transf(_trf * t._trf, (*this)(t._off));
    }

    /**@brief Return the transform of the vector */
    CU_FORCEINLINE __host__ __device__ REAL3 operator()(const REAL3& x) const
    {
        return dot3(x,
                    make_REAL3(_trf(0, 0), _trf(0, 1), _trf(0, 2)),
                    make_REAL3(_trf(1, 0), _trf(1, 1), _trf(1, 2)),
                    make_REAL3(_trf(2, 0), _trf(2, 1), _trf(2, 2)))
               + _off;
    }
}
g_transf;