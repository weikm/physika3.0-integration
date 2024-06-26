/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-10-25
 * @description: transform of rigid body
 * @version    : 1.0
 */
#pragma once
#include "collision/internal/collision_mat3f.hpp"

namespace Physika {
class alignas(16) transf
{
public:
    vec3f    _off;
    matrix3f _trf;

    transf()
    {
        _trf = matrix3f::identity();
    }

    transf(const vec3f& off)
    {
        _off = off;
        _trf = matrix3f::identity();
    }

    transf(const vec3f& off, const matrix3f& trf)
    {
        _off = off;
        _trf = trf;
    }

    transf(const matrix3f& trf, const vec3f& off)
    {
        _off = off;
        _trf = trf;
    }

    FORCEINLINE void setOrigin(const vec3f& org)
    {
        _off = org;
    }

    FORCEINLINE void setRotation(const matrix3f& r)
    {
		_trf = r;
	}

    FORCEINLINE void setRotation(const quaternion& q)
    {
        _trf = matrix3f::rotation(q);
    }

    /**@brief Return the basis matrix for the rotation */
    FORCEINLINE matrix3f& getBasis()
    {
        return _trf;
    }
    /**@brief Return the basis matrix3f for the rotation.	 */
    FORCEINLINE const matrix3f& getBasis() const
    {
        return _trf;
    }
    /**@brief Return the origin vector translation */
    FORCEINLINE vec3f& getOrigin()
    {
        return _off;
    }
    /**@brief Return the origin vector translation */
    FORCEINLINE const vec3f& getOrigin() const
    {
        return _off;
    }

    /**@brief Return a quaternion representing the rotation */
    FORCEINLINE quaternion getRotation() const
    {
        quaternion q;
        _trf.getRotation(q);
        return q;
    }

    /**@brief Set this transformation to the identity */
    FORCEINLINE void setIdentity()
    {
        _trf = matrix3f::identity();
        _off = vec3f::zero();
    }

    FORCEINLINE vec3f getVertex(const vec3f& v) const
    {
        return _trf * v + _off;
    }

    FORCEINLINE vec3f getVertexInv(const vec3f& v) const
    {
        vec3f vv = v - _off;
        // return _trf.getInverse() * vv;
        return _trf.getTranspose() * vv;
    }

    FORCEINLINE transf inverse() const
    {
        matrix3f inv = _trf.getTranspose();
        return transf(inv, inv * -_off);
    }

    FORCEINLINE transf operator*(const transf& t) const
    {
        return transf(_trf * t._trf, (*this)(t._off));
    }

    /**@brief Return the transform of the vector */
    FORCEINLINE vec3f operator()(const vec3f& x) const
    {
        return x.dot3(
                   vec3f(_trf(0, 0), _trf(0, 1), _trf(0, 2)),
                   vec3f(_trf(1, 0), _trf(1, 1), _trf(1, 2)),
                   vec3f(_trf(2, 0), _trf(2, 1), _trf(2, 2)))
               + _off;
    }
};
};

