#pragma once
#include "collision/internal/collision_bvh.hpp"
namespace Physika
{
class QBVHUtils
{
public:
    // this function is util for build qbvh
    // it divide the whole space
    // caculate the parameter of the qbvh
    static FORCEINLINE void calc_quantization_parameters(
        BOX<REAL>&       boundOut,
        vec3f&           bvhQuantization,
        const BOX<REAL>& boundIn,
        REAL             quantizationMargin)
    {
        const vec3f& srcMinBound = boundIn.getMin();
        const vec3f& srcMaxBound = boundIn.getMax();

        // enlarge the AABB to avoid division by zero when initializing the quantization values
        vec3f clampValue(quantizationMargin, quantizationMargin, quantizationMargin);

        boundOut.setMin(srcMinBound - clampValue);
        boundOut.setMax(srcMaxBound + clampValue);
        vec3f aabbSize  = boundOut.getMax() - boundOut.getMin();
        bvhQuantization = vec3f(
            REAL(65535.0) / aabbSize.x,
            REAL(65535.0) / aabbSize.y,
            REAL(65535.0) / aabbSize.z);
    }

    static FORCEINLINE void quantize_clamp(
        unsigned short*  out,
        const vec3f&     point,
        const BOX<REAL>& bx,
        const vec3f&     bvhQuantization)
    {
        const vec3f& min_bound = bx.getMin();
        const vec3f& max_bound = bx.getMax();

        vec3f clampedPoint(point);
        clampedPoint.setMax(min_bound);
        clampedPoint.setMin(max_bound);

        vec3f v = (clampedPoint - min_bound) * bvhQuantization;
        out[0]  = ( unsigned short )(v.x + 0.5f);
        out[1]  = ( unsigned short )(v.y + 0.5f);
        out[2]  = ( unsigned short )(v.z + 0.5f);
    }

    static FORCEINLINE vec3f unquantize(
        const unsigned short* vecIn,
        const vec3f&          offset,
        const vec3f&          bvhQuantization)
    {
        vec3f vecOut(
            ( REAL )(vecIn[0]) / (bvhQuantization.x),
            ( REAL )(vecIn[1]) / (bvhQuantization.y),
            ( REAL )(vecIn[2]) / (bvhQuantization.z));
        vecOut += offset;
        return vecOut;
    }
};
}