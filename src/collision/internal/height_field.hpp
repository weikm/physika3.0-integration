/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: height field in Physika
 * @version    : 1.0
 */

#pragma once
#include "collision/internal/collision_vec3.hpp"

namespace Physika {

#define COMM_FUNC __device__ __host__
#ifndef m_max
#define m_max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef m_min
#define m_min(a, b) (((a) < (b)) ? (a) : (b))
#endif

class heightField1d
{
public:
    heightField1d()
        : m_nx(1), m_ny(1), m_pitch(1)
    {
        m_data = new float[1];
        m_data[0] = 1;
        origin = vec3f(0.0f, 0.0f, 0.0f);
        m_dx   = 1.0f;
        m_dz   = 1.0f;
    }

    ~heightField1d(){};

    inline int Nx() const
    {
        return m_nx;
    }
    inline int Ny() const
    {
        return m_ny;
    }
    inline int Pitch()
    {
        return m_pitch;
    }

    inline float operator()(const int i, const int j) const
    {
        return m_data[i + j * m_pitch];
    }

    inline float& operator()(const int i, const int j)
    {
        return m_data[i + j * m_pitch];
    }

    inline float get(float x, float z)
    {   
        float gx = (x -origin[0])
        / m_dx + Nx() / 2.0 - 0.5;
        float gz  = (z - origin[2]) / m_dz + Ny() / 2.0 - 0.5;
        int   gix = ( int )gx, giz = ( int )gz;
        gix         = m_min(gix, Nx() - 1);
        gix         = m_max(gix, 0);
        giz         = m_min(giz, Ny() - 1);
        giz         = m_max(giz, 0);
        int gix_1   = gix + 1;
        gix_1       = m_min(gix_1, Nx() - 1);
        int giz_1   = giz + 1;
        giz_1       = m_min(giz_1, Ny() - 1);
        float fracx = gx - gix, fracz = gz - giz;

        float val00 = ( *this )(gix, giz);
        float val10 = ( *this )(gix_1, giz);
        float val01 = ( *this )(gix, giz_1);
        float val11 = ( *this )(gix_1, giz_1);

        return val00 * ((1.0 - fracx) * (1.0 - fracz)) + val01 * ((1.0 - fracx) * fracz) + val10 * (fracx * (1.0 - fracz)) + val11 * (fracx * fracz);
    }

    inline void gradient(float x, float z, float& dhdx, float& dhdz)
    {
        float gx = (x - origin[0]) / m_dx + Nx() / 2.0 - 0.5;
        float gz = (z - origin[2]) / m_dz + Ny() / 2.0 - 0.5;

        gx = m_min(gx, Nx() - 1.0);
        gx = m_max(gx, 0.0);
        gz = m_min(gz, Ny() - 1.0);
        gz = m_max(gz, 0.0);

        int gix = ( int )(gx), giz = ( int )(gz);
        int gix_1 = gix - 1, giz_1 = giz - 1;
        gix_1 = m_max(gix_1, 0);
        gix_1 = m_min(gix_1, Nx() - 1);
        giz_1 = m_max(giz_1, 0);
        giz_1 = m_min(giz_1, Ny() - 1);

        int gix_2 = gix + 1, giz_2 = giz + 1;
        gix_2 = m_max(gix_2, 0);
        gix_2 = m_min(gix_2, Nx() - 1);
        giz_2 = m_max(giz_2, 0);
        giz_2 = m_min(giz_2, Ny() - 1);

        float fracx = (gx - gix_1) / ( float )(gix_2 - gix_1);
        float fracz = (gz - giz_1) / ( float )(giz_2 - giz_1);

        float val00 = ( *this )(gix_1, giz_1);
        float val10 = ( *this )(gix_2, giz_1);
        float val01 = ( *this )(gix_1, giz_2);
        float val11 = ( *this )(gix_2, giz_2);
        dhdx        = ((1.0 - fracz) * (val10 - val00) + fracz * (val11 - val01)) / (m_dx * (gix_2 - gix_1));
        dhdz        = ((1.0 - fracx) * (val01 - val00) + fracx * (val11 - val10)) / (m_dz * (giz_2 - giz_1));
    }

    /**
        * @brief Calculate normal vector of a point in height field. 
        * 
        * @details normal = (-dh/dx, 1, -dh/dz).
        * @details Type float should support +/-* operators.
        */
    inline vec3f heightFieldNormal(float x, float z)
    {
        float dhdx = 0, dhdz = 0;
        gradient(x, z, dhdx, dhdz);
        vec3f ret = vec3f(-dhdx, 1.0, -dhdz);
        ret.normalize();
        return ret;
    }

    float* m_data;  // height of land

    int m_nx;
    int m_ny;
    int m_pitch;

    vec3f origin;

    float m_dx;
    float m_dz;
};

}  // namespace Physika