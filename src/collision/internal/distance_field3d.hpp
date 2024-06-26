/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-19
 * @description: 3d distance field in Physika.
 * @version    : 1.0
 */
#pragma once

#include <string>
#include <cuda_runtime.h>
#include "collision/internal/collision_array3d.hpp"
#include "collision/internal/collision_vec3.hpp"
namespace Physika {

    class DistanceField3D
    {
    public:
        typedef float Real;
        typedef vec3f Coord;
        DistanceField3D();

        DistanceField3D(std::string filename);

        /*!
            *    \brief    Should not release data here, call release() explicitly.
            */
        ~DistanceField3D();

        /**
             * @brief Release m_distance
             * Should be explicitly called before destruction to avoid GPU memory leak.
             */
        void release();

        /**
             * @brief Translate the distance field with a displacement
             *
             * @param t displacement
             */
        void translate(const Coord& t);

        /**
             * @brief Scale the distance field
             *
             * @param s scaling factor
             */
        void scale(const Real s);

        /**
             * @brief Query the signed distance for p
             *
             * @param p position
             * @param d return the signed distance at position p
             * @param normal return the normal at position p
             */
        COMM_FUNC void getDistance(const Coord& p, Real& d, Coord& normal)
        {
            // get cell and lerp values
            Coord     fp = (p - m_left) * Coord(1.0 / m_h[0], 1.0 / m_h[1], 1.0 / m_h[2]);
            const int i  = ( int )floorf(fp[0]);
            const int j  = ( int )floorf(fp[1]);
            const int k  = ( int )floorf(fp[2]);
            if (i < 0 || i >= m_distance.Nx() - 1 || j < 0 || j >= m_distance.Ny() - 1 || k < 0 || k >= m_distance.Nz() - 1)
            {
                if (m_bInverted)
                    d = -100000.0f;
                else
                    d = 100000.0f;
                normal = Coord(0, 0, 0);
                return;
            }
            Coord ip = Coord(i, j, k);

            Coord alphav = fp - ip;
            Real  alpha  = alphav[0];
            Real  beta   = alphav[1];
            Real  gamma  = alphav[2];

            Real d000 = m_distance(i, j, k);
            Real d100 = m_distance(i + 1, j, k);
            Real d010 = m_distance(i, j + 1, k);
            Real d110 = m_distance(i + 1, j + 1, k);
            Real d001 = m_distance(i, j, k + 1);
            Real d101 = m_distance(i + 1, j, k + 1);
            Real d011 = m_distance(i, j + 1, k + 1);
            Real d111 = m_distance(i + 1, j + 1, k + 1);

            Real dx00 = lerp(d000, d100, alpha);
            Real dx10 = lerp(d010, d110, alpha);
            Real dxy0 = lerp(dx00, dx10, beta);

            Real dx01 = lerp(d001, d101, alpha);
            Real dx11 = lerp(d011, d111, alpha);
            Real dxy1 = lerp(dx01, dx11, beta);

            Real d0y0 = lerp(d000, d010, beta);
            Real d0y1 = lerp(d001, d011, beta);
            Real d0yz = lerp(d0y0, d0y1, gamma);

            Real d1y0 = lerp(d100, d110, beta);
            Real d1y1 = lerp(d101, d111, beta);
            Real d1yz = lerp(d1y0, d1y1, gamma);

            Real dx0z = lerp(dx00, dx01, gamma);
            Real dx1z = lerp(dx10, dx11, gamma);

            normal[0] = d0yz - d1yz;
            normal[1] = dx0z - dx1z;
            normal[2] = dxy0 - dxy1;

            Real l = normal.length();
            if (l < 0.0001f)
                normal = Coord(0, 0, 0);
            else
                normal = normal.normalize();

            d = (1.0f - gamma) * dxy0 + gamma * dxy1;
        }

    public:
        /**
        * @brief load signed distance field from a file
        *
        * @param filename
        * @param inverted indicated whether the signed distance field should be inverted after initialization
        */
        void loadSDF(std::string filename, bool inverted = false);

        /**
        * @brief load Box signed distance field
        *
        * @param center    the center of the box
        * @param size      the size of the box
        * @param inverted  indicated whether the signed distance field should be inverted after initialization
        */
        void loadBox(Coord& lo, Coord& hi, bool inverted = false);

        /**
        * @brief load Cylinder signed distance field
        *
        * @param center    the center of the cylinder
        * @param radius    the radius of the cylinder
        * @param height    the height of the cylinder
        * @param axis      the axis of the cylinder 0: x-axis 1: y-axis 2: z-axis
        * @param inverted  indicated whether the signed distance field should be inverted after initialization
        */
        void loadCylinder(Coord& center, Real radius, Real height, int axis, bool inverted = false);

        /**
        * @brief load Sphere signed distance field
        *
        * @param center    the center of the sphere
        * @param radius    the radius of the sphere
        * @param inverted  indicated whether the signed distance field should be inverted after initialization
        */
        void loadSphere(Coord& center, Real radius, bool inverted = false);

        void setSpace(const Coord p0, const Coord p1, int nbx, int nby, int nbz);

    private:
        COMM_FUNC inline Real lerp(Real a, Real b, Real alpha) const
        {
            return (1.0f - alpha) * a + alpha * b;
        }

        /**
             * @brief Invert the signed distance field
             *
             */
        void invertSDF();

        /**
             * @brief Lower left corner
             *
             */
        vec3f m_left;

        /**
             * @brief grid spacing
             *
             */
        vec3f m_h;

        bool m_bInverted = false;

        /**
             * @brief Storing the signed distance field as a 3D array.
             *
             */
        Array3D m_distance;
    };

}  // namespace PhysIKA
