#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

#include "sdf.h"

void DistanceField3D::loadSDF(std::string filename)
{
    std::ifstream input(filename.c_str(), std::ios::in);
    if (!input.is_open())
    {
        std::cout << "Reading file " << filename << " error!" << std::endl;
        return;
    }

    int nbx, nby, nbz;
    int xx, yy, zz;

    input >> xx;
    input >> yy;
    input >> zz;

    input >> m_left[0];
    input >> m_left[1];
    input >> m_left[2];

    REAL t_h;
    input >> t_h;

    std::cout << "SDF: " << xx << ", " << yy << ", " << zz << std::endl;
    std::cout << "SDF: " << m_left[0] << ", " << m_left[1] << ", " << m_left[2] << std::endl;
    std::cout << "SDF: " << m_left[0] + t_h * xx << ", " << m_left[1] + t_h * yy << ", " << m_left[2] + t_h * zz << std::endl;

    nbx = xx;
    nby = yy;
    nbz = zz;
    m_h[0] = t_h;
    m_h[1] = t_h;
    m_h[2] = t_h;

    int   idd = 0;
    REAL* distances = new REAL[(nbx) * (nby) * (nbz)];
    for (int k = 0; k < zz; k++)
    {
        for (int j = 0; j < yy; j++)
        {
            for (int i = 0; i < xx; i++)
            {
                float dist;
                input >> dist;
                distances[i + nbx * (j + nby * k)] = dist;
            }
        }
    }
    input.close();

    m_nbx = nbx;
    m_nby = nby;
    m_nbz = nbz;
    m_distance = distances;
    std::cout << "read data successful" << std::endl;
}

void DistanceField3D::getDistance(const vec3f& p, REAL& d, vec3f& normal)
{
    // get cell and lerp values
    vec3f     fp = (p - m_left) * vec3f(1.0 / m_h[0], 1.0 / m_h[1], 1.0 / m_h[2]);
    const int i = (int)floorf(fp[0]);
    const int j = (int)floorf(fp[1]);
    const int k = (int)floorf(fp[2]);

    if (i < 0 || i >= m_nbx - 1 || j < 0 || j >= m_nby - 1 || k < 0 || k >= m_nbz - 1)
    {
        d = 100000.0f;
        normal = vec3f(0, 0, 0);
        return;
    }
    vec3f ip = vec3f(i, j, k);

    vec3f alphav = fp - ip;
    REAL  alpha = alphav[0];
    REAL  beta = alphav[1];
    REAL  gamma = alphav[2];

    REAL d000 = getAt(i, j, k);
    REAL d100 = getAt(i + 1, j, k);
    REAL d010 = getAt(i, j + 1, k);
    REAL d110 = getAt(i + 1, j + 1, k);
    REAL d001 = getAt(i, j, k + 1);
    REAL d101 = getAt(i + 1, j, k + 1);
    REAL d011 = getAt(i, j + 1, k + 1);
    REAL d111 = getAt(i + 1, j + 1, k + 1);

    REAL dx00 = lerp(d000, d100, alpha);
    REAL dx10 = lerp(d010, d110, alpha);
    REAL dxy0 = lerp(dx00, dx10, beta);

    REAL dx01 = lerp(d001, d101, alpha);
    REAL dx11 = lerp(d011, d111, alpha);
    REAL dxy1 = lerp(dx01, dx11, beta);

    REAL d0y0 = lerp(d000, d010, beta);
    REAL d0y1 = lerp(d001, d011, beta);
    REAL d0yz = lerp(d0y0, d0y1, gamma);

    REAL d1y0 = lerp(d100, d110, beta);
    REAL d1y1 = lerp(d101, d111, beta);
    REAL d1yz = lerp(d1y0, d1y1, gamma);

    REAL dx0z = lerp(dx00, dx01, gamma);
    REAL dx1z = lerp(dx10, dx11, gamma);

    normal[0] = d0yz - d1yz;
    normal[1] = dx0z - dx1z;
    normal[2] = dxy0 - dxy1;

    REAL l = normal.length2();
    if (l < 0.0001f)
        normal = vec3f(0, 0, 0);
    else
        normal.normalize();

    d = (1.0f - gamma) * dxy0 + gamma * dxy1;
}
