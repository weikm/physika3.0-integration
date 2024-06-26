#pragma once
#include "vec3f.h"
#include "box.h"

class DistanceField3D
{
public:
    DistanceField3D(std::string filename) {
        loadSDF(filename);
    }
    
    ~DistanceField3D()
    {
        if (m_distance)
            delete[] m_distance;
    }

    void getDistance(const vec3f & p, REAL & d, vec3f& normal);

public:
    void loadSDF(std::string filename);

private:
    inline REAL lerp(REAL a, REAL b, REAL alpha) const
    {
        return (1.0f - alpha) * a + alpha * b;
    }

    inline REAL getAt(int i, int j, int k) {
        return m_distance[i + m_nbx * (j + m_nby * k)];
    }

    // Lower left corner
    vec3f m_left;

    // grid spacing
    vec3f m_h;

    // Storing the signed distance field as a 3D array.
    int m_nbx, m_nby, m_nbz;
    REAL *m_distance;
};
