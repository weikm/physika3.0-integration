#include "collision/internal/distance_field3d.hpp"
#include <fstream>
#include "device_launch_parameters.h"
#include "collision/internal/cuda_utilities.hpp"
namespace Physika {

    __host__ __device__ float DistanceToPlane(const vec3f& p, const vec3f& o, const vec3f& n)
    {
        return fabs((p - o, n).length());
    }


    __host__ __device__ float DistanceToSegment(vec3f& pos, vec3f& lo, vec3f& hi)
    {
        vec3f                           seg = hi - lo;
        vec3f                           edge1 = pos - lo;
        vec3f                           edge2 = pos - hi;
        if (edge1.dot(seg) < 0.0f)
        {
            return edge1.length();
        }
        if (edge2.dot(-seg) < 0.0f)
        {
            return edge2.length();
        }
        float length1 = edge1.dot(edge1);
        seg.normalize();
        float length2 = edge1.dot(seg);
        return std::sqrt(length1 - length2 * length2);
    }

    __host__ __device__ float DistanceToSqure(vec3f& pos, vec3f& lo, vec3f& hi, int axis)
    {
        vec3f                           n;
        vec3f                           corner1, corner2, corner3, corner4;
        vec3f                           loCorner, hiCorner, p;
        switch (axis)
        {
        case 0:
            corner1 = vec3f(lo[0], lo[1], lo[2]);
            corner2 = vec3f(lo[0], hi[1], lo[2]);
            corner3 = vec3f(lo[0], hi[1], hi[2]);
            corner4 = vec3f(lo[0], lo[1], hi[2]);
            n = vec3f(1.0, 0.0, 0.0);

            loCorner = vec3f(lo[1], lo[2], 0.0);
            hiCorner = vec3f(hi[1], hi[2], 0.0);
            p = vec3f(pos[1], pos[2], 0.0f);
            break;
        case 1:
            corner1 = vec3f(lo[0], lo[1], lo[2]);
            corner2 = vec3f(lo[0], lo[1], hi[2]);
            corner3 = vec3f(hi[0], lo[1], hi[2]);
            corner4 = vec3f(hi[0], lo[1], lo[2]);
            n = vec3f(0.0f, 1.0f, 0.0f);

            loCorner = vec3f(lo[0], lo[2], 0.0f);
            hiCorner = vec3f(hi[0], hi[2], 0.0f);
            p = vec3f(pos[0], pos[2], 0.0f);
            break;
        case 2:
            corner1 = vec3f(lo[0], lo[1], lo[2]);
            corner2 = vec3f(hi[0], lo[1], lo[2]);
            corner3 = vec3f(hi[0], hi[1], lo[2]);
            corner4 = vec3f(lo[0], hi[1], lo[2]);
            n = vec3f(0.0f, 0.0f, 1.0f);

            loCorner = vec3f(lo[0], lo[1], 0.0);
            hiCorner = vec3f(hi[0], hi[1], 0.0);
            p = vec3f(pos[0], pos[1], 0.0f);
            break;
        }

        float dist1 = DistanceToSegment(pos, corner1, corner2);
        float dist2 = DistanceToSegment(pos, corner2, corner3);
        float dist3 = DistanceToSegment(pos, corner3, corner4);
        float dist4 = DistanceToSegment(pos, corner4, corner1);
        float dist5 = abs(n.dot(pos - corner1));
        if (p[0] < hiCorner[0] && p[0] > loCorner[0] && p[1] < hiCorner[1] && p[1] > loCorner[1])
            return dist5;
        else
            return fminf(fminf(dist1, dist2), fminf(dist3, dist4));
    }

    __host__ __device__ float DistanceToBox(vec3f& pos, vec3f& lo, vec3f& hi)
    {
        vec3f                           corner0(lo[0], lo[1], lo[2]);
        vec3f                           corner1(hi[0], lo[1], lo[2]);
        vec3f                           corner2(hi[0], hi[1], lo[2]);
        vec3f                           corner3(lo[0], hi[1], lo[2]);
        vec3f                           corner4(lo[0], lo[1], hi[2]);
        vec3f                           corner5(hi[0], lo[1], hi[2]);
        vec3f                           corner6(hi[0], hi[1], hi[2]);
        vec3f                           corner7(lo[0], hi[1], hi[2]);
        float                            dist0 = (pos - corner0).length();
        float                            dist1 = (pos - corner1).length();
        float                            dist2 = (pos - corner2).length();
        float                            dist3 = (pos - corner3).length();
        float                            dist4 = (pos - corner4).length();
        float                            dist5 = (pos - corner5).length();
        float                            dist6 = (pos - corner6).length();
        float                            dist7 = (pos - corner7).length();
        if (pos[0] < hi[0] && pos[0] > lo[0] && pos[1] < hi[1] && pos[1] > lo[1] && pos[2] < hi[2] && pos[2] > lo[2])
        {
            float distx = fminf(abs(pos[0] - hi[0]), abs(pos[0] - lo[0]));
            float disty = fminf(abs(pos[1] - hi[1]), abs(pos[1] - lo[1]));
            float distz = fminf(abs(pos[2] - hi[2]), abs(pos[2] - lo[2]));
            float mindist = fminf(distx, disty);
            mindist = fminf(mindist, distz);
            return mindist;
        }
        else
        {
            float distx1 = DistanceToSqure(pos, corner0, corner7, 0);
            float distx2 = DistanceToSqure(pos, corner1, corner6, 0);
            float disty1 = DistanceToSqure(pos, corner0, corner5, 1);
            float disty2 = DistanceToSqure(pos, corner3, corner6, 1);
            float distz1 = DistanceToSqure(pos, corner0, corner2, 2);
            float distz2 = DistanceToSqure(pos, corner4, corner6, 2);
            return -fminf(fminf(fminf(distx1, distx2), fminf(disty1, disty2)), fminf(distz1, distz2));
        }
    }

    __host__ __device__ float DistanceToCylinder(vec3f& pos, vec3f& center, float radius, float height, int axis)
    {
        float distR;
        float distH;
        switch (axis)
        {
        case 0:
            distH = abs(pos[0] - center[0]);
            distR = vec3f(0.0, pos[1] - center[1], pos[2] - center[2]).length();
            break;
        case 1:
            distH = abs(pos[1] - center[1]);
            distR = vec3f(pos[0] - center[0], 0.0, pos[2] - center[2]).length();
            break;
        case 2:
            distH = abs(pos[2] - center[2]);
            distR = vec3f(pos[0] - center[0], pos[1] - center[1], 0.0).length();
            break;
        }

        float halfH = height / 2.0f;
        if (distH <= halfH && distR <= radius)
        {
            return -fminf(halfH - distH, radius - distR);
        }
        else if (distH > halfH && distR <= radius)
        {
            return distH - halfH;
        }
        else if (distH <= halfH && distR > radius)
        {
            return distR - radius;
        }
        else
        {
            float l1 = distR - radius;
            float l2 = distH - halfH;
            return sqrt(l1 * l1 + l2 * l2);
            //return Vector<float, 2>(distR - radius, distH - halfH).length();
        }
    }

    __host__ __device__ float DistanceToSphere(vec3f& pos, vec3f& center, float radius)
    {
        return (pos - center).length() - radius;
    }


    DistanceField3D::DistanceField3D()
    {

    }


    DistanceField3D::DistanceField3D(std::string filename)
    {
        loadSDF(filename);
    }


    void DistanceField3D::setSpace(const Coord p0, const Coord p1, int nbx, int nby, int nbz)
    {
        m_left = p0;

        m_h = (p1 - p0) * Coord(1.0 / float(nbx + 1), 1.0 / float(nby + 1), 1.0 / float(nbz + 1));

        m_distance.Resize(nbx + 1, nby + 1, nbz + 1);
    }


    DistanceField3D::~DistanceField3D()
    {
    }


    void DistanceField3D::translate(const Coord& t)
    {
        m_left += t;
    }


    __global__ void K_Scale(Array3D distance, float s)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        int k = threadIdx.z + (blockIdx.z * blockDim.z);

        if (i >= distance.Nx())
            return;
        if (j >= distance.Ny())
            return;
        if (k >= distance.Nz())
            return;

        distance(i, j, k) = s * distance(i, j, k);
    }


    void DistanceField3D::scale(const Real s)
    {
        m_left[0] *= s;
        m_left[1] *= s;
        m_left[2] *= s;
        m_h[0] *= s;
        m_h[1] *= s;
        m_h[2] *= s;

        dim3 blockSize = make_uint3(8, 8, 8);
        dim3 gridDims  = cudaGridSize3D(make_uint3(m_distance.Nx(), m_distance.Ny(), m_distance.Nz()), blockSize);

        K_Scale<<<gridDims, blockSize>>>(m_distance, s);
    }


    __global__ void K_Invert(Array3D distance)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        int k = threadIdx.z + (blockIdx.z * blockDim.z);

        if (i >= distance.Nx())
            return;
        if (j >= distance.Ny())
            return;
        if (k >= distance.Nz())
            return;

        distance(i, j, k) = -distance(i, j, k);
    }


    void DistanceField3D::invertSDF()
    {
        dim3 blockSize = make_uint3(8, 8, 8);
        dim3 gridDims  = cudaGridSize3D(make_uint3(m_distance.Nx(), m_distance.Ny(), m_distance.Nz()), blockSize);

        K_Invert<<<gridDims, blockSize>>>(m_distance);
    }


    __global__ void K_DistanceFieldToBox(Array3D distance, vec3f start, vec3f h, vec3f lo, vec3f hi, bool inverted)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        int k = threadIdx.z + (blockIdx.z * blockDim.z);

        if (i >= distance.Nx())
            return;
        if (j >= distance.Ny())
            return;
        if (k >= distance.Nz())
            return;

        int   sign = inverted ? 1.0f : -1.0f;
        vec3f p = start + vec3f(i, j, k) * h;

        distance(i, j, k) = sign * DistanceToBox(p, lo, hi);
    }


    void DistanceField3D::loadBox(Coord& lo, Coord& hi, bool inverted)
    {
        m_bInverted = inverted;

        dim3 blockSize = make_uint3(4, 4, 4);
        dim3 gridDims  = cudaGridSize3D(make_uint3(m_distance.Nx(), m_distance.Ny(), m_distance.Nz()), blockSize);

        K_DistanceFieldToBox<<<gridDims, blockSize>>>(m_distance, m_left, m_h, lo, hi, inverted);
    }


    __global__ void K_DistanceFieldToCylinder(Array3D distance, vec3f start, vec3f h, vec3f center, float radius, float height, int axis, bool inverted)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        int k = threadIdx.z + (blockIdx.z * blockDim.z);

        if (i >= distance.Nx())
            return;
        if (j >= distance.Ny())
            return;
        if (k >= distance.Nz())
            return;

        int sign = inverted ? -1.0f : 1.0f;

        vec3f p = start + vec3f(i, j, k) * h;

        distance(i, j, k) = sign * DistanceToCylinder(p, center, radius, height, axis);
    }


    void DistanceField3D::loadCylinder(Coord& center, Real radius, Real height, int axis, bool inverted)
    {
        m_bInverted = inverted;

        dim3 blockSize = make_uint3(8, 8, 8);
        
        dim3 gridDims  = cudaGridSize3D(make_uint3(m_distance.Nx(), m_distance.Ny(), m_distance.Nz()), blockSize);

        K_DistanceFieldToCylinder<<<gridDims, blockSize>>>(m_distance, m_left, m_h, center, radius, height, axis, inverted);
    }

    __global__ void K_DistanceFieldToSphere(Array3D distance, vec3f start, vec3f h, vec3f center, float radius, bool inverted)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        int k = threadIdx.z + (blockIdx.z * blockDim.z);

        if (i >= distance.Nx())
            return;
        if (j >= distance.Ny())
            return;
        if (k >= distance.Nz())
            return;

        int sign = inverted ? -1.0f : 1.0f;

        vec3f p = start + vec3f(i, j, k) * h;

        vec3f dir = p - center;

        distance(i, j, k) = sign * (dir.length() - radius);
    }


    void DistanceField3D::loadSphere(Coord& center, Real radius, bool inverted)
    {
        m_bInverted = inverted;

        dim3 blockSize = make_uint3(8, 8, 8);
        dim3 gridDims  = cudaGridSize3D(make_uint3(m_distance.Nx(), m_distance.Ny(), m_distance.Nz()), blockSize);

        K_DistanceFieldToSphere<<<gridDims, blockSize>>>(m_distance, m_left, m_h, center, radius, inverted);
    }

    void DistanceField3D::loadSDF(std::string filename, bool inverted)
    {
        std::ifstream input(filename.c_str(), std::ios::in);
        if (!input.is_open())
        {
            std::cout << "Reading file " << filename << " error!" << std::endl;
            exit(0);
        }

        int nbx, nby, nbz;
        int xx, yy, zz;

        input >> xx;
        input >> yy;
        input >> zz;

        input >> m_left[0];
        input >> m_left[1];
        input >> m_left[2];

        float t_h;
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
        float* distances = new float[(nbx) * (nby) * (nbz)];
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

        m_distance.Resize(nbx, nby, nbz);
        cuSafeCall(cudaMemcpy(m_distance.GetDataPtr(), distances, (nbx) * (nby) * (nbz) * sizeof(float), cudaMemcpyHostToDevice));

        m_bInverted = inverted;
        if (inverted)
        {
            invertSDF();
        }

        std::cout << "read data successful" << std::endl;
    }


    void DistanceField3D::release()
    {
        m_distance.Release();
    }

}  // namespace Physika