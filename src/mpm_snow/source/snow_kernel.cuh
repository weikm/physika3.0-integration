/**
 * @author      : Yuanmu Xu (xyuan1517@gmail.com)
 * @date        : 2023-06-10
 * @version     : 1.0
 *
 * @file        snow_cuda_kernels.cu
 * @brief       Implementation of CUDA Kernel functions for snow simulation
 *
 * This file contains CUDA kernel functions that are essential for the snow simulation process.
 * It includes a series of GPU-accelerated operations to efficiently simulate various physical
 * behaviors of snow, such as particle-grid interactions, force calculations, collision handling,
 * and deformation updates. These kernels are designed to run on the GPU and are optimized for
 * performance while maintaining the accuracy of the physical simulation.
 *
 *   Key Kernel Functions:
 * - NX, dNX: Basis functions for the material-point method (MPM) grid and particle interactions.
 * - weight, gradWeight: Functions to calculate weight and gradient weight for particle-grid transfer.
 * - initialTransfer: Kernel to transfer initial particle data to the grid.
 * - computeVolumes: Kernel to compute particle volumes based on grid mass distribution.
 * - P2G, G2P: Transfer functions between particles and grid (Particle-to-Grid and Grid-to-Particle).
 * - updateDeformationGradient: Kernel to update deformation gradient of each particle.
 * - applyBoundaryCollisions: Function to handle particle and grid collisions with boundaries.
 * - GridClear: Kernel to reset grid states before each simulation step.
 * - getPositionToDevice, setPointPosition: Kernels to manage positions of the particles.
 *
 * @dependencies: Requires CUDA runtime support. It also depends on external files like
 *                'snow_params.cuh' for simulation parameters and 'helper_math.hpp' for 
 *                mathematical operations.
 *
 * @note        : Understanding of CUDA programming and parallel computing concepts is essential
 *                for modifying these kernels. Any changes should be thoroughly tested to ensure
 *                the accuracy and stability of the simulation.
 *
 * @remark      : These kernels are a critical component of the snow simulation system and have a
 *                significant impact on the overall performance and behavior of the simulation.
 */

#include "cuda_runtime.h"

#include "helper_math.hpp"

#include "snow_params.cuh"
#include "include/decomposition.hpp"

namespace Physika {
__constant__ SolverParam sp;
static const int         blockSize  = 128;
static bool              first_time = true;

__device__ float NX(const float& x)
{
    if (x < 1.0f)
    {
        return 0.5f * (x * x * x) - (x * x) + (2.0f / 3.0f);
    }
    else if (x < 2.0f)
    {
        return (-1.0f / 6.0f) * (x * x * x) + (x * x) - (2.0f * x) + (4.0f / 3.0f);
    }
    else
    {
        return 0.0f;
    }
}

__device__ float dNX(const float& x)
{
    float absx = fabs(x);
    if (absx < 1.0f)
    {
        return (1.5f * absx * x) - (2.0f * x);
    }
    else if (absx < 2.0f)
    {
        return -0.5f * (absx * x) + (2.0f * x) - (2.0f * x / absx);
    }
    else
    {
        return 0.0f;
    }
}

__device__ float weight(const float3& xpgpDiff)
{
    return NX(xpgpDiff.x) * NX(xpgpDiff.y) * NX(xpgpDiff.z);
}

__device__ float3 gradWeight(const float3& xpgpDiff)
{
    return (1.0f / sp.cellSize) * make_float3(dNX(xpgpDiff.x) * NX(fabs(xpgpDiff.y)) * NX(fabs(xpgpDiff.z)), NX(fabs(xpgpDiff.x)) * dNX(xpgpDiff.y) * NX(fabs(xpgpDiff.z)), NX(fabs(xpgpDiff.x)) * NX(fabs(xpgpDiff.y)) * dNX(xpgpDiff.z));
}

__device__ int getGridIndex(const int3& pos)
{
    return (pos.z * sp.gridSize.y * sp.gridSize.x) + (pos.y * sp.gridSize.x) + pos.x;
}

__device__ mat3 calcStress(const mat3& fe, mat3& fp)
{
    float je = mat3::determinant(fe);
    float jp = mat3::determinant(fp);

    float expFactor = expf(sp.hardening * (1 - jp));
    float lambda    = sp.lambda * expFactor;
    float mu        = sp.mu * expFactor;

    mat3 re;
    computePD(fe, re);

    return (2.0f * mu * mat3::multiplyABt(fe - re, fe)) + mat3(lambda * (je - 1) * je);
}

__device__ float getDisplacement(float3 pos, float*& displacement, float3 origin, float h, int x_num, int z_num)
{
    float x = (pos.x - origin.x) / h;
    float z = (pos.z - origin.z) / h;

    int i = floor(x);
    int j = floor(z);

    float fx = x - i;
    float fz = z - j;

    i = clamp(( int )i, ( int )0, x_num - 1);
    j = clamp(( int )j, ( int )0, z_num - 1);

    if (i == x_num - 1)
    {
        i  = x_num - 2;
        fx = 1.0f;
    }

    if (j == z_num - 1)
    {
        j  = z_num - 2;
        fz = 1.0f;
    }

    float d00 = displacement[i + j * x_num];
    float d10 = displacement[i + 1 + j * x_num];
    float d01 = displacement[i + (j + 1) * x_num];
    float d11 = displacement[i + 1 + (j + 1) * x_num];
    //__syncthreads();
    return d00 * (1 - fx) * (1 - fz) + d10 * fx * (1 - fz) + d01 * (1 - fx) * fz + d11 * fx * fz;
}

__device__ void applyBoundaryCollisions(
    const float3& position,
    float3&       velocity)
{
    float  vn;
    float3 vt;
    float3 normal;

    bool collision;

    for (int i = 0; i < 3; i++)
    {
        collision = false;

        if (i == 0)
        {
            //if (position.x <= sp.boxCorner1.x)
            if (position.x <= 0.f)
            {
                collision = true;
                normal    = make_float3(0);
                normal.x  = 1.0f;
            }
            else if (position.x >= sp.boxCorner2.x - sp.boxCorner1.x)
            {
                collision = true;
                normal    = make_float3(0);
                normal.x  = -1.0f;
            }
        }
        if (i == 1)
        {
            //if (position.y <= sp.boxCorner1.y)
            if (position.y <= 0.f)
            {
                collision = true;
                normal    = make_float3(0);
                normal.y  = 1.0f;
            }
            //else if (position.y >= sp.boxCorner2.y)
            else if (position.y >= sp.boxCorner2.y - sp.boxCorner1.y)
            {
                collision = true;
                normal    = make_float3(0);
                normal.y  = -1.0f;
            }
        }
        if (i == 2)
        {
            //if (position.z <= sp.boxCorner1.z)
            if (position.z <= 0.f)
            {
                collision = true;
                normal    = make_float3(0);
                normal.z  = 1.0f;
            }
            //else if (position.z >= sp.boxCorner2.z)
            else if (position.z >= sp.boxCorner2.z - sp.boxCorner1.z)
            {
                collision = true;
                normal    = make_float3(0);
                normal.z  = -1.0f;
            }
        }

        if (collision)
        {
            vn = dot(velocity, normal);

            if (vn >= 0)
            {
                continue;
            }

            if (sp.stickyWalls)
            {
                velocity = make_float3(0);
                return;
            }
            vt = velocity - vn * normal;

            if (length(vt) <= -sp.frictionCoeff * vn)
            {
                velocity = make_float3(0);
                return;
            }
            velocity = vt + sp.frictionCoeff * vn * normalize(vt);
        }
    }
}

__device__ void applyBoundaryCollisions(
    const float3& position, 
    float3& velocity,
    float* height,
    float  unit_height,
    int    height_x_num,
    int    height_z_num)
{
    float  vn;
    float3 vt;
    float3 normal;

    bool collision;

    for (int i = 0; i < 3; i++)
    {
        collision = false;

        if (i == 0)
        {
            //if (position.x <= 0.f)
            if (position.x <= sp.boxCorner1.x)
            {
                collision = true;
                normal    = make_float3(0);
                normal.x  = 1.0f;
            }
            //else if (position.x >= sp.boxCorner2.x - sp.boxCorner1.x)
            else if (position.x >= sp.boxCorner2.x)
            {
                collision = true;
                normal    = make_float3(0);
                normal.x  = -1.0f;
            }
        }
        if (i == 1)
        {
            //if (position.y <= getDisplacement(position, height, { 0.f, 0.f, 0.f }, unit_height, height_x_num, height_z_num) - sp.boxCorner1.y || position.y <= 0.f)
            if (position.y <= getDisplacement(position, height, sp.boxCorner1, unit_height, height_x_num, height_z_num) || position.y <= sp.boxCorner1.y)
            {
                collision = true;
                normal    = make_float3(0);
                normal.x = -getDisplacement(make_float3(position.x + unit_height, position.y, position.z), height, sp.boxCorner1, unit_height, height_x_num, height_z_num) - getDisplacement(make_float3(position.x - unit_height, position.y, position.z), height, sp.boxCorner1, unit_height, height_x_num, height_z_num);
                normal.y = 2.0f;
                normal.z = -getDisplacement(make_float3(position.x, position.y, position.z + unit_height), height, sp.boxCorner1, unit_height, height_x_num, height_z_num) - getDisplacement(make_float3(position.x, position.y, position.z - unit_height), height, sp.boxCorner1, unit_height, height_x_num, height_z_num);

                normal = normalize(normal);
            }
            //else if (position.y >= sp.boxCorner2.y - sp.boxCorner1.y)
            else if (position.y >= sp.boxCorner2.y)
            {
                collision = true;
                normal    = make_float3(0);
                normal.y  = -1.0f;
            }
        }
        if (i == 2)
        {
            //if (position.z <= 0.f)
            if (position.z <= sp.boxCorner1.z)
            {
                collision = true;
                normal    = make_float3(0);
                normal.z  = 1.0f;
            }
            //else if (position.z >= sp.boxCorner2.z - sp.boxCorner1.z)
            else if (position.z >= sp.boxCorner2.z)
            {
                collision = true;
                normal    = make_float3(0);
                normal.z  = -1.0f;
            }
        }

        if (collision)
        {
            vn = dot(velocity, normal);

            if (vn >= 0)
            {
                continue;
            }

            if (sp.stickyWalls)
            {
                velocity = make_float3(0);
                return;
            }
            vt = velocity - vn * normal;

            if (length(vt) <= -sp.frictionCoeff * vn)
            {
                velocity = make_float3(0);
                return;
            }
            velocity = vt + sp.frictionCoeff * vn * normalize(vt);
        }
    }
}

__device__ mat3 getVelocityGradient(Point& particle, Grid* cells)
{
    float hInv = 1.0f / sp.cellSize;
    float3 local_pos = particle.m_position;
    int3  pos  = make_int3(local_pos * hInv);
    mat3  velocityGrad(0.0f);

    for (int z = -2; z < 3; z++)
    {
        for (int y = -2; y < 3; y++)
        {
            for (int x = -2; x < 3; x++)
            {
                int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
                if (n.x >= 0 && n.x < sp.gridSize.x && n.y >= 0 && n.y < sp.gridSize.y && n.z >= 0 && n.z < sp.gridSize.z)
                {
                    float3 diff   = (local_pos- (make_float3(n) * sp.cellSize)) * hInv;
                    float3 gw     = gradWeight(diff);
                    int    gIndex = getGridIndex(n);

                    velocityGrad += mat3::outerProduct(cells[gIndex].velocity_star, gw);
                }
            }
        }
    }

    return velocityGrad;
}

__global__ void initialTransfer(Point* particles, Grid* cells, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;

    float hInv = 1.0f / sp.cellSize;
    float3 local_pos = particles[index].m_position;
    int3   pos       = make_int3(local_pos * hInv);

    for (int z = -2; z < 3; z++)
    {
        for (int y = -2; y < 3; y++)
        {
            for (int x = -2; x < 3; x++)
            {
                int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);

                if (n.x >= 0 && n.x < sp.gridSize.x && n.y >= 0 && n.y < sp.gridSize.y && n.z >= 0 && n.z < sp.gridSize.z)
                {
                    float3 diff   = (local_pos - (make_float3(n) * sp.cellSize)) * hInv;
                    int    gIndex = getGridIndex(n);

                    float mi = particles[index].m_mass * weight(fabs(diff));
                    atomicAdd(&(cells[gIndex].mass), mi);
                }
            }
        }
    }
}

__global__ void computeVolumes(Point* particles, Grid* cells, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;

    float hInv          = 1.0f / sp.cellSize;
    float3 local_pos     = particles[index].m_position;
    int3   pos           = make_int3(local_pos * hInv);
    float pDensity      = 0.0f;
    float invCellVolume = hInv * hInv * hInv;
    for (int z = -2; z < 3; z++)
    {
        for (int y = -2; y < 3; y++)
        {
            for (int x = -2; x < 3; x++)
            {
                int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);

                if (n.x >= 0 && n.x < sp.gridSize.x && n.y >= 0 && n.y < sp.gridSize.y && n.z >= 0 && n.z < sp.gridSize.z)
                {
                    float3 diff   = (local_pos - (make_float3(n) * sp.cellSize)) * hInv;
                    int    gIndex = getGridIndex(n);
                    pDensity += cells[gIndex].mass * invCellVolume * weight(fabs(diff));
                }
            }
        }
    }
    particles[index].m_volume = particles[index].m_mass / pDensity;
}

__global__ void world2local(Point* particles, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;
    particles[index].m_position = particles[index].m_position - sp.boxCorner1;
}

__global__ void local2world(Point* particles, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;
    particles[index].m_position = particles[index].m_position + sp.boxCorner1;
}


__global__ void P2G(Point* particles, Grid* cells, float3* extern_force, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;
    mat3 volumeStress = -particles[index].m_volume * calcStress(particles[index].fe, particles[index].fp);

    float hInv = 1.0f / sp.cellSize;
    float3 local_pos = particles[index].m_position;
    int3   pos       = make_int3(local_pos * hInv);
    for (int z = -1; z < 2; z++)
    {
        for (int y = -1; y < 2; y++)
        {
            for (int x = -1; x < 2; x++)
            {
                int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
                if (n.x >= 0 && n.x < sp.gridSize.x && n.y >= 0 && n.y < sp.gridSize.y && n.z >= 0 && n.z < sp.gridSize.z)
                {
                    float3 diff   = (local_pos -(make_float3(n) * sp.cellSize)) * hInv;
                    float3 gw     = gradWeight(diff);
                    int    gIndex = getGridIndex(n);
                    float3 force  = volumeStress * gw + extern_force[index] * gw;

                    float mi = particles[index].m_mass * weight(fabs(diff));
                    atomicAdd(&(cells[gIndex].mass), mi);
                    atomicAdd(&(cells[gIndex].velocity.x), particles[index].m_velocity.x * mi);
                    atomicAdd(&(cells[gIndex].velocity.y), particles[index].m_velocity.y * mi);
                    atomicAdd(&(cells[gIndex].velocity.z), particles[index].m_velocity.z * mi);
                    atomicAdd(&(cells[gIndex].force.x), force.x);
                    atomicAdd(&(cells[gIndex].force.y), force.y);
                    atomicAdd(&(cells[gIndex].force.z), force.z);
                }
            }
        }
    }
}

__global__ void UpdateVelocityGrid(Grid* cells, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < sp.gridSize.x && y < sp.gridSize.y && z < sp.gridSize.z)
    {
        int index = z * sp.gridSize.z * sp.gridSize.y + y * sp.gridSize.x + x;
        if (cells[index].mass > 0.f)
        {
            float invMass = 1.f / cells[index].mass;
            cells[index].force += cells[index].mass * sp.gravity;
            cells[index].velocity *= invMass;
            cells[index].velocity_star = cells[index].velocity + sp.dt * invMass * cells[index].force;
        }
    }
}

__global__ void bodyCollisions(
    Grid* cells, 
    int size,
    float* height,
    float  unit_height,
    int    height_x_num,
    int    height_z_num)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < sp.gridSize.x && y < sp.gridSize.y && z < sp.gridSize.z)
    {
        int    index = z * sp.gridSize.z * sp.gridSize.y + y * sp.gridSize.x + x;
        float  z     = index / (int(sp.gridSize.y) * int(sp.gridSize.x));
        float  y     = index % (int(sp.gridSize.y) * int(sp.gridSize.x)) / (int(sp.gridSize.x));
        float  x     = index % int(sp.gridSize.x);
        float3 pos   = make_float3(x, y, z) * sp.cellSize;
        applyBoundaryCollisions(pos + sp.dt * cells[index].velocity_star, cells[index].velocity_star);
        //applyBoundaryCollisions(pos + sp.dt * cells[index].velocity_star, cells[index].velocity_star, height, unit_height, height_x_num, height_z_num);
    }
}

__global__ void updateDeformationGradient(Point* particles, Grid* cells, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;

    mat3 velocityGrad = getVelocityGradient(particles[index], cells);
    mat3 newFe        = (mat3(1.0f) + (sp.dt * velocityGrad)) * particles[index].fe;
    mat3 newF         = newFe * particles[index].fp;

    mat3 U, S, V, Sinv;
    computeSVD(newFe, U, S, V);

    S    = mat3(clamp(S[0], 1 - sp.compression, 1 + sp.stretch), 0.0f, 0.0f, 0.0f, clamp(S[4], 1 - sp.compression, 1 + sp.stretch), 0.0f, 0.0f, 0.0f, clamp(S[8], 1 - sp.compression, 1 + sp.stretch));
    Sinv = mat3(1.0f / S[0], 0.0f, 0.0f, 0.0f, 1.0f / S[4], 0.0f, 0.0f, 0.0f, 1.0f / S[8]);

    particles[index].fe = mat3::multiplyADBt(U, S, V);
    particles[index].fp = mat3::multiplyADBt(V, Sinv, U) * newF;
}

__global__ void G2P(Point* particles, Grid* cells, int size)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;
    float hInv = 1.0f / sp.cellSize;
    int3  pos  = make_int3(particles[index].m_position * hInv);

    float3 velocityPic  = make_float3(0.0f);
    float3 velocityFlip = particles[index].m_velocity;
    for (int z = -1; z < 2; z++)
    {
        for (int y = -1; y < 2; y++)
        {
            for (int x = -1; x < 2; x++)
            {
                int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
                if (n.x >= 0 && n.x < sp.gridSize.x && n.y >= 0 && n.y < sp.gridSize.y && n.z >= 0 && n.z < sp.gridSize.z)
                {
                    float3 diff   = (particles[index].m_position - (make_float3(n) * sp.cellSize)) * hInv;
                    int    gIndex = getGridIndex(n);

                    float w = weight(fabs(diff));

                    velocityPic += cells[gIndex].velocity_star * w;
                    velocityFlip += (cells[gIndex].velocity_star - cells[gIndex].velocity) * w;
                }
            }
        }
    }
    particles[index].m_velocity = ((1 - sp.alpha) * velocityPic) + (sp.alpha * velocityFlip);
}

__global__ void particleBodyCollisions(
    Point* particles, 
    int size,
    float* height,
    float  unit_height,
    int    height_x_num,
    int    height_z_num)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= size)
        return;

    applyBoundaryCollisions(particles[index].m_position + sp.dt * particles[index].m_velocity, particles[index].m_velocity, height, unit_height, height_x_num, height_z_num);

    particles[index].m_position += sp.dt * particles[index].m_velocity;
}

__global__ void GridClear(Grid* cells, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < sp.gridSize.x && y < sp.gridSize.y && z < sp.gridSize.z)
    {
        int index                  = z * sp.gridSize.z * sp.gridSize.y + y * sp.gridSize.x + x;
        cells[index].mass          = 0.0f;
        cells[index].velocity      = make_float3(0.0f, 0.0f, 0.0f);
        cells[index].velocity_star = make_float3(0.0f, 0.0f, 0.0f);
        cells[index].force         = make_float3(0, 0, 0);
    }
}

__global__ void getPositionToDevice(float3* position, Point* dev_points, unsigned int num_particles)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= num_particles)
        return;
    position[index] = dev_points[index].m_position;
}

// get velocity from device
__global__ void getVelocityToDevice(float3* velocity, Point* dev_points, unsigned int num_particles)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= num_particles)
		return;
	velocity[index] = dev_points[index].m_velocity;
}

//get velocity and position from device
__global__ void getVelocityPositionToDevice(float3* position, float3* velocity, Point* dev_points, unsigned int num_particles)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= num_particles)
		return;
	velocity[index] = dev_points[index].m_velocity;
	position[index] = dev_points[index].m_position;
}

__global__ void setPointPosition(float3* position, Point* dev_points, unsigned int num_particles)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= num_particles)
        return;
    dev_points[index].m_position = position[index];
}

}  // namespace Physika