/**
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-17
 * @description: CUDA Kernel functions of solving granular particles
 * @version    : 1.0
 * 
 * @author     :Haikai Zeng (haok_z@126.com)
 * @date       :2023-11-18
 * @description:Multi-scale particle coupling is achieved.
 *
 */
#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>
#include <math_constants.h>

#include "granular_params.hpp"
#include "granular_kernel.cuh"
#include "helper_math.hpp"

using namespace cooperative_groups;
namespace Physika {

__constant__ GranularSimulateParams g_params_G;

__host__ void setSimulationParams(GranularSimulateParams* hostParam)
{
    cudaMemcpyToSymbol(g_params_G, hostParam, sizeof(GranularSimulateParams));
}
/**
 * @brief calculate the poly6 kernel value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel value
 */
__device__ float wPoly6Granular(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > g_params_G .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = g_params_G .m_sph_radius_squared - lengthSquared;
    return g_params_G .m_poly6_coff * iterm * iterm * iterm;
}

/**
 * @brief calculate the poly6 kernel gradient value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel gradient value
 */
__device__
    float3
    wSpikyGradGranular(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3      ret           = { 0.0f, 0.0f, 0.0f };
    if (lengthSquared > g_params_G .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float       iterm  = g_params_G .m_sph_radius - length;
    float       coff   = g_params_G .m_spiky_grad_coff * iterm * iterm / length;
    ret.x              = coff * r.x;
    ret.y              = coff * r.y;
    ret.z              = coff * r.z;
    return ret;
}
/**
 * @brief calculate which grid the particle belongs to
 * @param[in] p the position of the particle
 * @return the grid position
 */
__device__
    int3
    calcGridPosKernelGranular(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - g_params_G .m_world_origin.x) / g_params_G .m_cell_size.x);
    gridPos.y = floor((p.y - g_params_G .m_world_origin.y) / g_params_G .m_cell_size.y);
    gridPos.z = floor((p.z - g_params_G .m_world_origin.z) / g_params_G .m_cell_size.z);
    return gridPos;
}

/**
 * @brief calculate hash of grid positions (clamp them within cell boundary)
 * @param[in] gridPos the grid position
 * @return the hash value of given grid
 */
__device__ unsigned int calcGridHashKernelGranular(int3 gridPos)
{
    gridPos.x = gridPos.x & (g_params_G .m_grid_size.x - 1);
    gridPos.y = gridPos.y & (g_params_G .m_grid_size.y - 1);
    gridPos.z = gridPos.z & (g_params_G .m_grid_size.z - 1);
    return gridPos.z * g_params_G .m_grid_size.x * g_params_G .m_grid_size.y + gridPos.y * g_params_G .m_grid_size.x + gridPos.x;
}

__global__ void calcParticlesHashKernelGranular(
    unsigned int* gridParticleHash,
    float4*       pos,
    unsigned int  numParticles)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    volatile float4 curPos    = pos[index];
    int3            gridPos   = calcGridPosKernelGranular(make_float3(curPos.x, curPos.y, curPos.z));
    unsigned int    hashValue = calcGridHashKernelGranular(gridPos);
    gridParticleHash[index]   = hashValue;
}

__global__ void findCellRangeKernelGranular(
    unsigned int* cellStart,         // output: cell start index
    unsigned int* cellEnd,           // output: cell end index
    unsigned int* gridParticleHash,  // input: sorted grid hashes
    unsigned int  numParticles)
{
    thread_block                   cta = this_thread_block();
    extern __shared__ unsigned int sharedHash[];
    unsigned int                   index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int                   hashValue;

    if (index < numParticles)
    {
        hashValue                   = gridParticleHash[index];
        sharedHash[threadIdx.x + 1] = hashValue;

        // first thread in block must load neighbor particle hash
        if (index > 0 && threadIdx.x == 0)
            sharedHash[0] = gridParticleHash[index - 1];
    }

    sync(cta);

    if (index < numParticles)
    {
        if (index == 0 || hashValue != sharedHash[threadIdx.x])
        {
            cellStart[hashValue] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
            cellEnd[hashValue] = index + 1;
    }
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
    //if (d00 < -21.0f || d00 > -19.9f)
    //{
    //    printf("x:%f, z:%f, d00:%f, index:%d\n", x, z, d00, i + j * x_num);
    //    d00 = -20.0f;        
    //}
    //return d00;
    
}


__global__ void granularAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    float*       particlePhase,
    float*       height,
    float        unit_height,
    int          height_x_num,
    int          height_z_num,
    float        deltaTime,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    float4 readVel = velocity[index];
    float3 nVel    = make_float3(readVel);
    float3 nPos    = make_float3(position[index]);

    // float parRad = position[index].w;
    // float invMass;
    // invMass = 0.027 / (parRad * parRad * parRad);

    float invMass = position[index].w;
    float parRad;
    parRad = 0.3 / cbrt(invMass);

    nVel += deltaTime * g_params_G .m_gravity;
    nVel += deltaTime * collisionForce[index];
    bool contact = 0;

    float3 contactNormal = make_float3(0.f);
    // collision with walls.
    if (nPos.x > g_params_G .m_world_box_corner2.x - g_params_G .m_particle_radius)
    {
        contact       = 1;
        contactNormal = { -1.f, 0.f, 0.f };
        if (length(nVel) > 0.2)
            nPos.x        = g_params_G .m_world_box_corner2.x - g_params_G .m_particle_radius;
    }
    if (nPos.x < g_params_G .m_world_box_corner1.x + g_params_G .m_particle_radius)
    {
        contact       = 1;
        contactNormal = { 1.f, 0.f, 0.f };
        if (length(nVel) > 0.2)
            nPos.x        = g_params_G .m_world_box_corner1.x + g_params_G .m_particle_radius;
    }
    if (nPos.y > g_params_G .m_world_box_corner2.y - g_params_G .m_particle_radius)
        nPos.y = g_params_G .m_world_box_corner2.y - g_params_G .m_particle_radius;


    //height field
    float height_xz = getDisplacement(nPos, height, g_params_G.m_world_box_corner1, unit_height, height_x_num, height_z_num);
      
    if (nPos.y < height_xz + g_params_G.m_particle_radius)
    //if (nPos.y < g_params_G .m_world_box_corner1.y + g_params_G .m_particle_radius)
    {
        contact       = 2;
        contactNormal = { 0.f, 1.f, 0.f };
        if (length(nVel) > 0.2)
            nPos.y = height_xz + g_params_G.m_particle_radius;
            //nPos.y = g_params_G.m_world_box_corner1.y + g_params_G.m_particle_radius;
        nVel = make_float3(0.f);
    }



    if (nPos.z > g_params_G .m_world_box_corner2.z - g_params_G .m_particle_radius)
    {
        contact       = 1;
        contactNormal = { 0.f, 0.f, -1.f };
        if (length(nVel) > 0.2)
            nPos.z        = g_params_G .m_world_box_corner2.z - g_params_G .m_particle_radius;
        
    }
    if (nPos.z < g_params_G .m_world_box_corner1.z + g_params_G .m_particle_radius)
    {
        contact       = 1;
        contactNormal = { 0.f, 0.f, 1.f };
        if (length(nVel) > 0.2)
            nPos.z        = g_params_G .m_world_box_corner1.z + g_params_G .m_particle_radius;
    }

    if (contact == 1)
    {
        float3 normVel                 = dot(nVel, contactNormal) * contactNormal;
        float3 tangVel                 = nVel - normVel;
        //float  staticFrictionThreshold = 0.f * length(normVel);
        //if (length(tangVel) < staticFrictionThreshold)
        //{
        //    tangVel = make_float3(0.f);
        //}
        //else
        //{
        //    tangVel = tangVel * (1.0f - 0.01f);
        //}
        nVel = normVel * g_params_G .m_damp + tangVel;
    }
    if (contact == 2)
    {
        float3 normVel                 = dot(nVel, contactNormal) * contactNormal;
        float3 tangVel                 = nVel - normVel;
        float  staticFrictionThreshold = g_params_G .m_static_fricton_coeff * length(normVel);
        if (length(tangVel) < staticFrictionThreshold)
        {
            tangVel = make_float3(0.f);
        }
        else
        {
            tangVel = tangVel * (1.0f - g_params_G .m_dynamic_fricton_coeff);
        }
        nVel = normVel * g_params_G .m_damp + tangVel;
    }

    nPos += deltaTime * nVel;

    // float stakeHeight   = nPos.y - g_params_G .m_world_origin.y;
    // float invMass       = 1.f / expf(g_params_G .m_stack_height_coeff * stakeHeight);
    predictedPos[index] = make_float4(nPos, invMass);
    // if (length(nPos - oldPos) < 0.002f / cbrt(invMass))
    //     predictedPos[index] = make_float4(oldPos, invMass);
}

__global__ void addDeltaPositionGranular(
    float4*      predictedPos,
    float3*      deltaPos,
    float*       particlePhase,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    float3 readPos = make_float3(predictedPos[index]);
    float  readW   = predictedPos[index].w;

    readPos += deltaPos[index];

    predictedPos[index] = { readPos.x, readPos.y, readPos.z, readW };
}

__global__ void distanceConstrainGranluar(
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    float4 readPos     = predictedPos[index];
    float3 curPos      = make_float3(readPos);
    int3   gridPos     = calcGridPosKernelGranular(curPos);
    float3 deltaP      = make_float3(0.f);
    float3 frictdeltaP = make_float3(0.f);
    // float  parRad1     = readPos.w;
    // float  invMass1;
    // invMass1 = 0.027 / (parRad1 * parRad1 * parRad1);

    float invMass1 = readPos.w;
    float parRad1;
    parRad1 = 0.3 / cbrt(invMass1);


#pragma unroll 3
    for (int z = -1; z <= 1; ++z)
    {
#pragma unroll 3
        for (int y = -1; y <= 1; ++y)
        {
#pragma unroll 3
            for (int x = -1; x <= 1; ++x)
            {
                int3         neighbourGridPos   = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
                unsigned int neighbourGridIndex = calcGridHashKernelGranular(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    if (i == index)
                        continue;
                    float3 deltaP2       = make_float3(0.f);
                    float4 readNeighbour = predictedPos[i];
                    float3 neighbour     = make_float3(readNeighbour);
                    // float  parRad2       = readNeighbour.w;
                    // float  invMass2;
                    // invMass2 = 0.027 / (parRad2 * parRad2 * parRad2);

                    float invMass2 = readNeighbour.w;
                    float parRad2;
                    parRad2 = 0.3 / cbrt(invMass2);
                    
                    
                    
                    float  wSum          = invMass1 + invMass2;
                    float  weight1       = invMass1 / wSum;
                    float3 r             = curPos - neighbour;
                    float  len           = length(r);
                    r /= len;
                    if (len < parRad1 + parRad2)
                    {
                        float3 corr = g_params_G.m_stiffness * r * (parRad1 + parRad2 - len) / wSum;
                        deltaP += invMass1 * corr;
                        deltaP2 -= invMass2 * corr;
                    }
                    float3 relativedeltaP = deltaP - deltaP2;
                    frictdeltaP           = relativedeltaP - dot(relativedeltaP, r) * r;
                    float d_frictdeltaP   = length(frictdeltaP);
                    if (d_frictdeltaP < g_params_G.m_static_fricton_coeff * (parRad1 + parRad2))
                    {
                        frictdeltaP *= weight1 * g_params_G .m_stiffness;
                    }
                    else
                    {
                        frictdeltaP *= weight1 * min(1.f, g_params_G.m_dynamic_fricton_coeff * (parRad1 + parRad2) / d_frictdeltaP) * g_params_G.m_stiffness;
                    }
                }
            }
        }
    }
    deltaPos[index] = deltaP + frictdeltaP;
}

__global__ void updateVelocityAndPositionGranular(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        invDeltaTime,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    float4 oldPos  = position[index];
    float4 newPos  = predictedPos[index];
    float4 readVel = velocity[index];
    float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
    posDiff *= invDeltaTime;
    velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
    if (length(newPos - oldPos) > g_params_G .m_sleep_threshold / cbrt(newPos.w))
        position[index] = { newPos.x, newPos.y, newPos.z, newPos.w };
}
__global__ void solverCollisionConstrainGranular(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    float*        particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numCollisionParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    uint   particle_id  = collision_particle_id[index];
    float3 oldpos      = make_float3(position[particle_id]);
    float3 prePos       = make_float3(predictedPos[particle_id]);
    
    float3 deltaP = moveDirection[index] * moveDistance[index];
    prePos += deltaP;
    predictedPos[particle_id] = { prePos.x, prePos.y, prePos.z, predictedPos[particle_id].w };
    position[particle_id] += { deltaP.x, deltaP.y, deltaP.z, 0.f };
}

}  // namespace Physika