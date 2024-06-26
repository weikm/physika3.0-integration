/**
 * @author     : Yuege Xiong (candybear0714@163.com)
 * @date       : 2023-11-22
 * @description: CUDA Kernel functions of solving granular particles
 * @version    : 1.0
 */
#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>
#include <math_constants.h>

#include "fluid_params.hpp"
#include "fluid_kernel.cuh"

#include "helper_math.hpp"

using namespace cooperative_groups;
namespace Physika {

__constant__ PBDFluidSimulateParams g_params;

__host__ void setSimulationParams(PBDFluidSimulateParams* hostParam)
{
    cudaMemcpyToSymbol(g_params, hostParam, sizeof(PBDFluidSimulateParams));
}

__device__ inline float Q_rsqrt(float number)
{
    long        i;
    float       x2, y;
    const float threehalfs = 1.5f;

    x2 = number * 0.5f;
    y  = number;
    i  = *( long* )&y;
    i  = 0x5f3759df - (i >> 1);
    y  = *( float* )&i;
    y  = y * (threehalfs - (x2 * y * y));
    return y;
}

/**
 * @brief calculate the poly6 kernel value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel value
 */
__device__ float wPoly6(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > g_params .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = g_params .m_sph_radius_squared - lengthSquared;
    return g_params .m_poly6_coff * iterm * iterm * iterm;
}

/**
 * @brief calculate the spline kernel value for adhesion force
 * @param[in] r the distance between two particles
 * @return the spline kernel value
 */
__device__ float wSpline(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared >= g_params.m_sph_radius_squared && lengthSquared < 2 * g_params.m_sph_radius_squared)
    {
        float term = 0.f;
        term       = -4 * lengthSquared * lengthSquared / g_params.m_sph_radius_squared + 6 * lengthSquared - 2 * g_params.m_sph_radius_squared;
        return 0.007 * powf(term, 0.25) / powf(lengthSquared, 3.25);
    }
    else
        return 0.0f;
}

/**
 * @brief calculate the SDF value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel gradient value
 */

__device__ float SdfBox(const float3& p, const float3& b)
{
    float3 q = fabs(p) - b;
    float3 Q = make_float3(0.0f);
    Q.x = max(q.x, 0.0);
    Q.y = max(q.y, 0.0);
    Q.z = max(q.z, 0.0);
    return length(Q) + min(max(q.x, max(q.y, q.z)), 0.0);
}

__device__
    float3
    wSpikyGrad(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3      ret           = { 0.0f, 0.0f, 0.0f };
    if (lengthSquared > g_params .m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float       iterm  = g_params .m_sph_radius - length;
    float       coff   = g_params .m_spiky_grad_coff * iterm * iterm / length;
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
    calcGridPosKernel(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - g_params .m_world_origin.x) / g_params .m_cell_size.x);
    gridPos.y = floor((p.y - g_params .m_world_origin.y) / g_params .m_cell_size.y);
    gridPos.z = floor((p.z - g_params .m_world_origin.z) / g_params .m_cell_size.z);
    return gridPos;
}

/**
 * @brief calculate hash of grid positions (clamp them within cell boundary)
 * @param[in] gridPos the grid position
 * @return the hash value of given grid
 */
__device__ unsigned int calcGridHashKernel(int3 gridPos)
{
    gridPos.x = gridPos.x & (g_params .m_grid_size.x - 1);
    gridPos.y = gridPos.y & (g_params .m_grid_size.y - 1);
    gridPos.z = gridPos.z & (g_params .m_grid_size.z - 1);
    return gridPos.z * g_params .m_grid_size.x * g_params .m_grid_size.y + gridPos.y * g_params .m_grid_size.x + gridPos.x;
}

__global__ void calcParticlesHashKernel(
    unsigned int* gridParticleHash,
    float4*       pos,
    unsigned int  numParticles)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    volatile float4 curPos    = pos[index];
    int3            gridPos   = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z));
    unsigned int    hashValue = calcGridHashKernel(gridPos);
    gridParticleHash[index]   = hashValue;
}

__global__ void findCellRangeKernel(
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

__global__ void fluidAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        deltaTime,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    float4 readVel = velocity[index];
    float3 nVel    = make_float3(readVel);
    float3 nPos    = make_float3(position[index]);
    nVel += deltaTime * g_params .m_gravity;
    if (phase == 2.f)
        return;
    if (phase == 0.f)
        nPos += deltaTime * nVel;

    bool contact = false;

    float3 contactNormal = make_float3(0.f);
    // collision with walls.
    if (nPos.x > g_params .m_world_box_corner2.x - g_params .m_particle_radius)
    {
        contact       = false;
        contactNormal = { -1.f, 0.f, 0.f };
        nPos.x        = g_params .m_world_box_corner2.x - g_params .m_particle_radius;
    }
    if (nPos.x < g_params .m_world_box_corner1.x + g_params .m_particle_radius)
    {
        contact       = false;
        contactNormal = { 1.f, 0.f, 0.f };
        nPos.x        = g_params .m_world_box_corner1.x + g_params .m_particle_radius;
    }
    if (nPos.y > g_params .m_world_box_corner2.y - g_params .m_particle_radius)
        nPos.y = g_params .m_world_box_corner2.y - g_params .m_particle_radius;
    if (nPos.y < g_params .m_world_box_corner1.y + g_params .m_particle_radius)
    {
        contact       = true;
        nPos.y        = g_params .m_world_box_corner1.y + g_params .m_particle_radius;
        contactNormal = { 0.f, 1.f, 0.f };
    }
    if (nPos.z > g_params .m_world_box_corner2.z - g_params .m_particle_radius)
    {
        contact       = false;
        nPos.z        = g_params .m_world_box_corner2.z - g_params .m_particle_radius;
        contactNormal = { 0.f, 0.f, -1.f };
    }
    if (nPos.z < g_params .m_world_box_corner1.z + g_params .m_particle_radius)
    {
        contact       = false;
        contactNormal = { 0.f, 0.f, 1.f };
        nPos.z        = g_params .m_world_box_corner1.z + g_params .m_particle_radius;
    }

    if (contact)
    {
        float3 normVel                 = dot(nVel, contactNormal) * contactNormal;
        float3 tangVel                 = nVel - normVel;
        float  staticFrictionThreshold = g_params .m_static_fricton_coeff * length(normVel);
        if (length(tangVel) < staticFrictionThreshold)
        {
            tangVel = make_float3(0.f);
        }
        else
        {
            tangVel = tangVel * (1.0f - g_params .m_dynamic_fricton_coeff);
        }
        nVel = normVel * g_params .m_damp + tangVel;
    }

    float invMass = 1.f;
    // granular advect
    if (phase == 1.f)
    {
        nPos += deltaTime * nVel;
        float stakeHeight = nPos.y - g_params.m_world_origin.y;
        float invMass     = 1.f / expf(g_params.m_stack_height_coeff * stakeHeight);
    }
    predictedPos[index] = make_float4(nPos, invMass);
}

__global__ void contactSDF(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float*       particlePhase,
    float        deltaTime,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    float  phase         = particlePhase[index];
    float4 readVel       = velocity[index];
    float3 nVel          = make_float3(readVel);
    float3 oldPos        = make_float3(position[index]);
    float3 newPos        = make_float3(predictedPos[index]);
    float3 b             = { 40.0f, 2.0f, 20.0f };
    float3 dis           = { 0.0f, 0.0f, 0.0f };
    float  sdf           = 1000.f;  // sdf value
    bool   contact       = false;
    float3 contactNormal = { 0.f, 1.f, 0.f };
    float3 edge          = { 0.f, 0.f, 0.f };
    if (phase == 0.f)
    {
        float sdf = SdfBox(newPos, b);
        if (sdf <= 0.f)
        {
            contact = true;
            newPos.y  = 2.f + g_params.m_particle_radius;
        }
    }
    float invMass = 1.f;
    //velocity[index] = make_float4(nVel, 0);
    position[index] = make_float4(oldPos, invMass);
    predictedPos[index] = make_float4(newPos, invMass);

    /* if (phase != 0.f)
        return;

    float3 curPos  = make_float3(predictedPos[index]);
    int3   gridPos = calcGridPosKernel(curPos);
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                // empty cell.
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    if (i == index)
                        continue;
                    float  phase2    = particlePhase[i];  // neighbour particle phase
                    if (phase2 == 3.f)
                    {
                        float4 neighbour = predictedPos[i];
                        dis              = { neighbour.x - curPos.x, neighbour.y - curPos.y, neighbour.z - curPos.x };
                        sdf              = min(sdf, length(dis));
                        contactNormal.x += dis.y - dis.z;
                        contactNormal.y += dis.z - dis.x;
                        contactNormal.z += dis.x - dis.y;
                    }
                    // normalized normal vector
                    float magnitude = sqrt(contactNormal.x * contactNormal.x + contactNormal.y * contactNormal.y + contactNormal.z * contactNormal.z);
                    contactNormal.x /= magnitude;
                    contactNormal.y /= magnitude;
                    contactNormal.z /= magnitude;
                }
            }
        }
    }
    if (sdf <= 0.5f)
    {
        contact = true;
        float3 I = nVel / sqrtf(dot(nVel, nVel));
        float3 R = { 0.0f, 0.0f, 0.0f };

        R = I - 2.f * dot(contactNormal, I) * contactNormal;  // reflection vector
        newPos += contactNormal * g_params.m_particle_radius;
    }
    float invMass = 1.f;
    position[index]     = make_float4(oldPos, invMass);
    predictedPos[index] = make_float4(newPos, invMass);*/
}

__global__ void calcLagrangeMultiplier(
    float4*       predictedPos,
    float4*       velocity,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles,
    unsigned int  numCells)
{
    // calculate current particle's density and lagrange multiplier.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    if (phase == 1.0)
        return;

    float3 readVel = make_float3(velocity[index]);
    float3 curPos  = make_float3(predictedPos[index]);
    int3   gridPos = calcGridPosKernel(curPos);

    float  density             = 0.0f;
    float  gradSquaredSum_j    = 0.0f;
    float  gradSquaredSumTotal = 0.0f;
    float3 curGrad, gradSum_i = { 0.0f, 0.0f, 0.0f };
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                // empty cell.
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    float4 neighbour = predictedPos[i];
                    float3 r         = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
                    density += wPoly6(r);
                    curGrad = wSpikyGrad(r);
                    curGrad.x *= g_params .m_inv_rest_density;
                    curGrad.y *= g_params.m_inv_rest_density;
                    curGrad.z *= g_params.m_inv_rest_density;

                    gradSum_i.x += curGrad.x;
                    gradSum_i.y += curGrad.y;
                    gradSum_i.y += curGrad.y;
                    if (i != index)
                        gradSquaredSum_j += curGrad.x * curGrad.x + curGrad.y * curGrad.y + curGrad.z * curGrad.z;
                }
            }
        }
    }
    gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;

    // density constraint.
    predictedPos[index].w = density;
    float constraint      = density * g_params.m_inv_rest_density - 1.0f;
    float lambda          = -(constraint) / (gradSquaredSumTotal + g_params.m_lambda_eps);
    velocity[index]       = { readVel.x, readVel.y, readVel.z, lambda };
}

__global__ void calcDeltaPosition(
    float4*       predictedPos,
    float4*       velocity,
    float3*       deltaPos,
    float*        particlePhase,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];

    float4 readPos     = predictedPos[index];
    float4 readVel     = velocity[index];
    float3 curPos      = { readPos.x, readPos.y, readPos.z };
    int3   gridPos     = calcGridPosKernel(curPos);
    float  curLambda   = readVel.w;
    float3 deltaP      = { 0.0f, 0.0f, 0.0f };
    float3 frictdeltaP = make_float3(0.f);
    float  invMass1    = readPos.w;
    float3 nVel        = make_float3(0.f);
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    float4 readneighbour   = predictedPos[i];
                    float  neighbourLambda = velocity[i].w;
                    float  phase2          = particlePhase[i];
                    if (phase == 0.f)
                    {
                        float3 r        = { curPos.x - readneighbour.x, curPos.y - readneighbour.y, curPos.z - readneighbour.z };
                        float  len      = 1.f / Q_rsqrt(dot(r, r));
                        float  corrTerm = wPoly6(r) * g_params.m_one_div_wPoly6;
                        float  coff     = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
                        float3 grad     = wSpikyGrad(r);
                        r /= len;
                        deltaP += coff * grad;
                        //if (len <= 1.4f * params.m_particleRadius)
                        // nVel += 1.f/60.f * 1000.0f * 1 * 1 * r * cos(len * 3.36f / params.m_particleRadius);
                    }
                    if (phase == 1.f)
                    {
                        if (i == index)
                            continue;
                        float3 deltaP2   = make_float3(0.f);
                        float3 neighbour = make_float3(readneighbour);
                        float  invMass2  = readneighbour.w;
                        float  wSum      = invMass1 + invMass2;
                        float  weight1   = invMass1 / wSum;
                        float3 r         = curPos - neighbour;
                        float  len       = length(r);
                        r /= len;
                        if (len < g_params.m_particle_radius * 2)
                        {
                            float3 corr = g_params.m_stiffness * r * (g_params.m_particle_radius * 2 - len) / wSum;
                            deltaP += invMass1 * corr;
                            deltaP2 -= invMass2 * corr;
                        }
                        if (phase2 == 0.f)
                            continue;
                        // friction model
                        float3 relativedeltaP = deltaP - deltaP2;
                        frictdeltaP           = relativedeltaP - dot(relativedeltaP, r) * r;
                        float d_frictdeltaP   = length(frictdeltaP);
                        if (d_frictdeltaP < g_params.m_static_frict_threshold)
                        {
                            frictdeltaP *= weight1 * g_params.m_stiffness;
                        }
                        else
                        {
                            frictdeltaP *= weight1 * min(1.f, g_params.m_dynamic_frict_threshold / d_frictdeltaP) * g_params.m_stiffness;
                        }
                    }
                }
            }
        }
    }
    float3 ret = make_float3(0.f);
    if (phase == 0.0)
        ret = { deltaP.x * g_params.m_inv_rest_density, deltaP.y * g_params.m_inv_rest_density, deltaP.z * g_params.m_inv_rest_density };
    if (phase == 1.0)
        ret = deltaP + frictdeltaP;
    deltaPos[index] = ret;
}

__global__ void addDeltaPosition(
    float4*      predictedPos,
    float3*      deltaPos,
    float*       particlePhase,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float phase = particlePhase[index];
    //if (phase != 1.f)
         //return;
    float3 readPos = make_float3(predictedPos[index]);
    readPos += deltaPos[index];

    predictedPos[index] = { readPos.x, readPos.y, readPos.z, 1.f };
}

__global__ void distanceConstrain(
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
    int3   gridPos     = calcGridPosKernel(curPos);
    float3 deltaP      = make_float3(0.f);
    float3 frictdeltaP = make_float3(0.f);
    float  invMass1    = readPos.w;
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
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
                    float  invMass2      = readNeighbour.w;
                    float  wSum          = invMass1 + invMass2;
                    float  weight1       = invMass1 / wSum;
                    float3 r             = curPos - neighbour;
                    float  len           = length(r);
                    r /= len;
                    if (len < g_params .m_particle_radius * 2)
                    {
                        float3 corr = g_params .m_stiffness * r * (g_params .m_particle_radius * 2 - len) / wSum;
                        deltaP += invMass1 * corr;
                        deltaP2 -= invMass2 * corr;
                    }
                    float3 relativedeltaP = deltaP - deltaP2;
                    frictdeltaP           = relativedeltaP - dot(relativedeltaP, r) * r;
                    float d_frictdeltaP   = length(frictdeltaP);
                    if (d_frictdeltaP < g_params .m_static_frict_threshold)
                    {
                        frictdeltaP *= weight1 * g_params .m_stiffness;
                    }
                    else
                    {
                        frictdeltaP *= weight1 * min(1.f, g_params .m_dynamic_frict_threshold / d_frictdeltaP) * g_params .m_stiffness;
                    }
                }
            }
        }
    }
    deltaPos[index] = deltaP + frictdeltaP;
}

__global__ void updateVelocityAndPosition(
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
    //if (phase != 1.f)
        //return;
    float4 oldPos  = position[index];
    float4 newPos  = predictedPos[index];
    float4 readVel = velocity[index];
    float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
    posDiff *= invDeltaTime;
    velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
    if (length(newPos - oldPos) > g_params .m_sleep_threshold)
        position[index] = { newPos.x, newPos.y, newPos.z, newPos.w };
}
__global__ void solverCollisionConstrain(
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
    predictedPos[particle_id] = { prePos.x, prePos.y, prePos.z, 1.f };
    position[particle_id] += { deltaP.x, deltaP.y, deltaP.z, 0.f };
}

// cohesion
__global__ void add_surfacetension(
    float4*       velocity,
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    float         deltaTime,
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
    if (phase == 1.f)
        return;

    float3 readVel = make_float3(velocity[index]);
    float3 curPos  = make_float3(predictedPos[index]);

    //float3 nVel = readVel;
    float3 nVel    = { 0.f, 0.f, 0.f };
    float4 readPos = predictedPos[index];
    int3   gridPos = calcGridPosKernel(curPos);

    float Mass_i = readPos.w;
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                // empty cell.
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    if (i == index)
                        continue;
                    float  phase2    = particlePhase[i];  // neighbour particle phase
                    float4 neighbour = predictedPos[i];
                    float3 x_ij      = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
                    float  len       = length(x_ij);
                    x_ij /= len;
                    if (phase2 == 0.f && len <= 1.5f * g_params.m_sph_radius)
                        nVel += g_params.m_sf_coeff * deltaTime * g_params.m_scaling_factor * 1 * x_ij * cos(len * 3.36f / g_params.m_sph_radius);
                }
            }
        }
    }
    velocity[index] += make_float4(nVel, 0);
}


__global__ void add_adhesionforce(
    float4*       velocity,
    float4*       predictedPos,
    float3*       deltaPos,
    float*        particlePhase,
    float         deltaTime,
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
    if (phase == 1.f)
        return;

    float3 readVel = make_float3(velocity[index]);
    float3 curPos  = make_float3(predictedPos[index]);

    // float3 nVel = readVel;
    float3 nVel    = { 0.f, 0.f, 0.f };
    float4 readPos = predictedPos[index];
    int3   gridPos = calcGridPosKernel(curPos);

    float Mass_i = readPos.w;
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
                unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                // empty cell.
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    if (i == index)
                        continue;
                    float  phase2    = particlePhase[i];  // neighbour particle phase
                    float4 neighbour = predictedPos[i];
                    float3 r         = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
                    float3 x_ij      = r;
                    float  len       = length(x_ij);
                    x_ij /= len;
                    // compute adhesion between fluid & boundary
                    if (phase2 == 3.f && len <= 1.1f * g_params.m_sph_radius)
                        //nVel -= g_params.m_adhesion_coeff * 1 * deltaTime * wSpline(r) * x_ij;
                        nVel += g_params.m_adhesion_coeff * deltaTime * g_params.m_scaling_factor * 1 * x_ij * cos(len * 3.36f / g_params.m_sph_radius);
                }
            }
        }
    }
    velocity[index] += make_float4(nVel, 0);
}

}  // namespace Physika