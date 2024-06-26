/**
 * @author     : Yuhang Xu (mr.xuyh@qq.com)
 * @date       : 2023-08-17
 * @description: CUDA Kernel functions of solving fluid solid particles
 * @version    : 1.0
 */
#include "fluid_solid_coupling_kernel.cuh"

#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>
#include <math_constants.h>

#include "fluid_solid_coupling_params.hpp"
#include "helper_math.hpp"

using namespace cooperative_groups;
namespace Physika {

__constant__ FluidSolidCouplingParams fluid_solid_params;

__host__ void setSimulationParams(FluidSolidCouplingParams* hostParam)
{
    cudaMemcpyToSymbol(fluid_solid_params, hostParam, sizeof(FluidSolidCouplingParams));
}
/**
 * @brief calculate the poly6 kernel value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel value
 */
__device__ float wPoly6Coupling(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    if (lengthSquared > fluid_solid_params.m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return 0.0f;
    float iterm = fluid_solid_params.m_sph_radius_squared - lengthSquared;
    return fluid_solid_params.m_poly6_coff * iterm * iterm * iterm;
}

/**
 * @brief calculate the poly6 kernel gradient value
 * @param[in] r the distance between two particles
 * @return the poly6 kernel gradient value
 */
__device__
    float3
    wSpikyGradCoupling(const float3& r)
{
    const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
    float3      ret           = { 0.0f, 0.0f, 0.0f };
    if (lengthSquared > fluid_solid_params.m_sph_radius_squared || lengthSquared <= 0.00000001f)
        return ret;
    const float length = sqrtf(lengthSquared);
    float       iterm  = fluid_solid_params.m_sph_radius - length;
    float       coff   = fluid_solid_params.m_spiky_grad_coff * iterm * iterm / length;
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
    calcGridPosCouplingKernel(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - fluid_solid_params.m_world_origin.x) / fluid_solid_params.m_cell_size.x);
    gridPos.y = floor((p.y - fluid_solid_params.m_world_origin.y) / fluid_solid_params.m_cell_size.y);
    gridPos.z = floor((p.z - fluid_solid_params.m_world_origin.z) / fluid_solid_params.m_cell_size.z);
    return gridPos;
}

/**
 * @brief calculate hash of grid positions (clamp them within cell boundary)
 * @param[in] gridPos the grid position
 * @return the hash value of given grid
 */
__device__ unsigned int calcGridHashCouplingKernel(int3 gridPos)
{
    gridPos.x = gridPos.x & (fluid_solid_params.m_grid_size.x - 1);
    gridPos.y = gridPos.y & (fluid_solid_params.m_grid_size.y - 1);
    gridPos.z = gridPos.z & (fluid_solid_params.m_grid_size.z - 1);
    return gridPos.z * fluid_solid_params.m_grid_size.x * fluid_solid_params.m_grid_size.y + gridPos.y * fluid_solid_params.m_grid_size.x + gridPos.x;
}

__global__ void calcCouplingParticlesHashKernel(
    unsigned int* gridParticleHash,
    float4*       pos,
    unsigned int  numParticles)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;

    volatile float4 curPos    = pos[index];
    int3            gridPos   = calcGridPosCouplingKernel(make_float3(curPos.x, curPos.y, curPos.z));
    unsigned int    hashValue = calcGridHashCouplingKernel(gridPos);
    gridParticleHash[index]   = hashValue;
}

__global__ void findCellRangeCouplingKernel(
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

__global__ void particleAdvect(
    float4*      position,
    float4*      velocity,
    float4*      predictedPos,
    float3*      collisionForce,
    int*         phase,
    float        deltaTime,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float4 readVel = velocity[index];
    float3 nVel    = make_float3(readVel);
    float3 nPos    = make_float3(position[index]);
    float  read_w  = position[index].w;
    nVel += deltaTime * (fluid_solid_params.m_gravity + collisionForce[index]);
    bool contact = false;

    float3 contactNormal = make_float3(0.f);
    // collision with walls.
    if (nPos.x > fluid_solid_params.m_world_box_corner2.x - fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        contactNormal = { -1.f, 0.f, 0.f };
        nPos.x        = fluid_solid_params.m_world_box_corner2.x - fluid_solid_params.m_particle_radius;
    }
    if (nPos.x < fluid_solid_params.m_world_box_corner1.x + fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        contactNormal = { 1.f, 0.f, 0.f };
        nPos.x        = fluid_solid_params.m_world_box_corner1.x + fluid_solid_params.m_particle_radius;
    }
    if (nPos.y > fluid_solid_params.m_world_box_corner2.y - fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        contactNormal = { 0.f, -1.f, 0.f };
        nPos.y        = fluid_solid_params.m_world_box_corner2.y - fluid_solid_params.m_particle_radius;
    }
    if (nPos.y < fluid_solid_params.m_world_box_corner1.y + fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        nPos.y        = fluid_solid_params.m_world_box_corner1.y + fluid_solid_params.m_particle_radius;
        contactNormal = { 0.f, 1.f, 0.f };
    }
    if (nPos.z > fluid_solid_params.m_world_box_corner2.z - fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        nPos.z        = fluid_solid_params.m_world_box_corner2.z - fluid_solid_params.m_particle_radius;
        contactNormal = { 0.f, 0.f, -1.f };
    }
    if (nPos.z < fluid_solid_params.m_world_box_corner1.z + fluid_solid_params.m_particle_radius)
    {
        contact       = true;
        contactNormal = { 0.f, 0.f, 1.f };
        nPos.z        = fluid_solid_params.m_world_box_corner1.z + fluid_solid_params.m_particle_radius;
    }

    if (contact)
    {
        float3 normVel                 = dot(nVel, contactNormal) * contactNormal;
        float3 tangVel                 = nVel - normVel;
        float  staticFrictionThreshold = fluid_solid_params.m_static_fricton_coeff * length(normVel);
        if (length(tangVel) < staticFrictionThreshold)
        {
            tangVel = make_float3(0.f);
        }
        else
        {
            tangVel = tangVel * (1.0f - fluid_solid_params.m_dynamic_fricton_coeff);
        }
        nVel = normVel * fluid_solid_params.m_damp + tangVel;
    }

    nPos += deltaTime * nVel;

    predictedPos[index] = make_float4(nPos, read_w);

    //if (phase[index] == 2)
    //{
    //    float stakeHeight   = nPos.y - fluid_solid_params.m_world_origin.y;
    //    float invMass       = read_w / expf(fluid_solid_params.m_stack_height_coeff * stakeHeight);
    //    predictedPos[index] = make_float4(nPos, invMass);
    //}
    //else
    //{
    //    predictedPos[index] = make_float4(nPos, read_w);
    //}
}

__global__ void addDeltaPositionCoupling(
    float4*      postion,
    float4*      predictedPos,
    float3*      deltaPos,
    int*         phase,
    unsigned int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float3 readPos = make_float3(predictedPos[index]);
    float  read_w  = postion[index].w;
    readPos += deltaPos[index];

    predictedPos[index] = { readPos.x, readPos.y, readPos.z, read_w };
}

__global__ void calcLagrangeMultiplierCoupling(
    int*          phase,
    float4*       predictedPos,
    float4*       velocity,
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
    //if (phase[index] != 3)
    //    return;

    float3 readVel = make_float3(velocity[index]);
    float3 curPos  = make_float3(predictedPos[index]);
    int3   gridPos = calcGridPosCouplingKernel(curPos);

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
                unsigned int neighbourGridIndex = calcGridHashCouplingKernel(neighbourGridPos);
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
                    float  neighboour_mass = (1.f / neighbour.w);
                    density += wPoly6Coupling(r) * neighboour_mass;
                    curGrad = wSpikyGradCoupling(r) * neighboour_mass;
                    curGrad *= fluid_solid_params.m_inv_rest_density;

                    gradSum_i += curGrad;
                    if (i != index)
                        gradSquaredSum_j += curGrad.x * curGrad.x + curGrad.y * curGrad.y + curGrad.z * curGrad.z;
                        //gradSquaredSum_j *= neighbour.w;
                }
            }
        }
    }
    gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;
    gradSquaredSumTotal *= predictedPos[index].w;

    // density constraint.
    //predictedPos[index].w = 1.f / density;
    float constraint      = density * fluid_solid_params.m_inv_rest_density - 1.0f;
    float lambda     = 0.0;
    if (constraint > 0.0)
    {
        lambda = -(constraint) / (gradSquaredSumTotal + fluid_solid_params.m_lambda_eps);
    }
    velocity[index]       = { readVel.x, readVel.y, readVel.z, lambda };
}

__global__ void calcDeltaPositionCoupling(
    int*          phase,
    float4*       predictedPos,
    float4*       velocity,
    float3*       deltaPos,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridParticleHash,
    unsigned int  numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    float4 readPos = predictedPos[index];
    float4 readVel = velocity[index];
    float3 curPos  = { readPos.x, readPos.y, readPos.z };
    int3   gridPos = calcGridPosCouplingKernel(curPos);

    float  curLambda = readVel.w;
    float3 deltaP    = { 0.0f, 0.0f, 0.0f };

    // solid
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
                unsigned int neighbourGridIndex = calcGridHashCouplingKernel(neighbourGridPos);
                unsigned int startIndex         = cellStart[neighbourGridIndex];
                if (startIndex == 0xffffffff)
                    continue;
                unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
                for (unsigned int i = startIndex; i < endIndex; ++i)
                {
                    if (phase[index] != 2)  // fluid
                    {
                        float4 neighbour       = predictedPos[i];
                        float  neighbourLambda = velocity[i].w;
                        float3 r               = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
                        float  corrTerm        = wPoly6Coupling(r) * fluid_solid_params.m_one_div_wPoly6;
                        float  coff            = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
                        float3 grad            = wSpikyGradCoupling(r);
                        deltaP.x += coff * grad.x;
                        deltaP.y += coff * grad.y;
                        deltaP.z += coff * grad.z;
                    }
                    else  // solid
                    {
                        if (phase[i] == 2)  // 刚体内部粒子不产生碰撞与摩擦
                        {
                            continue;
                        }
                        float3 deltaP2       = make_float3(0.f);
                        float4 readNeighbour = predictedPos[i];
                        float3 neighbour     = make_float3(readNeighbour);
                        float  invMass2      = readNeighbour.w;
                        float  wSum          = invMass1 + invMass2;
                        float  weight1       = invMass1 / wSum;
                        float3 r             = curPos - neighbour;
                        float  len           = length(r);
                        r /= len;
                        if (len < fluid_solid_params.m_particle_radius * 2)
                        {
                            // 这里可以更改所乘的刚度，这样会造成非线性
                            float3 corr = fluid_solid_params.m_stiffness * r * (fluid_solid_params.m_particle_radius * 2 - len) / wSum;
                            deltaP += invMass1 * corr;
                            deltaP2 -= invMass2 * corr;
                        }
                        float3 relativedeltaP = deltaP - deltaP2;
                        frictdeltaP           = relativedeltaP - dot(relativedeltaP, r) * r;
                        float d_frictdeltaP   = length(frictdeltaP);
                        if (d_frictdeltaP < fluid_solid_params.m_static_frict_threshold)
                        {
                            frictdeltaP *= weight1 * fluid_solid_params.m_stiffness;
                        }
                        else
                        {
                            frictdeltaP *= weight1 * min(1.f, fluid_solid_params.m_dynamic_frict_threshold / d_frictdeltaP) * fluid_solid_params.m_stiffness;
                        }
                    }
                }
            }
        }
    }

    if (phase[index] != 2)  // fluid
    {
        float3 ret      = { deltaP.x * fluid_solid_params.m_inv_rest_density, deltaP.y * fluid_solid_params.m_inv_rest_density, deltaP.z * fluid_solid_params.m_inv_rest_density };
        deltaPos[index] = ret * predictedPos[index].w;
    }
    else if (phase[index] == 2)
    {
        deltaPos[index] = deltaP + frictdeltaP;
    }
    else
    {
        deltaPos[index] = make_float3(0.0);
    }
}

__global__ void calMassCenterMatrixR(
    int*          phase,
    float4*       predicted_pos,
    float3*       cm,
    const float3* radius_pos,
    const bool    allow_stretch,
    unsigned int  num_particles,
    mat3*         R)
{
    // center of mass
    *cm        = make_float3(0.f);
    float wsum = 0.0;
    float eps  = static_cast<float>(1e-6);

    for (int i = 0; i < num_particles; i++)
    {
        if (phase[i] != 2)  // not solid
        {
            continue;
        }
        float wi = static_cast<float>(1.0) / (predicted_pos[i].w + eps);
        *cm += make_float3(predicted_pos[i]) * wi;
        wsum += wi;
    }
    if (wsum == 0.0)
        return;
    *cm /= wsum;

    // Apq
    mat3 mat;
    for (int i = 0; i < num_particles; i++)
    {
        if (phase[i] != 2)  // not solid
        {
            continue;
        }
        float3 q               = radius_pos[i];
        float3 p               = make_float3(predicted_pos[i]) - *cm;

        float w = static_cast<float>(1.0) / (predicted_pos[i].w + eps);
        p *= w;

        mat3 temp(p * q.x, p * q.y, p * q.z);
        //mat3 temp(p.x * q, p.y * q, p.z * q);
        mat += temp;
    }

    if (allow_stretch)
    {
        *R = mat;
    }
    else
    {
        mat3 temp;
        //mat = mat.transpose();
        //mat.polarDecomposition(temp, *R);
        mat.polarDecomposition(*R);
        *R = R->transpose();
        
    }
    return;
}


__global__ void updateSolidVelocityAndPosition(
    int*          phase,
    float4*       position,
    float4*       predicted_pos,
    float3*       delta_pos,
    const float3* cm,
    const float3* radius_pos,
    const float   stiffness,
    float4*       velocity,
    const float   invDeltaTime,
    const mat3*   R,
    unsigned int  num_particles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles)
        return;

    float3 oldPos = make_float3(position[index]);
    float3 newPos;
    float  predicted_pos_w = predicted_pos[index].w;
    if (phase[index] != 2)
    {
        newPos = make_float3(predicted_pos[index]);
    }
    else
    {
        newPos = *cm + *R * radius_pos[index];
    }
    float4 readVel = velocity[index];
    float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
    posDiff *= invDeltaTime;
    velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
    //if (length(newPos - oldPos) > fluid_solid_params.m_sleep_threshold)
    position[index] = { newPos.x, newPos.y, newPos.z, predicted_pos_w };
}

__global__ void solverCollisionConstrainCoupling(
    float4*       position,
    float4*       predictedPos,
    float3*       moveDirection,
    float*        moveDistance,
    int*          particlePhase,
    unsigned int* collision_particle_id,
    unsigned int  numCollisionParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numCollisionParticles)
        return;
    float phase = particlePhase[index];
    if (phase != 1.f)
        return;
    uint   particle_id = collision_particle_id[index];
    float3 oldpos      = make_float3(position[particle_id]);
    float3 prePos      = make_float3(predictedPos[particle_id]);

    float3 deltaP = moveDirection[index] * moveDistance[index];
    prePos += deltaP;
    predictedPos[particle_id] = { prePos.x, prePos.y, prePos.z, 1.f };
    position[particle_id] += { deltaP.x, deltaP.y, deltaP.z, 0.f };
}

}  // namespace Physika