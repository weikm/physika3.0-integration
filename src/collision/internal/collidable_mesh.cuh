#pragma once
#include "collision/internal/collision_box.cuh"
#include "collision/internal/collision_tool.cuh"
#include "collision/internal/collision_tri3.cuh"


typedef struct
{
    uint    numFace, numVert;   // num of face and vertex in mesh
    float3 *_dx, *_dx0;         // vertex data
    tri3f*  _df;                // face data
    g_box*  _dfBx;              // face bounding box

    // init function
    void init()
    {
        numFace = 0;
        numVert = 0;
        _dx = _dx0 = NULL;
        _df        = NULL;
        _dfBx      = NULL;
    }

    void destroy()
    {
        if (_dx == NULL)
            return;
        checkCudaErrors(cudaFree(_dx));
        checkCudaErrors(cudaFree(_dx0));
        checkCudaErrors(cudaFree(_df));
        checkCudaErrors(cudaFree(_dfBx));
    }

    void computeWSdata(float thickness, bool ccd);
} g_mesh;
