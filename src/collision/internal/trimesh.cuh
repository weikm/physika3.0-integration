#pragma once

#include "collision/internal/collision_def.cuh"
#include "collision/internal/collision_tool.cuh"
#include "collision/internal/collision_tri3.cuh"
#include "collision/internal/collision_bvh.cuh"
#include "collision/internal/collision_qbvh.cuh"
class g_triMesh
{
public:
    int          _num_vtx;
    int          _num_tri;
    g_tri3f*     _tris;
    REAL3*       _vtxs;
    int          _num_bvh_nodes;
    g_bvh_node*  _bvh_nodes;
    g_qbvh_node* _qbvh_nodes;

    g_sbvh_node* _sbvh_upper_nodes;
    g_sbvh_node* _sbvh_lower_nodes;
    int          _sbvh_upper_num, _sbvh_lower_num;

    g_triMesh()
    {
        _num_vtx       = 0;
        _num_tri       = 0;
        _tris          = nullptr;
        _vtxs          = nullptr;
        _num_bvh_nodes = 0;
        _bvh_nodes     = nullptr;
        _qbvh_nodes    = nullptr;

        _sbvh_upper_nodes = _sbvh_lower_nodes = nullptr;
        _sbvh_upper_num = _sbvh_lower_num = 0;
    }

    void clear()
    {
        if (_tris)
            cudaFree(_tris);
        if (_vtxs)
            cudaFree(_vtxs);
        if (_bvh_nodes)
            cudaFree(_bvh_nodes);
        if (_qbvh_nodes)
            cudaFree(_qbvh_nodes);

        if (_sbvh_lower_nodes)
            cudaFree(_sbvh_lower_nodes);
    }
};