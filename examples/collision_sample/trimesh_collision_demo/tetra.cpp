#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>
#include "crigid.h"
extern aabb<REAL> g_projBx;
using Physika::stri3f;
//
// Tetra meshes
//

struct TetraBunny
{
#include "bunny.inl"
};

static int nextLine(const char* buffer)
{
    int numBytesRead = 0;

    while (*buffer != '\n')
    {
        buffer++;
        numBytesRead++;
    }

    if (buffer[0] == 0x0a)
    {
        buffer++;
        numBytesRead++;
    }
    return numBytesRead;
}

void appendTetra(std::set<stri3f>& tris, int n0, int n1, int n2, int n3)
{
    tris.insert(stri3f(n0, n1, n2));
    tris.insert(stri3f(n0, n2, n3));
    tris.insert(stri3f(n0, n3, n1));
    tris.insert(stri3f(n3, n1, n2));
}

void getTetraBunnyData(
    unsigned int& numVtx,
    unsigned int& numTri,
    vec3f*&       ovtxs,
    tri3f*&       otris,
    REAL          scale,
    vec3f&        shift)
{
    const char* ele  = TetraBunny::getElements();
    const char* node = TetraBunny::getNodes();

    // make a large id map
    // int* idmap = new int[1024];

    int nnode     = 0;
    int ndims     = 0;
    int nattrb    = 0;
    int hasbounds = 0;
    int result    = sscanf(node, "%d %d %d %d", &nnode, &ndims, &nattrb, &hasbounds);
    node += nextLine(node);

    std::vector<vec3f> pos;
    pos.resize(nnode);
    for (int i = 0; i < pos.size(); ++i)
    {
        int   index = 0;
        float x, y, z;
        sscanf(node, "%d %f %f %f", &index, &x, &y, &z);

        node += nextLine(node);

        vec3f t    = vec3f(x, y, z);
        pos[index] = t * scale + shift;
    }

    std::set<stri3f> tris;
    {
        int ntetra  = 0;
        int ncorner = 0;
        int neattrb = 0;
        sscanf(ele, "%d %d %d", &ntetra, &ncorner, &neattrb);
        ele += nextLine(ele);

        for (int i = 0; i < ntetra; ++i)
        {
            int index = 0;
            int ni[4];

            sscanf(ele, "%d %d %d %d %d", &index, &ni[0], &ni[1], &ni[2], &ni[3]);
            ele += nextLine(ele);
            appendTetra(tris, ni[0], ni[1], ni[2], ni[3]);
        }
    }
    // delete[] idmap;

    numVtx = pos.size();
    ovtxs  = new vec3f[numVtx];
    memcpy(ovtxs, pos.data(), sizeof(vec3f) * numVtx);

    numTri = tris.size();
    otris  = new tri3f[numTri];
    int i  = 0;
    for (auto tt : tris)
        otris[i++] = tt.getTri();
}
