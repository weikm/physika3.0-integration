#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"

#include "crigid.h"
//#include "collision/internal/manifold.hpp"
#include "collision/internal/cplane.hpp"
#include "collision/internal/contact.hpp"
#include "collision/interface/mesh_mesh_discrete_collision_detect.hpp"
using Physika::cplane;
using Physika::manifold;
using Physika::manifoldPoint;
//using Physika::

#include <stdio.h>
#include <omp.h>
#include <mutex>
// mutex to lock critical region
std::mutex mylock;
int        maxTriPairs = 0;

manifold* sManifoldSet = nullptr;
id_pair*  sPairSet     = nullptr;
id_pair*  sRigSet      = nullptr;
bool      gFlat        = false;

// #define FOR_PARTICLES
// #define FOR_VOLMESH
// #define FOR_SDF

class myTimer
{
    double t0;
    char   msg[512];

public:
    myTimer(const char* msgIn)
    {
        t0 = omp_get_wtime();
        strcpy(msg, msgIn);
    }
    ~myTimer()
    {
        double tdelta = omp_get_wtime() - t0;
        printf("%s: %2.5f s\n", msg, tdelta);
    }
};

class myTimer2
{
    double dt;
    char   msg[512];

public:
    myTimer2(const char* msgIn)
    {
        dt = 0;
        strcpy(msg, msgIn);
    }

    void print()
    {
        printf("%s: %2.5f s\n", msg, dt);
    }

    void inc(double delta)
    {
        dt += delta;
    }
};

#define BUNNY_SCALE 1.f

#pragma warning(disable : 4996)

extern void  drawSdfPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void  drawMinPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void  drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs);
extern void  drawRigid(crigid*, bool cyl, int level, vec3f&);
extern void  drawPlanes(bool);
extern float getLargestVelocityNorm(crigid* body1, crigid* body2);

#define SIMDSQRT12 float(0.7071067811865475244008443621048490)
#define RecipSqrt(x) (( float )(float(1.0) / sqrtf(float(x)))) /* reciprocal square root */
static int stepIdx = 0;

enum AnisotropicFrictionFlags
{
    CF_ANISOTROPIC_FRICTION_DISABLED = 0,
    CF_ANISOTROPIC_FRICTION          = 1,
    CF_ANISOTROPIC_ROLLING_FRICTION  = 2
};

__forceinline float restitutionCurve(float rel_vel, float restitution)
{
    return restitution * -rel_vel;
}

__forceinline void PlaneSpace1(const vec3f& n, vec3f& p, vec3f& q)
{
    //	float a = fabs(n[2]);
    //	float b = SIMDSQRT12;

    if (fabs(n[2]) > SIMDSQRT12)
    {
        // choose p in y-z plane
        float a = n[1] * n[1] + n[2] * n[2];
        float k = RecipSqrt(a);
        p[0]    = 0;
        p[1]    = -n[2] * k;
        p[2]    = n[1] * k;
        // set q = n x p
        q[0] = a * k;
        q[1] = -n[0] * p[2];
        q[2] = n[0] * p[1];
    }
    else
    {
        // choose p in x-y plane
        float a = n[0] * n[0] + n[1] * n[1];
        float k = RecipSqrt(a);
        p[0]    = -n[1] * k;
        p[1]    = n[0] * k;
        p[2]    = 0;
        // set q = n x p
        q[0] = -n[2] * p[1];
        q[1] = n[2] * p[0];
        q[2] = a * k;
    }
}

aabb<REAL>  g_box;
aabb<REAL>  g_projBx;
REAL g_time = 0.0f;

extern bool verb;

vec3f      projDir(0.0f, -1.0f, 0.0f);
REAL       maxDist = 20.0;
static int sidx    = 0;

class cscene
{
    std::vector<triMesh*>    _meshs;
    std::vector<crigid*>   _rigids;
    std::vector<id_pair>   _rigid_pairs;

    // for GPU updating...
    std::vector<transf> _trfs;

public:

public:
    ~cscene()
    {
        clear();
    }

    void clear()
    {
        for (auto r : _rigids)
            delete r;

        for (auto m : _meshs)
            delete m;

        _meshs.clear();
        _rigids.clear();
    }

    FORCEINLINE crigid* getRigid(int rid)
    {
        return (rid < 0) ? nullptr : _rigids[rid];
    }

    FORCEINLINE int getRigidID(crigid* r)
    {
        return r == nullptr ? -1 : r->getID();
    }

    FORCEINLINE int getPlaneID(cplane* p)
    {
        return p == nullptr ? -1 : p->getID();
    }

    FORCEINLINE void setID()
    {
        for (int i = 0; i < _rigids.size(); i++)
        {
            _rigids[i]->setID(i);
        }
    }

    void draw(int level, bool showCD, bool showBody, bool showOnly)
    {
        if (showCD)
        {
            drawMinPair(_rigids[0], _rigids[1], minPairs);
            drawCDPair(_rigids[0], _rigids[1], cdPairs);
            drawSdfPair(_rigids[0], _rigids[1], sdfPairs);
        }

        if (showBody)
        {
            for (auto r : _rigids)
            {
                drawRigid(r, false, level, vec3f());
                if (showOnly)
                    break;
            }
        }
    }

    void addMesh(triMesh* km)
    {
        _meshs.push_back(km);
    }

    void addRigid(crigid* rig)
    {
        _rigids.push_back(rig);
    }

    // for distance query
    std::vector<vec3f> minPairs;

    // for collision detection
    std::vector<id_pair> cdPairs, cdPairs2;

    // for SDF query
    std::vector<vec3f> sdfPairs;
} g_scene;

vec3f dPt0, dPt1, dPtw;

bool readobjfile(const char*   path,
                 unsigned int& numVtx,
                 unsigned int& numTri,
                 tri3f*&       tris,
                 vec3f*&       vtxs,
                 REAL          scale,
                 vec3f         shift,
                 bool          swap_xyz,
                 vec2f*&       texs,
                 tri3f*&       ttris)
{
    vector<tri3f> triset;
    vector<vec3f> vtxset;
    vector<vec2f> texset;
    vector<tri3f> ttriset;

    FILE* fp = fopen(path, "rt");
    if (fp == NULL)
        return false;

    char buf[1024];
    while (fgets(buf, 1024, fp))
    {
        if (buf[0] == 'v' && buf[1] == ' ')
        {
            double x, y, z;
            sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);

            if (swap_xyz)
                vtxset.push_back(vec3f(z, x, y) * scale + shift);
            else
                vtxset.push_back(vec3f(x, y, z) * scale + shift);
        }
        else

            if (buf[0] == 'v' && buf[1] == 't')
        {
            double x, y;
            sscanf(buf + 3, "%lf%lf", &x, &y);

            texset.push_back(vec2f(x, y));
        }
        else if (buf[0] == 'f' && buf[1] == ' ')
        {
            int  id0, id1, id2, id3     = 0;
            int  tid0, tid1, tid2, tid3 = 0;
            bool quad = false;

            int   count = sscanf(buf + 2, "%d/%d", &id0, &tid0);
            char* nxt   = strchr(buf + 2, ' ');
            sscanf(nxt + 1, "%d/%d", &id1, &tid1);
            nxt = strchr(nxt + 1, ' ');
            sscanf(nxt + 1, "%d/%d", &id2, &tid2);

            nxt = strchr(nxt + 1, ' ');
            if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9')
            {  // quad
                if (sscanf(nxt + 1, "%d/%d", &id3, &tid3))
                    quad = true;
            }

            id0--, id1--, id2--, id3--;
            tid0--, tid1--, tid2--, tid3--;

            triset.push_back(tri3f(id0, id1, id2));
            if (count == 2)
            {
                ttriset.push_back(tri3f(tid0, tid1, tid2));
            }

            if (quad)
            {
                triset.push_back(tri3f(id0, id2, id3));
                if (count == 2)
                    ttriset.push_back(tri3f(tid0, tid2, tid3));
            }
        }
    }
    fclose(fp);

    if (triset.size() == 0 || vtxset.size() == 0)
        return false;

    numVtx = vtxset.size();
    vtxs   = new vec3f[numVtx];
    for (unsigned int i = 0; i < numVtx; i++)
        vtxs[i] = vtxset[i];

    int numTex = texset.size();
    if (numTex == 0)
        texs = NULL;
    else
    {
        texs = new vec2f[numTex];
        for (unsigned int i = 0; i < numTex; i++)
            texs[i] = texset[i];
    }

    numTri = triset.size();
    tris   = new tri3f[numTri];
    for (unsigned int i = 0; i < numTri; i++)
        tris[i] = triset[i];

    int numTTri = ttriset.size();
    if (numTTri == 0)
        ttris = NULL;
    else
    {
        ttris = new tri3f[numTTri];
        for (unsigned int i = 0; i < numTTri; i++)
            ttris[i] = ttriset[i];
    }

    return true;
}

// http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
vec3f randDir()
{
    REAL phi      = REAL(rand()) / RAND_MAX * M_PI * 2;
    REAL costheta = REAL(rand()) / RAND_MAX * 2 - 1;
    REAL theta    = acos(costheta);

    REAL x = sin(theta) * cos(phi);
    REAL y = sin(theta) * sin(phi);
    REAL z = cos(theta);

    vec3f ret(x, y, z);
    ret.normalize();
    return ret;
}

REAL randDegree()
{
    return REAL(rand()) / RAND_MAX * 90;
}

inline float randAngle()
{
    return REAL(rand()) / RAND_MAX * M_PI * 2;
}

crigid* putMesh(triMesh* km, const vec3f& startPos, const vec3f& angle)
{
    crigid* body = new crigid(km, startPos, 4.f);
    body->getWorldTransform().setOrigin(startPos);
    body->getWorldTransform().setRotation(quaternion(angle.x, angle.y, angle.z));
    return body;
}

extern void getTetraBunnyData(unsigned int& numVtx, unsigned int& numTri, vec3f*& vtxs, tri3f*& tris, REAL scale, vec3f& shift);

triMesh* initVolBunny(REAL scale = 1.0, vec3f& shift = vec3f())
{
    unsigned int numVtx = 0, numTri = 0;
    vec3f*       vtxs = NULL;
    tri3f*       tris = NULL;

    getTetraBunnyData(numVtx, numTri, vtxs, tris, scale, shift);

    triMesh* vbunny = new triMesh(numVtx, numTri, tris, vtxs, false);
    g_scene.addMesh(vbunny);
    return vbunny;
}

void initModel(const char* c1file, const char* c2file)
{
    {
        triMesh* kmA = initVolBunny();
        triMesh* kmB = initVolBunny();  // initBunny(c2file);

        crigid* rigA = putMesh(kmA, vec3f(), vec3f());
        g_box += rigA->bound();
        crigid* rigB = putMesh(kmB, vec3f(-6, 0, 0), vec3f());
        g_box += rigB->bound();

        g_scene.addRigid(rigA);
        g_scene.addRigid(rigB);
        return;
    }
}

bool exportModel(const char* cfile)
{
    return true;
}

bool importModel(const char* cfile)
{
    return true;
}

void quitModel()
{
    g_scene.clear();
}

extern void beginDraw(aabb<REAL>&);
extern void endDraw();

void drawOther();

void drawBVH(int level)
{
    NULL;
}

void setMat(int i, int id);

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
    if (!g_box.empty())
        beginDraw(g_box);

    drawOther();
    g_scene.draw(level, tri, pnt, edge);

    drawBVH(level);

    if (!g_box.empty())
        endDraw();
}

extern double totalQuery;

bool dynamicModel(char*, bool, bool)
{
    static int st = 0;

    {
        {
            crigid* body = g_scene.getRigid(0);
            transf& trf  = body->getWorldTransform();
            vec3f   axis(1, 1, 1);
            axis.normalize();
            matrix3f rot  = matrix3f::rotation(axis, 0.01);
            matrix3f rold = trf.getBasis();
            matrix3f rnew = rold * rot;
            trf.setRotation(rnew);
        }


        {
            crigid* body = g_scene.getRigid(1);
            transf& trf  = body->getWorldTransform();
            vec3f   axis(-1, 1, 1);
            axis.normalize();
            matrix3f rot  = matrix3f::rotation(axis, 0.01);
            matrix3f rold = trf.getBasis();
            matrix3f rnew = rold * rot;
            trf.setRotation(rnew);
        }
        return true;
    }
}

extern std::vector<id_pair> minPairs;

void checkCollision()
{
    {
        double     tstart = omp_get_wtime();
        crigid*    bodyA  = g_scene.getRigid(0);
        crigid*    bodyB  = g_scene.getRigid(1);
        static int idx    = 0;

        g_scene.cdPairs.clear();
        int    broadNum = bodyA->checkCollision(bodyB, g_scene.cdPairs);
        double tdelta   = omp_get_wtime() - tstart;
        printf("checkCollison %d: (%zd/%d pairs) at %2.5f s\n", idx++, g_scene.cdPairs.size(), broadNum, tdelta);
        totalQuery += tdelta;
        return;
    }
}