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

class cscene
{
    std::vector<triMesh*>    _meshs;
    std::vector<crigid*>   _rigids;
    std::vector<cplane*>   _plns;
    std::vector<manifold*> _predictiveManifolds;
    std::vector<id_pair>   _rigid_pairs, _rigid_pln_pairs;

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

        for (auto p : _plns)
            delete p;

        for (auto m : _meshs)
            delete m;

        _meshs.clear();
        _plns.clear();
        _rigids.clear();
    }

    FORCEINLINE crigid* getRigid(int rid)
    {
        return (rid < 0) ? nullptr : _rigids[rid];
    }

    FORCEINLINE const cplane* getPlane(int pid)
    {
        return (pid < 0) ? nullptr : _plns[pid];
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

        for (int i = 0; i < _plns.size(); i++)
            _plns[i]->setID(i);
    }

    void draw(int level, bool showCD, bool showBody, bool showOnly)
    {
        if (showCD)
        {
            drawCDPair(_rigids[1], _rigids[2], cdPairs);
            drawSdfPair(_rigids[0], _rigids[1], sdfPairs);
            drawCDPair(_rigids[2], _rigids[3], cdPairs2);
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

    void addPlane(cplane* pl)
    {
        _plns.push_back(pl);
    }

    void addRigid(crigid* rig)
    {
        _rigids.push_back(rig);
    }

    bool output(const char* fname)
    {
        FILE* fp = fopen(fname, "wt");
        if (fp == NULL)
            return false;

        fprintf(fp, "%zd\n", _rigids.size());
        for (int i = 0; i < _rigids.size(); i++)
        {
            transf&    trf = _rigids[i]->getWorldTransform();
            vec3f&     off = trf.getOrigin();
            quaternion q   = trf.getRotation();
            fprintf(fp, "%lf, %lf, %lf\n", off.x, off.y, off.z);
            fprintf(fp, "%lf, %lf, %lf, %lf\n", q.x(), q.y(), q.z(), q.w());
        }
        fclose(fp);
        return true;
    }

    bool input(const char* fname)
    {
        FILE* fp = fopen(fname, "rt");
        if (fp == NULL)
            return false;

        int  num = 0;
        char buffer[512];
        fgets(buffer, 512, fp);
        sscanf(buffer, "%d", &num);
        if (num != _rigids.size())
            return false;

        for (int i = 0; i < _rigids.size(); i++)
        {
            transf& trf = _rigids[i]->getWorldTransform();

            fgets(buffer, 512, fp);
            double x, y, z, w;
            sscanf(buffer, "%lf, %lf, %lf", &x, &y, &z);
            vec3f off(x, y, z);
            fgets(buffer, 512, fp);
            sscanf(buffer, "%lf, %lf, %lf, %lf", &x, &y, &z, &w);
            quaternion q(x, y, z, w);

            trf.setOrigin(off);
            trf.setRotation(q);
        }
        fclose(fp);
        return true;
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

void myReplace(std::string&       str,
               const std::string& oldStr,
               const std::string& newStr)
{
    std::string::size_type pos = 0u;
    while ((pos = str.find(oldStr, pos)) != std::string::npos)
    {
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
}

triMesh* initBunny(const char* ofile)
{
    unsigned int numVtx = 0, numTri = 0;
    vec3f*       vtxs  = NULL;
    tri3f*       tris  = NULL;
    vec2f*       texs  = NULL;
    tri3f*       ttris = NULL;

    REAL  scale = BUNNY_SCALE;
    vec3f shift(0, 0, 0);

    if (false == readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris))
    {
        printf("loading %s failed...\n", ofile);
        exit(-1);
    }

    triMesh* bunny = new triMesh(numVtx, numTri, tris, vtxs, false);

    g_scene.addMesh(bunny);

    //std::string sdfPath(ofile);
    //myReplace(sdfPath, ".obj", ".sdf");
    //DistanceField3D* sdf = new DistanceField3D(sdfPath);
    //bunny->setSDF(sdf);

    return bunny;
}

void initModel(const char* c1file, const char* c2file)
{
#ifdef FOR_MIXED
    {
        triMesh* kmA = initBunny(c1file);
        triMesh* kmB = initVolBunny(0.25);

        crigid* rigA = putMesh(kmA, vec3f(1.25, 0, 0), vec3f());

        crigid* rigB = putMesh(kmA, vec3f(), vec3f());

        crigid* rigC = putMesh(kmA, vec3f(-1.25, 0, 0), vec3f());

        crigid* rigD = putMesh(kmB, vec3f(-2.5, 0, 0), vec3f());

        g_scene.addRigid(rigA);
        g_scene.addRigid(rigB);
        g_scene.addRigid(rigC);
        g_scene.addRigid(rigD);
        return;
    }
#endif
}

bool exportModel(const char* cfile)
{
    return g_scene.output(cfile);
}

bool importModel(const char* cfile)
{
    bool ret = g_scene.input(cfile);
    return ret;
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

    drawOther();
    g_scene.draw(level, tri, pnt, edge);

    drawBVH(level);
}

extern double totalQuery;

bool dynamicModel(char*, bool, bool)
{
    static int st = 0;
    //	if (st == 29)
    //		printf("here!\n");


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

        {
            crigid* body = g_scene.getRigid(2);
            transf& trf  = body->getWorldTransform();
            vec3f   axis(1, -1, 1);
            axis.normalize();
            matrix3f rot  = matrix3f::rotation(axis, 0.01);
            matrix3f rold = trf.getBasis();
            matrix3f rnew = rold * rot;
            trf.setRotation(rnew);
        }

        {
            crigid* body = g_scene.getRigid(3);
            transf& trf  = body->getWorldTransform();
            vec3f   axis(1, 1, -1);
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
    //double  tstart = omp_get_wtime();
    //crigid* bodyA  = g_scene.getRigid(0);
    //crigid* bodyB  = g_scene.getRigid(1);
    //crigid* bodyC  = g_scene.getRigid(2);
    //crigid* bodyD  = g_scene.getRigid(3);

    //g_scene.sdfPairs.clear();
    //bodyA->checkSdfCollision(bodyB, g_scene.sdfPairs);

    //g_scene.cdPairs.clear();
    //bodyB->checkCollision(bodyC, g_scene.cdPairs);

    //g_scene.cdPairs2.clear();
    //bodyC->checkCollision(bodyD, g_scene.cdPairs2);

    //double tdelta = omp_get_wtime() - tstart;
    //printf("checkCollison (%zd pairs) at %2.5f s\n", g_scene.cdPairs.size(), tdelta);
    //totalQuery += tdelta;
    return;
}