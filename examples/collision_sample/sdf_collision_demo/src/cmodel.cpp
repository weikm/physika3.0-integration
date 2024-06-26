//#define B150 1

using namespace std;
#include <omp.h>
#include "crigid.h"
#include "sdf.h"

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "collision/internal/collision_transf.hpp"
#include "collision/interface/collidable_points.hpp"
#include "collision/interface/sdf_points_collision_solver.hpp"

class myTimer
{
	double t0;
	char msg[512];
public:
	myTimer(const char* msgIn) {
		t0 = omp_get_wtime();
		strcpy(msg, msgIn);
	}
	~myTimer() {
		double tdelta = omp_get_wtime() - t0;
		printf("%s: %2.5f s\n", msg, tdelta);
	}
};

class myTimer2
{
	double dt;
	char msg[512];
public:
	myTimer2(const char* msgIn) {
		dt = 0;
		strcpy(msg, msgIn);
	}

	void print() {
		printf("%s: %2.5f s\n", msg, dt);
	}

	void inc(double delta) {
		dt += delta;
	}
};

#define BUNNY_SCALE 1.f

#pragma warning(disable: 4996)


// global var for phyiska
auto sdf_scene  = Physika::World::instance().createScene();
Physika::Object* obj_bunny1 = Physika::World::instance().createObject();
auto obj_bunny2 = Physika::World::instance().createObject();
auto sdf_solver = Physika::World::instance().createSolver<Physika::SDFPointsCollisionSolver>();
vec3f*           initPosition;

extern void drawSdfPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void drawMinPair(crigid* r0, crigid* r1, std::vector<vec3f>&pairs);
extern void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs);
extern void drawRigid(crigid*, bool cyl, int level, vec3f &);

#define MAX_TRI_CLIPPING 16

class cscene {
	std::vector<kmesh*> _meshs;
	std::vector<crigid*> _rigids;
	std::vector<id_pair> _rigid_pairs;

	//for GPU updating...
	std::vector<transf> _trfs;

public:
	~cscene() { clear(); }

	void clear() {
		for (auto r : _rigids)
			delete r;

		for (auto m : _meshs)
			delete m;

		_meshs.clear();
		_rigids.clear();
	}

	FORCEINLINE crigid* getRigid(int rid) {
		return (rid < 0) ? nullptr : _rigids[rid];
	}

	FORCEINLINE int getRigidID(crigid* r) {
		return r == nullptr ? -1 : r->getID();
	}

	FORCEINLINE void setID()
	{
		for (int i = 0; i < _rigids.size(); i++) {
			_rigids[i]->setID(i);
		}
	}

	void draw(int level, bool showCD, bool showBody, bool showOnly) {
		if (showCD) {

			drawMinPair(_rigids[0], _rigids[1], minPairs);
			drawCDPair(_rigids[0], _rigids[1], cdPairs);
			drawSdfPair(_rigids[0], _rigids[1], sdfPairs);
		}

		if (showBody) {
			for (auto r : _rigids) {
				drawRigid(r, false, level, vec3f());
				if (showOnly)
					break;
			}
		}
	}

	void addMesh(kmesh* km) {
		_meshs.push_back(km);
	}

	void addRigid(crigid* rig) {
		_rigids.push_back(rig);
	}

	//for distance query
	std::vector<vec3f> minPairs;

	//for collision detection
	std::vector<id_pair> cdPairs, cdPairs2;

	//for SDF query
	std::vector<vec3f> sdfPairs;
} g_scene;

vec3f dPt0, dPt1, dPtw;

bool readobjfile(const char *path, 
				 unsigned int &numVtx, unsigned int &numTri, 
				 tri3f *&tris, vec3f *&vtxs, REAL scale, vec3f shift, bool swap_xyz, vec2f *&texs, tri3f *&ttris)
{
	vector<tri3f> triset;
	vector<vec3f> vtxset;
	vector<vec2f> texset;
	vector<tri3f> ttriset;

	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
				double x, y, z;
				sscanf(buf+2, "%lf%lf%lf", &x, &y, &z);

				if (swap_xyz)
					vtxset.push_back(vec3f(z, x, y)*scale+shift);
				else
					vtxset.push_back(vec3f(x, y, z)*scale+shift);
		} else

			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y;
				sscanf(buf + 3, "%lf%lf", &x, &y);

				texset.push_back(vec2f(x, y));
			}
			else
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2, id3=0;
				int tid0, tid1, tid2, tid3=0;
				bool quad = false;

				int count = sscanf(buf+2, "%d/%d", &id0, &tid0);
				char *nxt = strchr(buf+2, ' ');
				sscanf(nxt+1, "%d/%d", &id1, &tid1);
				nxt = strchr(nxt+1, ' ');
				sscanf(nxt+1, "%d/%d", &id2, &tid2);

				nxt = strchr(nxt+1, ' ');
				if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
					if (sscanf(nxt+1, "%d/%d", &id3, &tid3))
						quad = true;
				}

				id0--, id1--, id2--, id3--;
				tid0--, tid1--, tid2--, tid3--;

				triset.push_back(tri3f(id0, id1, id2));
				if (count == 2) {
					ttriset.push_back(tri3f(tid0, tid1, tid2));
				}

				if (quad) {
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
	vtxs = new vec3f[numVtx];
	for (unsigned int i=0; i<numVtx; i++)
		vtxs[i] = vtxset[i];

	int numTex = texset.size();
	if (numTex == 0)
		texs = NULL;
	else {
		texs = new vec2f[numTex];
		for (unsigned int i = 0; i < numTex; i++)
			texs[i] = texset[i];
	}

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i=0; i<numTri; i++)
		tris[i] = triset[i];

	int numTTri = ttriset.size();
	if (numTTri == 0)
		ttris = NULL;
	else {
		ttris = new tri3f[numTTri];
		for (unsigned int i = 0; i < numTTri; i++)
			ttris[i] = ttriset[i];
	}

	return true;
}

crigid* putMesh(kmesh* km, const vec3f& startPos, const vec3f &angle)
{
	crigid* body = new crigid(km, startPos, 4.f);
	body->getWorldTransform().setOrigin(startPos);
	body->getWorldTransform().setRotation(quaternion(angle.x, angle.y, angle.z));
	return body;
}

void myReplace(std::string& str,
	const std::string& oldStr,
	const std::string& newStr)
{
	std::string::size_type pos = 0u;
	while ((pos = str.find(oldStr, pos)) != std::string::npos) {
		str.replace(pos, oldStr.length(), newStr);
		pos += newStr.length();
	}
}

kmesh* initBunny(const char* ofile)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;
	vec2f* texs = NULL;
	tri3f* ttris = NULL;

	REAL scale = BUNNY_SCALE;
	vec3f shift(0, 0, 0);

	if (false == readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris)) {
		printf("loading %s failed...\n", ofile);
		exit(-1);
	}

	kmesh* bunny = new kmesh(numVtx, numTri, tris, vtxs, false);

	g_scene.addMesh(bunny);

	std::string sdfPath(ofile);
	myReplace(sdfPath, ".obj", ".sdf");
	DistanceField3D* sdf = new DistanceField3D(sdfPath);
	bunny->setSDF(sdf);

	return bunny;
}

void initModel(const char* c1file, const char* c2file)
{
	// data init for render
    kmesh* kmA = initBunny(c1file);
    kmesh* kmB = initBunny(c2file);

    crigid* rigA = putMesh(kmA, vec3f(), vec3f());

    crigid* rigB = putMesh(kmB, vec3f(-1.25, 0, 0), vec3f());

    g_scene.addRigid(rigA);
    g_scene.addRigid(rigB);


	// init data for physika solver
    sdf_scene->addObject(obj_bunny1);
    sdf_scene->addObject(obj_bunny2);
    sdf_scene->addSolver(sdf_solver);

	obj_bunny2->addComponent<Physika::CollidableSDFComponent>();
    obj_bunny2->getComponent<Physika::CollidableSDFComponent>()->m_sdf->loadSDF("my-bunny.sdf");
    //obj_bunny2->addComponent<Physika::CollidableTriangleMeshComponent>();

    obj_bunny1->addComponent<Physika::CollidablePointsComponent>();
    obj_bunny1->getComponent<Physika::CollidablePointsComponent>()->m_num = kmA->_num_vtx;
    obj_bunny1->getComponent<Physika::CollidablePointsComponent>()->m_pos = new Physika::vec3f[kmA->_num_vtx];
    initPosition                                                                    = new Physika::vec3f[kmA->_num_vtx];
    for (int i = 0; i < kmA->_num_vtx; i++)
    {
        initPosition[i] = kmA->_vtxs[i];
    }
	sdf_solver->attachObject(obj_bunny2);
    sdf_solver->attachObject(obj_bunny1);

	sdf_solver->initialize();

	return;
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

void drawOther();

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	drawOther();
	g_scene.draw(level, tri, pnt, edge);
}

extern double totalQuery;

bool dynamicModel(char*, bool, bool)
{
	crigid* body = g_scene.getRigid(1);
	transf& trf = body->getWorldTransform();
	vec3f axis(-1, 1, 1);
	axis.normalize();
	matrix3f rot = matrix3f::rotation(axis, 0.01);
	matrix3f rold = trf.getBasis();
	matrix3f rnew = rold * rot;
	trf.setRotation(rnew);
	return true;
}

void checkCollision()
{
	crigid* bodyA = g_scene.getRigid(0);
	crigid* bodyB = g_scene.getRigid(1);
	
	// apply transformation to the points
    g_scene.sdfPairs.clear();

    const transf& trfA   = bodyA->getTrf();
    const transf& trfB   = bodyB->getTrf();
    const transf  trfA2B = trfB.inverse() * trfA;

    kmesh* mA = bodyA->getMesh();
    kmesh* mB = bodyB->getMesh();

    vec3f*     Avtxs = mA->getVtxs();
    aabb<REAL> bx    = mB->bound();
    int        Anum  = mA->getNbVertices();
    for (int i = 0; i < Anum; i++)
    {
        vec3f& pt  = Avtxs[i];
        vec3f  ppt = trfA2B.getVertex(pt);
        obj_bunny1->getComponent<Physika::CollidablePointsComponent>()->m_pos[i] = pt;
    }
    //sdf_solver->run();
    //sdf_solver->getResult(g_scene.sdfPairs);
	bodyA->checkSdfCollision(bodyB, g_scene.sdfPairs);
	return;
}