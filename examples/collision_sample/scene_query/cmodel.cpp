//#define B150 1

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "crigid.h"
#include "pair.h"
#include "plane.h"
#include "tmbvh.hpp"
#include <stdio.h>
#include <omp.h>

#pragma warning(disable: 4996)

extern void drawRigid(crigid*, bool cyl, int level, vec3f &);
extern void drawPlans();

#ifdef GPU
extern void pushMesh1(int);
extern void pushVtxSet(int num, void* data);
//extern void pushRigid1(int);
//extern void checkBVH();

extern void gpuTimerBegin();
extern float gpuTimerEnd(char*, bool);

#endif

BOX g_box;
BOX g_projBx;
REAL g_time = 0.0f;

extern bool verb;

std::vector<int> fset;
std::set<int> vset;
std::set<id_pair> eset;
std::vector<vec3f> vtxset;

vec3f projDir(0.0f, -1.0f, 0.0f);
REAL maxDist = 20.0;

class cscene {
	std::vector<kmesh*> _meshs;
	std::vector<cplane *> _plns;
	std::vector<crigid *> _rigids;
	std::vector<bool> _active;
	std::vector<REAL> _qryRets;

public:
	~cscene() { clear(); }

	void clear() {
		for (auto r : _rigids)
			delete r;

		for (auto p : _plns)
			delete p;

		for (auto m : _meshs)
			delete m;

		_meshs.clear();
		_plns.clear();
		_rigids.clear();
		_active.clear();
	}

	REAL checkVtxRay(crigid*);
	void checkVtxRayInv(crigid*, crigid*, REAL&);

	void sweepQueryOld() {
		_qryRets.clear();

		std::vector<crigid*> act;
		std::vector<int> ids;

		//broad phrase
		_active.clear();
		int id = 0;
		for (auto r : _rigids) {
			BOX bx = r->bound();
			if (bx.overlaps(g_projBx)) {
				act.push_back(r);
				ids.push_back(id);
			}

			id++;
			_active.push_back(true);
		}

		//narrow phrase
		//_active[0] = true;
		//_active[ids[1]] = true;

		int num = act.size();

#pragma omp parallel for
		for (int i=1; i<num; i++) {
			crigid* r = act[i];

			REAL dMin = checkVtxRay(r);
			checkVtxRayInv(r, act[0], dMin);
			if (dMin < maxDist) {
				if (verb)
					printf("hit distance = %lf\n", dMin);

				_qryRets.push_back(dMin);//unsafe for multiThread
			}
		}
	}

	void drawQryRets()
	{
		for (auto t : _qryRets) {
			vec3f off = projDir * t;
			drawRigid(_rigids[0], false, -1, off);
		}
	}

	void draw(int level) {
		int i = 0;
		for (auto r : _rigids) {
			if (_active.size() == 0 || _active[i] == true)
				drawRigid(r, r == _rigids[0], level, vec3f());

			i++;
		}
		//drawPlans();
		drawQryRets();
	}

	void addMesh(kmesh*km) {
		_meshs.push_back(km);
	}

	void addPlane(cplane* pl) {
		_plns.push_back(pl);
	}

	void addRigid(crigid* rig) {
		_rigids.push_back(rig);
	}

	void updateRigids() {
		g_time += 0.005f;

		REAL amplitude = 4.0f;
		int numObjs = _rigids.size()-1;
		for (int i = 0; i < numObjs; i++) {
			crigid* rig = _rigids[i + 1];
			const REAL coeff = REAL(i);
			const REAL phase = M_PI * 2.0f * REAL(i) / REAL(numObjs);
			REAL z = 0.0f;
			REAL y = sinf(phase + g_time * 1.17f) * amplitude;
			REAL x = cosf(phase + g_time * 1.17f) * amplitude;

			matrix3f rtx = matrix3f::rotation(vec3f(1, 0, 0), g_time + coeff);
			matrix3f rty = matrix3f::rotation(vec3f(0, 1, 0), g_time * 1.17f + coeff);
			matrix3f rtz = matrix3f::rotation(vec3f(0, 0, 1), g_time * 0.33f + coeff);
			matrix3f rot = rtx * rty * rtz;

			rig->updatePos(rot, vec3f(x, y, z));
			g_box += rig->bound();
		}

#ifdef GPU
		update2GPU();
#endif
	}

#ifdef GPU
	void push2GPU()
	{
		printf("cpu size = %zd, %zd, %zd, %zd\n",
			sizeof(BOX), sizeof(vec3f), sizeof(matrix3f), sizeof(bvh_node));

		pushMesh1(_meshs.size());

		for (int i = 0; i < _meshs.size(); i++) {
			_meshs[i]->push2G(i);
		}

#if 0
		pushRigid1(_rigids.size());
		for (int i = 0; i < _rigids.size(); i++) {
			_rigids[i]->push2G(i);
		}
#endif
		pushVtxSet(vtxset.size(), vtxset.data());
	}

	void update2GPU()
	{
#if 0
		for (int i = 0; i < _rigids.size(); i++) {
			_rigids[i]->push2G(i);
		}
		//checkBVH();
#endif
	}

	void checkVtxRayGPU(crigid* rig, int i, REAL& dMin);
	void checkVtxRayInvGPU(crigid* rig, crigid *obj, int i, REAL& dMin);

	void sweepQueryGPU() {
		_qryRets.clear();

		//broad phrase
		int num = _rigids.size();
		crigid* o = _rigids[0];
		for (int i = 1; i < num; i++) {
			crigid* r = _rigids[i];

			BOX bx = r->bound();
			if (!bx.overlaps(g_projBx)) {
				//broad prhase culling
				continue;
			}

			//narrow phrase
			REAL dMin = maxDist;

			//dMin = checkVtxRay(r);
			checkVtxRayGPU(r, i, dMin);
			checkVtxRayInvGPU(r, o, i, dMin);

			if (dMin < maxDist) {
				if (verb)
					printf("hit distance = %lf\n", dMin);

				_qryRets.push_back(dMin);//unsafe for multiThread
			}
		}
	}
#endif

	void sweepQuery() {
		_qryRets.clear();

		//broad phrase
		int num = _rigids.size();
		crigid* o = _rigids[0];
		for (int i = 1; i < num; i++) {
			crigid* r = _rigids[i];

			BOX bx = r->bound();
			if (!bx.overlaps(g_projBx)) {
				//broad prhase culling
				continue;
			}

			//narrow phrase
			REAL dMin = checkVtxRay(r);
			checkVtxRayInv(r, o, dMin);

			if (dMin < maxDist) {
				if (verb)
					printf("hit distance = %lf\n", dMin);

				_qryRets.push_back(dMin);//unsafe for multiThread
			}
		}
	}
} g_scene;

vec3f dPt0, dPt1, dPtw;

void
cscene::checkVtxRayInv(crigid* obj, crigid* rig, REAL& dMin)
{
	//1. get the prjDir in r's local coordinate system
	matrix3f rot = rig->getTrf();
	matrix3f rotInv = rot.getInverse();

	vec3f p0 = rotInv * vec3f();
	vec3f p1 = rotInv * -projDir; //here inversed!
	vec3f projDirLcs = p1 - p0;
	projDirLcs.normalize();
	vec3f projDirLcsR(1.0 / projDirLcs.x, 1.0 / projDirLcs.y, 1.0 / projDirLcs.z);

	//2. for each vtx
	vec3f off = rig->getOffset();
	kmesh* km = rig->getMesh();

	kmesh* kmObj = obj->getMesh();
	matrix3f rotObj = obj->getTrf();
	vec3f offObj = obj->getOffset();

	matrix3f sumRot = rotInv * rotObj;
	vec3f sumOff = rotInv * (offObj - off);

	for (int i = 0; i < kmObj->_num_vtx; i++) {
		vec3f vo = kmObj->_vtxs[i];
#if 0
		vec3f vw = rotObj * vo + offObj;
		vec3f vLcs = rotInv * (vw - off);
#else
		vec3f vLcs = sumRot * vo + sumOff;
#endif

		//km->rayCasting(vLcs, projDirLcs, dMin);
		km->rayCasting2(vLcs, projDirLcs, projDirLcsR, dMin);

#if 0
		dPt0 = vLcs;
		dPt1 = vLcs + projDirLcs * maxDist;
		dPtw = vw;
		return;
#endif
	}
}

REAL
cscene::checkVtxRay(crigid* rig)
{
	//1. get the prjDir in r's local coordinate system
	matrix3f rot = rig->getTrf();
	matrix3f rotInv = rot.getInverse();

	vec3f p0 = rotInv * vec3f();
	vec3f p1 = rotInv * projDir;
	vec3f projDirLcs = p1 - p0;
	projDirLcs.normalize();
	vec3f projDirLcsR(1.0 / projDirLcs.x, 1.0 / projDirLcs.y, 1.0 / projDirLcs.z);

	//2. for each vtx
	vec3f off = rig->getOffset();
	kmesh* km = rig->getMesh();

	REAL dMin = maxDist;
	int num = vtxset.size();
	for (int i = 0; i < num; i++) {
		vec3f& v = vtxset[i];
		//for (auto v : vtxset) {
		vec3f vLcs = rotInv * (v - off); //correct
		//vec3f vLcs = (v - off) * rotInv; //wrong!!!!

		//km->rayCasting(vLcs, projDirLcs, dMin);
		km->rayCasting2(vLcs, projDirLcs, projDirLcsR, dMin);

#if 0
		dPt0 = vLcs;
		dPt1 = vLcs + projDirLcs * maxDist;
		return dMin;
#endif
	}

	return dMin;
}


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

//http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
vec3f randDir()
{
	REAL phi = REAL(rand()) / RAND_MAX * M_PI * 2;
	REAL costheta = REAL(rand()) / RAND_MAX*2-1;
	REAL theta = acos(costheta);

	REAL x = sin(theta)*cos(phi);
	REAL y = sin(theta)*sin(phi);
	REAL z = cos(theta);

	vec3f ret(x, y, z);
	ret.normalize();
	return ret;
}

REAL randDegree()
{
	return  REAL(rand()) / RAND_MAX * 90;
}

kmesh* initCyl()
{
	const REAL width = 3.0f;
	const REAL radius = 0.5f;
	int splits = 64;

	vec3f* points = new vec3f[2 * splits];
	for (int i = 0; i < splits; i++)
	{
		const REAL cosTheta = cos(i * M_PI * 2.0f / REAL(splits));
		const REAL sinTheta = sin(i * M_PI * 2.0f / REAL(splits));
		const REAL y = radius * cosTheta;
		const REAL z = radius * sinTheta;
		points[2 * i + 0] = vec3f(-width / 2.0f, y, z);
		points[2 * i + 1] = vec3f(+width / 2.0f, y, z);
	}

	int numTri = splits * 2;
	tri3f* tris = new tri3f[numTri];

	int idx = 0;
	for (int i = 0; i < splits-1; i++)
	{
		tris[idx++] = tri3f(i * 2, i * 2 + 1, i * 2 + 2);
		tris[idx++] = tri3f(i * 2+2, i * 2 + 1, i * 2 + 3);
	}
	tris[idx++] = tri3f(splits * 2 - 2, splits * 2 - 1, 0);
	tris[idx++] = tri3f(0, splits * 2 - 1, 1);

	//reverse all
	for (int i = 0; i < splits * 2; i++) {
		tri3f& t = tris[i];
		t.reverse();
	}

	kmesh* cyl = new kmesh(splits*2, numTri, tris, points, true);
	g_scene.addMesh(cyl);
	return cyl;
}

kmesh* initBunny(const char* ofile)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;
	vec2f* texs = NULL;
	tri3f* ttris = NULL;

	REAL scale = 1.;
	vec3f shift(0, 0, 0);

	if (false == readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris)) {
		printf("loading %s failed...\n", ofile);
		exit(-1);
	}
#if 0
	for (int i = 0; i < 10; i++) {
		vec3f& p = vtxs[i];
		printf("%lf, %lf, %lf\n", p.x, p.y, p.z);
	}
#endif

	kmesh* bunny = new kmesh(numVtx, numTri, tris, vtxs, false);
	g_scene.addMesh(bunny);
	return bunny;
}

void putCyl(kmesh* km, BOX &projBx)
{
	const vec3f origin(0.0f, 10.0f, 0.0f);
	crigid* rig = new crigid(km, origin);
	g_box += rig->bound();
	g_scene.addRigid(rig);

	{
		matrix3f rot = rig->getTrf();
		vec3f off = rig->getOffset();

		matrix3f rotInv = rot.getInverse();

		vec3f p0 = rotInv * vec3f();
		vec3f p1 = rotInv * projDir;
		vec3f invProj = p1 - p0;
		kmesh* m = rig->getMesh();
		vec3f* fnrms = m->getFNrms();
		tri3f* tris = m->getTris();
		for (int i = 0; i < m->getNbFaces(); i++) {
			if (fnrms[i].dot(invProj) > 0) {
				fset.push_back(i);

				tri3f& t = tris[i];
				vset.insert(t.id0());
				vset.insert(t.id1());
				vset.insert(t.id2());

				eset.insert(id_pair(t.id0(), t.id1()));
				eset.insert(id_pair(t.id1(), t.id2()));
				eset.insert(id_pair(t.id2(), t.id0()));
			}
		}

		for (auto v : vset) {
			vec3f vt = m->_vtxs[v];
			vec3f vtw = vt * rot + off;
			vtxset.push_back(vtw);
		}

		m->updateDL(true, -1);
	}
	{
		BOX bx = rig->bound();
		std::vector<vec3f> crns;
		bx.getCorners(crns);

		for (int i = 0; i < crns.size(); i++) {
			vec3f& p = crns[i];
			vec3f pp = p + projDir * maxDist;
			bx += pp;
		}
		projBx = bx;
	}
}

void putBunny(kmesh* km)
{
	int numObjs = 14;
	for (int i = 0; i < numObjs; i++) {
		crigid* rig = new crigid(km);
		g_scene.addRigid(rig);
	}
}

void initRigids(kmesh* km)
{
	int idx = 0;
	REAL dx = 0, dy = 0, dz = 0;
	aabb bx = km->bound();
	vec3f off;
#ifdef B150
	dx = bx.width() * 1.2;
	dy = bx.height() * 1.2;
	dz = bx.depth() * 1.2;
	off = vec3f(-dx*5*0.5, -dy*5*0.5, dz * 6*0.6);
#else

	dx = bx.width() * 1.5;
	dy = bx.height() * 1.5;
	dz = bx.depth() * 1.5;
//	off = vec3f(-dx*3*0.4, -dy*3*0.4, dz * 3*0.6);
	off = vec3f(-dx * 0.5, -dy *  0.5, dz * 1.6);
#endif

#ifdef B150
	//150 bunnys
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 6; k++)
#else
/*
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
*/
				for (int i = 0; i < 2; i++)
					for (int j = 0; j < 2; j++)
						for (int k = 0; k < 1; k++)

#endif
			{
				vec3f rot = randDir();
				REAL theta = randDegree(); // 30.0;

				crigid* rig = new crigid(km, vec3f(i * dx, j * dy, k * dz)+off, rot, theta);
				g_box += rig->bound();
				g_scene.addRigid(rig);
			}
}

void initPlanes()
{
	g_scene.addPlane(new cplane(vec3f(0, 1, 1), 0));
	g_scene.addPlane(new cplane(vec3f(0, -1, 1), 0));
	g_scene.addPlane(new cplane(vec3f(1, 0, 1), 0));
	g_scene.addPlane(new cplane(vec3f(-1, 0, 1), 0));
}

void initModel(const char *cfile)
{
	kmesh* kmC = initCyl();
	kmesh* kmB = initBunny(cfile);

	//initRigids(kmC);
	//initPlanes();

	putCyl(kmC, g_projBx);
	putBunny(kmB);

#ifdef GPU
	g_scene.push2GPU();
#endif

	g_scene.updateRigids();


}

void quitModel()
{
	g_scene.clear();
}

extern void beginDraw(BOX &);
extern void endDraw();

void drawOther();

void drawBVH(int level) {
	NULL;
}

void setMat(int i, int id);

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	if (!g_box.empty())
		beginDraw(g_box);

	drawOther();
	g_scene.draw(level);
#if 0
	for (int i=0; i<150; i++)
		if (lions[i]) {
			setMat(i, midBunny);

#if 1
			if (!pnt) {
				vec3f off = lions[i]->_off;
				vec3f axis = lions[i]->_axis;
				REAL theta = lions[i]->_theta;

				useBunnyDL(off.x, off.y, off.z, axis.x, axis.y, axis.z, theta);
			}
			else
#endif
				lions[i]->display(tri, false, false, level, true, i == 0 ? lion_set : dummy_set, dummy_vtx, i);
		}
#endif

	drawBVH(level);

	if (!g_box.empty())
		endDraw();
}

bool dynamicModel(char*, bool, bool)
{
	g_scene.updateRigids();
	return true;
}

extern double totalQuery;
extern bool verb;
void sweepQuery()
{
#ifdef GPU
	gpuTimerBegin();
	g_scene.sweepQueryGPU();
	totalQuery += gpuTimerEnd("sweepMultipleGPU", verb);
#else

	double now = omp_get_wtime();
	g_scene.sweepQuery();
	double delta = omp_get_wtime() - now;

	if (verb)
		printf("##sweepMultipleCPU: %3.5f s\n", delta);

	totalQuery += delta;
#endif
}

#ifdef GPU
extern void checkVtxRayGPU1(int, const void *, void *, void *, void *, REAL&);
extern void checkVtxRayGPU2(int, const void*, void*, void*, void*, REAL&);

void
cscene::checkVtxRayGPU(crigid* rig, int i, REAL &dMin)
{
	//1. get the prjDir in r's local coordinate system
	matrix3f rot = rig->getTrf();
	matrix3f rotInv = rot.getInverse();

	vec3f p0 = rotInv * vec3f();
	vec3f p1 = rotInv * projDir;
	vec3f projDirLcs = p1 - p0;
	projDirLcs.normalize();
	vec3f projDirLcsR(1.0 / projDirLcs.x, 1.0 / projDirLcs.y, 1.0 / projDirLcs.z);

	//2. for each vtx
	vec3f off = rig->getOffset();
	checkVtxRayGPU1(i, rotInv.asColMajor(), off.v, projDirLcs.v, projDirLcsR.v, dMin);
}

void
cscene::checkVtxRayInvGPU(crigid* obj, crigid *rig, int i, REAL& dMin)
{
	//1. get the prjDir in r's local coordinate system
	matrix3f rot = rig->getTrf();
	matrix3f rotInv = rot.getInverse();

	vec3f p0 = rotInv * vec3f();
	vec3f p1 = rotInv * -projDir;
	vec3f projDirLcs = p1 - p0;
	projDirLcs.normalize();
	vec3f projDirLcsR(1.0 / projDirLcs.x, 1.0 / projDirLcs.y, 1.0 / projDirLcs.z);

	//2. for each vtx
	vec3f off = rig->getOffset();
	kmesh* km = rig->getMesh();

	kmesh* kmObj = obj->getMesh();
	matrix3f rotObj = obj->getTrf();
	vec3f offObj = obj->getOffset();

	matrix3f sumRot = rotInv * rotObj;
	vec3f sumOff = rotInv * (offObj - off);

	checkVtxRayGPU2(i, sumRot.asColMajor(), sumOff.v, projDirLcs.v, projDirLcsR.v, dMin);
}

#endif

