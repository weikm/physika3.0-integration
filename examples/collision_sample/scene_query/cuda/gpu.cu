#include "def.cuh"
#include "vec3.cuh"
#include "box.cuh"
#include "mat3f.cuh"
#include "bvh.cuh"
#include "tri3f.cuh"
#include "mesh.cuh"
#include "atomic.cuh"

static g_mesh* meshes;
static  REAL* gmin;
static g_matrix3f* grotInv;

static int vsetNum;
static REAL3* vsetData;

#if 0
static g_rigid* rigids;

void pushRigid1(int num)
{
	rigids = new g_rigid[num];
	for (int i = 0; i < num; i++) {
		cudaMalloc(&rigids[i]._rot, sizeof(g_matrix3f));
		rigids[i]._mesh = 1;
	}
	rigids[0]._mesh = 0;
	reportMemory("pushRigid1");
}


void pushRigid2(int i, void* rot, REAL* off)
{
	g_rigid& r = rigids[i];

	r._off.x = off[0];
	r._off.y = off[1];
	r._off.z = off[2];

	cudaMemcpy(r._rot, rot, sizeof(g_matrix3f), cudaMemcpyHostToDevice);
	getCudaError("pushRigid2");
}

void checkBVH()
{
	int bvhNum = meshes[1]._num_bvh_nodes;
	g_bvh_node* data = new g_bvh_node[bvhNum];
	cudaMemcpy(data, meshes[1]._bvh_nodes, sizeof(g_bvh_node) * bvhNum, cudaMemcpyDeviceToHost);
	printf("here!");
	delete[] data;
}
#endif

void pushVtxSet(int num, void* data)
{
	vsetNum = num;
	cudaMalloc(&vsetData, sizeof(REAL3) * num);
	cudaMemcpy(vsetData, data, sizeof(REAL3) * num, cudaMemcpyHostToDevice);
}

void pushMesh1(int meshNum)
{
	printf("gpu size = %zd, %zd, %zd, %zd\n",
		sizeof(g_box), sizeof(REAL3), sizeof(g_matrix3f), sizeof(g_bvh_node));
	reportMemory("pushMesh1");

	meshes = new g_mesh[meshNum];

//some data initialization
	cudaMalloc(&gmin, sizeof(REAL));
	REAL dummy = -99.9;
	cudaMemcpy(gmin, &dummy, sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMalloc(&grotInv, sizeof(g_matrix3f));
}

void pushMesh2(int i, int numVtx, int numTri, void* tris, void* vtxs, int fNum, void* bvhNodes)
{
	g_mesh& m = meshes[i];
	m._num_vtx = numVtx;
	m._num_tri = numTri;
	cudaMalloc(&m._tris, sizeof(g_tri3f) * numTri);
	cudaMemcpy(m._tris, tris, sizeof(g_tri3f) * numTri, cudaMemcpyHostToDevice);
	cudaMalloc(&m._vtxs, sizeof(REAL3) * numVtx);
	cudaMemcpy(m._vtxs, vtxs, sizeof(REAL3) * numVtx, cudaMemcpyHostToDevice);

	int bvhNum = fNum * 2 - 1;
	m._num_bvh_nodes = bvhNum;
	cudaMalloc(&m._bvh_nodes, sizeof(g_bvh_node) * bvhNum);
	cudaMemcpy(m._bvh_nodes, bvhNodes, sizeof(g_bvh_node) * bvhNum, cudaMemcpyHostToDevice);
	getCudaError("pushMesh2");
	reportMemory("pushMesh2");

#if 0
	{
		g_bvh_node* data = new g_bvh_node[bvhNum];
		cudaMemcpy(data, m._bvh_nodes, sizeof(g_bvh_node) * bvhNum, cudaMemcpyDeviceToHost);
		printf("here!");
		delete[] data;
	}
#endif

}

/// /////////////////////////////////////////////////////////////////////////////////////////////////////

CU_FORCEINLINE __device__ static bool RayIntersectsTriangle(REAL3& ro, REAL3& rv, REAL3& t0, REAL3& t1, REAL3& t2, REAL& ret)
{
	const REAL EPSILON = 0.0000001;
	REAL3& vertex0 = t0;
	REAL3& vertex1 = t1;
	REAL3& vertex2 = t2;
	REAL3 edge1, edge2, h, s, q;
	REAL a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = cross(rv, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.

	f = 1.0 / a;
	s = ro - vertex0;
	u = f * dot(s, h);

	if (u < 0.0 || u > 1.0)
		return false;

	q = cross(s, edge1);
	v = f * dot(rv, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	REAL t = f * dot(edge2, q);

	if (t > EPSILON) // ray intersection
	{
		ret = t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

//ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
CU_FORCEINLINE __device__ static REAL IntersectAABB2(g_box & bx, REAL3& rayO, REAL3& rayD, REAL3& rayRD, const REAL& rayT)
{
	REAL3& bmin = bx._min;
	REAL3& bmax = bx._max;

	REAL tx1 = (bmin.x - rayO.x) * rayRD.x, tx2 = (bmax.x - rayO.x) * rayRD.x;
	REAL tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	REAL ty1 = (bmin.y - rayO.y) * rayRD.y, ty2 = (bmax.y - rayO.y) * rayRD.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	REAL tz1 = (bmin.z - rayO.z) * rayRD.z, tz2 = (bmax.z - rayO.z) * rayRD.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < rayT && tmax > 0) return tmin; else return 1e30f;
}

CU_FORCEINLINE __device__ static REAL IntersectAABB(g_box& bx, REAL3& rayO, REAL3& rayD, const REAL& rayT)
{
	REAL3& bmin = bx._min;
	REAL3& bmax = bx._max;

	REAL tx1 = (bmin.x - rayO.x) / rayD.x, tx2 = (bmax.x - rayO.x) / rayD.x;
	REAL tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	REAL ty1 = (bmin.y - rayO.y) / rayD.y, ty2 = (bmax.y - rayO.y) / rayD.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	REAL tz1 = (bmin.z - rayO.z) / rayD.z, tz2 = (bmax.z - rayO.z) / rayD.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));

	if (tmax >= tmin && tmin < rayT && tmax > 0)
		return tmin;
	else
		return 1e30f;
}

CU_FORCEINLINE __device__ static void getTriangleVtxs(g_tri3f* tris, REAL3* vtxs, int fid, REAL3 &v0, REAL3 &v1, REAL3& v2)
{
	g_tri3f& f = tris[fid];
	v0 = vtxs[f.id0()];
	v1 = vtxs[f.id1()];
	v2 = vtxs[f.id2()];
}

#if 0
//failed to run
__device__ void
rayCastingRecursive(int idx,
	REAL3 pt, REAL3 dir, REAL3 dirR,
	g_tri3f* tris, REAL3* vtxs, g_bvh_node* root, REAL* gmin)
{
	g_bvh_node* node = root;

	if (node->isLeaf())
	{
		int id = node->triID();
		REAL3 v0, v1, v2;
		getTriangleVtxs(tris, vtxs, id, v0, v1, v2);

		REAL t;
		if (RayIntersectsTriangle(pt, dir, v0, v1, v2, t)) {
			atomicMin(gmin, t);
		}

		return;
	}

	g_bvh_node* l = node->left();
	g_bvh_node* r = node->right();

	REAL distl = IntersectAABB(l->box(), pt, dir, *gmin);
	REAL distr = IntersectAABB(r->box(), pt, dir, *gmin);
	if (distl > distr) {
		if (distr != 1e30f)
			rayCastingRecursive(idx, pt, dir, dirR, tris, vtxs, r, gmin);
		if (distl != 1e30f)
			rayCastingRecursive(idx, pt, dir, dirR, tris, vtxs, l, gmin);
	}
	else {
		if (distl != 1e30f)
			rayCastingRecursive(idx, pt, dir, dirR, tris, vtxs, l, gmin);
		if (distr != 1e30f)
			rayCastingRecursive(idx, pt, dir, dirR, tris, vtxs, r, gmin);
	}
}
#endif

#define STACK_SIZE 32

__device__ void
rayCasting(int idx,
	 REAL3 pt, REAL3 dir, REAL3 dirR,
	 g_tri3f* tris, REAL3* vtxs, g_bvh_node*root, REAL *gmin)
{
	g_bvh_node* stack[STACK_SIZE];
	g_bvh_node* node = root;
	unsigned int stackPtr = 0;

	while (1) {
		if (node->isLeaf()) {
			int id = node->triID();
			REAL3 v0, v1, v2;
			getTriangleVtxs(tris, vtxs, id, v0, v1, v2);

			REAL t;
			if (RayIntersectsTriangle(pt, dir, v0, v1, v2, t)) {
				atomicMin(gmin, t);
			}

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		g_bvh_node* l = node->left();
		g_bvh_node* r = node->right();

#if 0
		REAL distl = IntersectAABB2(l->box(), pt, dir, dirR, *gmin);
		REAL distr = IntersectAABB2(r->box(), pt, dir, dirR, *gmin);
#else
		REAL distl = IntersectAABB(l->box(), pt, dir, *gmin);
		REAL distr = IntersectAABB(r->box(), pt, dir, *gmin);
#endif

		if (distl > distr) {
			g_bvh_node* tn = l;
			l = r;
			r = tn;
			REAL td = distl;
			distl = distr;
			distr = td;
		}
		if (distl == 1e30f)
		{
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
		}
		else
		{
			node = l;
			if (distr != 1e30f) {
				stack[stackPtr++] = r;
				if (stackPtr > STACK_SIZE) {
					printf("Error: stack overflow...\n");
				}
			}
		}
	}
}

__global__ void
kernel_ray_casting1(
	g_matrix3f *rotInv, REAL3 off, REAL3 projDirLcs, REAL3 projDirLcsR,
	REAL3 *vset, g_tri3f *tris, REAL3 *vtxs, g_bvh_node *bvh_nodes,
	REAL *gmin, int num)
{
	LEN_CHK(num);
	REAL3 v = vset[idx];
	REAL3 vLcs = (*rotInv) * (v - off); //correct

	rayCasting(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);
	//rayCastingRecursive(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);
}

__global__ void
kernel_ray_casting2(
	g_matrix3f* rotInv, REAL3 off, REAL3 projDirLcs, REAL3 projDirLcsR,
	REAL3* vset, g_tri3f* tris, REAL3* vtxs, g_bvh_node* bvh_nodes,
	REAL* gmin, int num)
{
	LEN_CHK(num);
	REAL3 v = vset[idx];
	REAL3 vLcs = (*rotInv) * v + off;

	rayCasting(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);
	//rayCastingRecursive(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);

}

void checkVtxRayGPU1(int rid, const void *rotInv, void *offIn,
	void *projDirLcsIn, void *projDirLcsRIn, REAL &dMin)
{
	getLastCudaError("1");

	//input
	cudaMemcpy(gmin, &dMin, sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(grotInv, rotInv, sizeof(g_matrix3f), cudaMemcpyHostToDevice);

	//2. for each vtx
	int num = vsetNum;
	//BLK_PAR(num);
	int T = 32;
	int B = BPG(num, T);

	REAL3 projDirLcs = make_REAL3((REAL*)projDirLcsIn);
	REAL3 projDirLcsR = make_REAL3((REAL*)projDirLcsRIn);
	REAL3 off = make_REAL3((REAL*)offIn);
	g_mesh& km = meshes[1];

#if 0
	{
		int bvhNum = km._num_bvh_nodes;
		g_bvh_node* data = new g_bvh_node[bvhNum];
		cudaMemcpy(data, km._bvh_nodes, sizeof(g_bvh_node) * bvhNum, cudaMemcpyDeviceToHost);
		printf("here!");
		delete[] data;
	}
#endif

#if 1
	kernel_ray_casting1 << <B, T >> > (
		grotInv, off, projDirLcs, projDirLcsR, vsetData,
		km._tris, km._vtxs, km._bvh_nodes,
		gmin, num);
	getLastCudaError("kernel_vtx_skinning");
#else
	kernel_ray_casting1 << <1, 1 >> > (
		grotInv, off, projDirLcs, projDirLcsR, vsetData,
		km._tris, km._vtxs, km._bvh_nodes,
		gmin, num);
	getLastCudaError("kernel_vtx_skinning");
#endif

#if 0
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
	}
#endif

	//output
	cudaMemcpy(&dMin, gmin, sizeof(REAL), cudaMemcpyDeviceToHost);
	getLastCudaError("2");

}

void checkVtxRayGPU2(int rid, const void* rotInv, void* offIn,
	void* projDirLcsIn, void* projDirLcsRIn, REAL& dMin)
{
	getLastCudaError("1");

	//input
	cudaMemcpy(gmin, &dMin, sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(grotInv, rotInv, sizeof(g_matrix3f), cudaMemcpyHostToDevice);

	//2. for each vtx
	int num = meshes[1]._num_vtx;
	//BLK_PAR(num);
	int T = 32;
	int B = BPG(num, T);

	REAL3 projDirLcs = make_REAL3((REAL*)projDirLcsIn);
	REAL3 projDirLcsR = make_REAL3((REAL*)projDirLcsRIn);
	REAL3 off = make_REAL3((REAL*)offIn);
	g_mesh& km = meshes[0];

#if 1
	kernel_ray_casting2 << <B, T >> > (
		grotInv, off, projDirLcs, projDirLcsR, meshes[1]._vtxs,
		km._tris, km._vtxs, km._bvh_nodes,
		gmin, num);
	getLastCudaError("kernel_vtx_skinning");
#else
	kernel_ray_casting2 << <1, 1 >> > (
		grotInv, off, projDirLcs, projDirLcsR, meshes[1]._vtxs,
		km._tris, km._vtxs, km._bvh_nodes,
		gmin, num);
	getLastCudaError("kernel_vtx_skinning");
#endif

#if 0
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
	}
#endif

	//output
	cudaMemcpy(&dMin, gmin, sizeof(REAL), cudaMemcpyDeviceToHost);
	getLastCudaError("2");

}

static REAL gg_start;
static GPUTimer2 gg;

void gpuTimerBegin()
{
	gg_start = omp_get_wtime();
	gg.tick();
}

float gpuTimerEnd(char* msg, bool verb)
{
	float gpuT = gg.tock2();
	REAL tmp_timing_finish = omp_get_wtime();
	REAL  tmp_timing_duration = tmp_timing_finish - gg_start;
	if (verb)
		printf("%s: %2.5f s (%3.5f ms) \n", msg, tmp_timing_duration, gpuT);
	return gpuT;
}
