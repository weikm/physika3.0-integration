// Utilities and system includes
#include <assert.h>
#include "vec3.cuh"
#include "mat3f.cuh"
#include "transf.cuh"
#include "box.cuh"

#include "bvh.cuh"
#include "tri3f.cuh"
#include "mesh.cuh"
#include "atomic.cuh"
#include "pair.cuh"

g_pair sTriPairs;
g_pair sIdPairs;
g_pair sRigPairs;

static int sMeshNum;
static g_mesh* sMeshes;

static int sRigidNum;
static g_transf* sRigids = NULL;

static  REAL* gmin;
static g_matrix3f* grotInv;

#define MAX_SBVH_NODE_NUM 4096 //=65536/16 //so we can put first 10 levels into the buffer
__constant__ g_sbvh_node cSbvhBuffer[MAX_SBVH_NODE_NUM];

#if 0
void checkBVH()
{
	int bvhNum = sMeshes[1]._num_bvh_nodes;
	g_bvh_node* data = new g_bvh_node[bvhNum];
	cudaMemcpy(data, sMeshes[1]._bvh_nodes, sizeof(g_bvh_node) * bvhNum, cudaMemcpyDeviceToHost);
	printf("here!");
	delete[] data;
}
#endif

void clearGPU()
{
	sTriPairs.destroy();
	sIdPairs.destroy();
	sRigPairs.destroy();

	for (int i=0; i<sMeshNum; i++)
		sMeshes[i].clear();

	delete[] sMeshes;

	if (sRigids)
		cudaFree(sRigids);
}

void pushRigids(int rigidNum, void* rigids)
{
	sRigidNum = rigidNum;
	
	if (sRigids == NULL)
		cudaMalloc(&sRigids, sizeof(g_transf) * sRigidNum);

	cudaMemcpy(sRigids, rigids, sizeof(g_transf)*sRigidNum, cudaMemcpyHostToDevice);
}

/// /////////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
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
#endif


__device__ void setMax2(REAL& a, const REAL& b)
{
	if (a < b)
	{
		a = b;
	}
}

__device__ void setMin2(REAL& a, const REAL& b)
{
	if (b < a)
	{
		a = b;
	}
}

__device__ void	setMax(REAL3& a, const REAL3& other)
{
	setMax2(a.x, other.x);
	setMax2(a.y, other.y);
	setMax2(a.z, other.z);
}

__device__ void	setMin(REAL3& a, const REAL3& other)
{
	setMin2(a.x, other.x);
	setMin2(a.y, other.y);
	setMin2(a.z, other.z);
}

__device__ void quantize_clamp(
	unsigned short* out,
	const REAL3& point,
	const REAL3& min_bound,
	const REAL3& max_bound,
	const REAL3& bvhQuantization)
{
	REAL3 clampedPoint = point;
	setMax(clampedPoint, min_bound);
	setMin(clampedPoint, max_bound);

	REAL3 v = clampedPoint - min_bound;
	out[0] = (unsigned short)(v.x * bvhQuantization.x + 0.5f);
	out[1] = (unsigned short)(v.y * bvhQuantization.y + 0.5f);
	out[2] = (unsigned short)(v.z * bvhQuantization.z + 0.5f);
}

CU_FORCEINLINE __device__ static void getTriangleVtxs(g_tri3f* tris, REAL3* vtxs, int fid, REAL3 &v0, REAL3 &v1, REAL3& v2)
{
	//if (fid == 8324)
	//	fid = 8324;

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

#if 0
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
	g_mesh& km = sMeshes[1];

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
	int num = sMeshes[1]._num_vtx;
	//BLK_PAR(num);
	int T = 32;
	int B = BPG(num, T);

	REAL3 projDirLcs = make_REAL3((REAL*)projDirLcsIn);
	REAL3 projDirLcsR = make_REAL3((REAL*)projDirLcsRIn);
	REAL3 off = make_REAL3((REAL*)offIn);
	g_mesh& km = sMeshes[0];

#if 1
	kernel_ray_casting2 << <B, T >> > (
		grotInv, off, projDirLcs, projDirLcsR, sMeshes[1]._vtxs,
		km._tris, km._vtxs, km._bvh_nodes,
		gmin, num);
	getLastCudaError("kernel_vtx_skinning");
#else
	kernel_ray_casting2 << <1, 1 >> > (
		grotInv, off, projDirLcs, projDirLcsR, sMeshes[1]._vtxs,
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
#endif

void getPairsGPU(void* data, int num, void *data2)
{
	sTriPairs.popData(data, num);
	if (data2)
		sIdPairs.popData(data2, num);
}

void pushRigidPairsGPU(int num, void* data)
{
	sRigPairs.pushData(num, data);
}

#define STACK_SIZE 64

CU_FORCEINLINE __device__ void addPair(
	uint a, uint b, uint2* pairs, uint* idx,
	uint2 rids, uint2* pairId)
{
	if (*idx < MAX_PAIR_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		pairs[offset].x = a;
		pairs[offset].y = b;

		if (pairId) {
			pairId[offset] = rids;
		}
	}
}

CU_FORCEINLINE __device__ void
boxQuery(int idx, const g_box& bx,
	g_tri3f* tris, REAL3* vtxs, g_bvh_node* root,
	uint2* pairInter, uint* pairInterIdx,
	uint2 rids, uint2* pairId)
{
	const g_bvh_node* stack[STACK_SIZE];
	const g_bvh_node* node = root;
	unsigned int stackPtr = 0;
#if 1
	while (1) {
		if (!node->box().overlaps(bx)) {
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		if (node->isLeaf()) {
			addPair(idx, node->triID(), pairInter, pairInterIdx, rids, pairId);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		stack[stackPtr++] = node->right();
		if (stackPtr > STACK_SIZE) { printf("Error: stack overflow...\n"); }
		node = node->left();
	}
#endif
}


CU_FORCEINLINE __device__ void
qboxQuery(int idx,
	unsigned short *minPt,
	unsigned short *maxPt,
	g_tri3f* tris, REAL3* vtxs, g_qbvh_node* root,
	uint2* pairInter, uint* pairInterIdx,
	uint2 rids, uint2* pairId)
{
	const g_qbvh_node* stack[STACK_SIZE];
	const g_qbvh_node* node = root;
	unsigned int stackPtr = 0;
#if 1
	while (1) {
		//if (!node->box().overlaps(bx)) {
		if (!node->testQuantizedBoxOverlapp(minPt, maxPt)) {
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		if (node->isLeaf()) {
			addPair(idx, node->triID(), pairInter, pairInterIdx, rids, pairId);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		stack[stackPtr++] = node->right();
		if (stackPtr > STACK_SIZE) { printf("Error: stack overflow...\n"); }
		node = node->left();
	}
#endif
}


CU_FORCEINLINE __device__ void
sboxQuery(int idx,
	unsigned short* minPt,
	unsigned short* maxPt,
	g_tri3f* tris, REAL3* vtxs,
	g_sbvh_node* upperNodes,
	g_sbvh_node* lowerNodes,
	int upperNum,
	int lowerNum,
	uint2* pairInter, uint* pairInterIdx,
	uint2 rids, uint2* pairId)
{
	const g_sbvh_node* stack[STACK_SIZE];
	const g_sbvh_node* node = upperNodes;
	unsigned int stackPtr = 0;
	while (1) {
		//if (!node->box().overlaps(bx)) {
		if (!node->testQuantizedBoxOverlapp(minPt, maxPt)) {
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		if (node->isLeaf()) {
			addPair(idx, node->triID(), pairInter, pairInterIdx, rids, pairId);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		stack[stackPtr++] = node->right(upperNodes, lowerNodes, upperNum);
		if (stackPtr > STACK_SIZE) { printf("Error: stack overflow...\n"); }
		node = node->left(upperNodes, lowerNodes, upperNum);
	}
}

__global__ void
kernel_tritriCD(g_matrix3f* rot, REAL3 off, g_tri3f* tris, REAL3* vtxs, g_bvh_node* bvh_nodes,
	REAL margin, uint2* pairInter, uint* pairInterIdx, int num,
	g_qbvh_node* qbvh_nodes,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2)
{
	LEN_CHK(num);
	REAL3 v0, v1, v2;
	getTriangleVtxs(tris, vtxs, idx, v0, v1, v2);

//	if (idx == 139)
//		idx = 139;

	g_box bx;
	bx.set((*rot) * v0 + off);
	bx.add((*rot) * v1 + off);
	bx.add((*rot) * v2 + off);
	bx.enlarge(margin);

	uint2 dummy;
	dummy.x = dummy.y = 0;
#if 0
	boxQuery(idx, bx, tris, vtxs, bvh_nodes, pairInter, pairInterIdx, dummy, nullptr);
	//rayCastingRecursive(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);
#else
	REAL3 bMin = make_REAL3(x0, y0, z0);
	REAL3 bMax = make_REAL3(x1, y1, z1);
	REAL3 quan = make_REAL3(x2, y2, z2);

	unsigned short minPt[3], maxPt[3];
	quantize_clamp(minPt, bx.minV(), bMin, bMax, quan);
	quantize_clamp(maxPt, bx.maxV(), bMin, bMax, quan);

	//_qbvh->query(minPt, maxPt, rets);
	qboxQuery(idx, minPt, maxPt, tris, vtxs, qbvh_nodes, pairInter, pairInterIdx, dummy, nullptr);
#endif


}

__global__ void
kernel_tritriCDFused(
	g_tri3f* tris, REAL3* vtxs,
	g_bvh_node* bvh_nodes,
	g_qbvh_node* qbvh_nodes, 
	g_sbvh_node *sbvh_upper_nodes,
	g_sbvh_node* sbvh_lower_nodes,
	int sbvh_upper_num,
	int sbvh_lower_num,
	REAL margin,
	uint2* pairInter, uint* pairInterIdx,
	uint2* pairId, uint* pairIdIdx,
	uint2* pairRig, uint* pairRigIdx,
	g_transf *rigs, int num,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2)
{
	LEN_CHK(num);
	REAL3 v0, v1, v2;
	getTriangleVtxs(tris, vtxs, idx, v0, v1, v2);

	REAL3 bMin = make_REAL3(x0, y0, z0);
	REAL3 bMax = make_REAL3(x1, y1, z1);
	REAL3 quan = make_REAL3(x2, y2, z2);

	int rigNum = *pairRigIdx;
	for (int i = 0; i < rigNum; i++)
	{
		uint2 rp = pairRig[i];
		g_transf& trfA = rigs[rp.x];
		g_transf& trfB = rigs[rp.y];
		g_transf trfA2B = trfB.inverse() * trfA;

		g_box bx;
		bx.set(trfA2B.apply(v0));
		bx.add(trfA2B.apply(v1));
		bx.add(trfA2B.apply(v2));
		bx.enlarge(margin);

#if 0
		boxQuery(idx, bx, tris, vtxs, bvh_nodes, pairInter, pairInterIdx, rp, pairId);
		//rayCastingRecursive(idx, vLcs, projDirLcs, projDirLcsR, tris, vtxs, bvh_nodes, gmin);
#else
		unsigned short minPt[3], maxPt[3];
		quantize_clamp(minPt, bx.minV(), bMin, bMax, quan);
		quantize_clamp(maxPt, bx.maxV(), bMin, bMax, quan);
		
#if 0
		//_qbvh->query(minPt, maxPt, rets);
		qboxQuery(idx, minPt, maxPt, tris, vtxs, qbvh_nodes, pairInter, pairInterIdx, rp, pairId);
#else
		sboxQuery(idx, minPt, maxPt, tris, vtxs,
			cSbvhBuffer, sbvh_lower_nodes,
			//sbvh_upper_nodes, sbvh_lower_nodes,
			sbvh_upper_num, sbvh_lower_num,
			pairInter, pairInterIdx, rp, pairId);
#endif

#endif
	}
}

CU_FORCEINLINE __device__ void
bvhQuery(int idx, const g_box2boxTrf& trf,
	g_tri3f* tris, REAL3* vtxs, g_bvh_node* root,
	uint2* pairInter, uint* pairInterIdx,
	uint2 rids, uint2* pairId)
{
	const g_bvh_node* stack[STACK_SIZE * 2];
	const g_bvh_node* nodeA = root;
	const g_bvh_node* nodeB = root;
	unsigned int stackPtr = 0;

	while (1) {
		if (!overlapping_trans_cache(nodeA->_box, nodeB->_box, trf, true)) {
			if (stackPtr == 0)
				break;
			else {
				nodeB = stack[--stackPtr];
				nodeA = stack[--stackPtr];
			}

			continue;
		}

		if (nodeA->isLeaf() && nodeB->isLeaf()) {
			addPair(nodeA->triID(), nodeB->triID(), pairInter, pairInterIdx, rids, pairId);

			if (stackPtr == 0)
				break;
			else {
				nodeB = stack[--stackPtr];
				nodeA = stack[--stackPtr];
			}

			continue;
		}

		if (nodeA->isLeaf()) {
			stack[stackPtr++] = nodeA;
			stack[stackPtr++] = nodeB->right();
			nodeB = nodeB->left();
		}
		else {
#if 0
			bvh_node* l = nodeA->left();
			bvh_node* r = nodeA->right();

			nodeA = l;
			stack[stackPtr++] = r;
			stack[stackPtr++] = nodeB;
#else
			if (nodeB->isLeaf()) {
				stack[stackPtr++] = nodeA->right();
				stack[stackPtr++] = nodeB;
				nodeA = nodeA->left();
			}
			else {
				stack[stackPtr++] = nodeA->right();
				stack[stackPtr++] = nodeB->left();
				stack[stackPtr++] = nodeA->left();
				stack[stackPtr++] = nodeB->right();
				stack[stackPtr++] = nodeA->right();
				stack[stackPtr++] = nodeB->right();
				nodeA = nodeA->left();
				nodeB = nodeB->left();
			}

#endif

		}

		if (stackPtr > STACK_SIZE * 2) {
			stackPtr = 0;
			printf("Error: stack overflow at %d...\n", idx);
		}
	}
}

__global__ void
kernel_RigidRigidCD(
	g_tri3f* tris, REAL3* vtxs, g_bvh_node* bvh_nodes, REAL margin,
	uint2* pairInter, uint* pairInterIdx,
	uint2* pairId, uint* pairIdIdx,
	uint2* pairRig, uint* pairRigIdx,
	g_transf* rigs, int num)
{
	LEN_CHK(num);
	uint2 rp = pairRig[idx];

	g_transf& trfA = rigs[rp.x];
	g_transf& trfB = rigs[rp.y];
	g_box2boxTrf trfBtoA;

	trfBtoA.calc_from_homogenic(trfA, trfB);
	bvhQuery(idx, trfBtoA, tris, vtxs, bvh_nodes, pairInter, pairInterIdx, rp, pairId);
}

int checkRigidRigidCDGPU(REAL margin)
{
	int num = sRigPairs._length;
	int T = 32;
	int B = BPG(num, T);

	sTriPairs.clear();
	sIdPairs.clear();

	g_mesh& km = sMeshes[0];
	kernel_RigidRigidCD << <B, T >> > (
		km._tris, km._vtxs, km._bvh_nodes, margin,
		sTriPairs._dPairs, sTriPairs._dIdx,
		sIdPairs._dPairs, sIdPairs._dIdx,
		sRigPairs._dPairs, sRigPairs._dIdx,
		sRigids, num);
	getLastCudaError("kernel_RigidRigidCD");

	return sTriPairs.length();
}

int checkTriTriCDFusedGPU(REAL margin,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2)
{
	int num = sMeshes[0]._num_tri;
	int T = 32;
	int B = BPG(num, T);

	sTriPairs.clear();
	sIdPairs.clear();

	g_mesh& km = sMeshes[0];
	kernel_tritriCDFused << <B, T >> > (
		km._tris, km._vtxs, km._bvh_nodes,
		km._qbvh_nodes,
		km._sbvh_upper_nodes,
		km._sbvh_lower_nodes,
		km._sbvh_upper_num,
		km._sbvh_lower_num,
		margin,
		sTriPairs._dPairs, sTriPairs._dIdx,
		sIdPairs._dPairs, sIdPairs._dIdx,
		sRigPairs._dPairs, sRigPairs._dIdx,
		sRigids, num,
		x0, y0, z0, x1, y1, z1, x2, y2, z2);
	getLastCudaError("kernel_tritriCDFused");

	return sTriPairs.length();

}

int checkTriTriCDGPU(const void* offIn, const void* rotIn, REAL margin,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2)
{
	//input
	cudaMemcpy(grotInv, rotIn, sizeof(g_matrix3f), cudaMemcpyHostToDevice);
	REAL3 off = make_REAL3((REAL*)offIn);
	g_mesh& km = sMeshes[0];

	//2. for each triangle
	int num = sMeshes[0]._num_tri;
	//BLK_PAR(num);
	int T = 32;// 32;
	int B = BPG(num, T);

	sTriPairs.clear();

	kernel_tritriCD <<<B, T >>> (
		grotInv, off, km._tris, km._vtxs, km._bvh_nodes, margin, sTriPairs._dPairs, sTriPairs._dIdx, num,
		km._qbvh_nodes, x0, y0, z0, x1, y1, z1, x2, y2, z2);
	getLastCudaError("kernel_vtx_skinning");

	return sTriPairs.length();
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


 /* This sample queries the properties of the CUDA devices present in the system
  * via CUDA Runtime API. */

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

int* pArgc = NULL;
char** pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T* attribute, CUdevice_attribute device_attribute,
	int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(
			stderr,
			"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int device_query_main(int argc, char** argv) {
	pArgc = &argc;
	pArgv = argv;

	printf("%s Starting...\n\n", argv[0]);
	printf(
		" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);
		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
			memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize);
		}

#endif

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total shared memory per multiprocessor:        %zu bytes\n",
			deviceProp.sharedMemPerMultiprocessor);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %zu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %zu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Managed Memory:                %s\n",
			deviceProp.managedMemory ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char* sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown", NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	// If there are 2 or more GPUs, query to determine whether RDMA is supported
	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

			// Only boards based on Fermi or later can support P2P
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				// on Windows (64-bit), the Tesla Compute Cluster driver for windows
				// must be enabled to support this
				&& prop[i].tccDriver
#endif
				) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		// Show all the combinations of support P2P GPUs
		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
						prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
						can_access_peer ? "Yes" : "No");
				}
			}
		}
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	printf("%s", sProfileString.c_str());

	printf("Result = PASS\n");

	// finish
	exit(EXIT_SUCCESS);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@

void pushMesh1(int meshNum)
{
	printf("gpu size = %zd, %zd, %zd, %zd, %zd, %zd, %zd\n",
		sizeof(g_box), sizeof(REAL3), sizeof(g_matrix3f), 
		sizeof(g_bvh_node), sizeof(g_transf), sizeof(g_qbvh_node), sizeof(g_qbvh_node));
	reportMemory("pushMesh1");

	cudaFuncSetCacheConfig(kernel_tritriCDFused, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernel_tritriCD, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernel_RigidRigidCD, cudaFuncCachePreferL1);

	sTriPairs.init(MAX_PAIR_NUM); // MAX_PAIR_NUM);
	sIdPairs.init(MAX_PAIR_NUM);
	sRigPairs.init(1024 * 4);

	sMeshNum = meshNum;
	sMeshes = new g_mesh[meshNum];

	//some data initialization
	cudaMalloc(&gmin, sizeof(REAL));
	REAL dummy = -99.9;
	cudaMemcpy(gmin, &dummy, sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMalloc(&grotInv, sizeof(g_matrix3f));
}

void pushMesh2(int i, int numVtx, int numTri, void* tris, void* vtxs, int fNum, void* bvhNodes, void *qbvhNodes,
	void *sbvhUpperNodes, void *sbvhLowerNodes, int sbvhUpperNum, int sbvhLowerNum)
{
	g_mesh& m = sMeshes[i];
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

	cudaMalloc(&m._qbvh_nodes, sizeof(g_qbvh_node) * bvhNum);
	cudaMemcpy(m._qbvh_nodes, qbvhNodes, sizeof(g_qbvh_node) * bvhNum, cudaMemcpyHostToDevice);

	printf("######################################\n");
	printf("BVH vs QBVH: %zd - %zd\n", sizeof(g_bvh_node) * bvhNum, sizeof(g_qbvh_node) * bvhNum);
	printf("######################################\n");

	if (sbvhUpperNum != MAX_SBVH_NODE_NUM) {
		printf("Something is wrong about sbvh!\n");
		exit(0);
	}

	m._sbvh_upper_num = sbvhUpperNum;
	m._sbvh_lower_num = sbvhLowerNum;
	cudaMemcpyToSymbol(cSbvhBuffer, sbvhUpperNodes, sizeof(g_sbvh_node)* sbvhUpperNum);

	m._sbvh_upper_nodes = cSbvhBuffer;
	if (m._sbvh_lower_num) {
		cudaMalloc(&m._sbvh_lower_nodes, sizeof(g_sbvh_node) * sbvhLowerNum);
		cudaMemcpy(m._sbvh_lower_nodes, sbvhLowerNodes, sizeof(g_sbvh_node) * sbvhLowerNum, cudaMemcpyHostToDevice);
	}

#if 0
	size_t bvhSize = sizeof(g_bvh_node) * bvhNum;
	printf("The BVH needs %zd bytes...\n", bvhSize);
	exit(0);
#endif

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