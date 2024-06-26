#if defined(_WIN32)
#include <Windows.h>
#endif

#include <GL/gl.h>

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "tmbvh.hpp"
#include <climits>
#include <utility>
using namespace std;

static vector<kmesh *>  *ptCloth;

void self_mesh(vector<kmesh *> &meshes)
{
	ptCloth = &meshes;
}

class aap {
public:
	char _xyz;
	REAL _p;

	FORCEINLINE aap(const BOX &total) {
		vec3f center = total.center();
		char xyz = 2;

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		} else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

			_xyz = xyz;
			_p = center[xyz];
	}

	FORCEINLINE bool inside(const vec3f &mid) const {
		return mid[_xyz]>_p;
	}
};

bvh::bvh(std::vector<kmesh*> &ms, REAL tol)
{
	_num = 0;
	_nodes = NULL;

	construct(ms, tol);
	reorder();
	resetParents(); //update the parents after reorder ...
}

static vec3f *s_fcenters;
static BOX *s_fboxes;
static unsigned int *s_idx_buffer;
static bvh_node *s_current;

void bvh::construct(std::vector<kmesh*> &ms, REAL tol)
{
	BOX total;

	for (int i=0; i<ms.size(); i++)
		for (int j = 0; j<ms[i]->_num_vtx; j++) {
			total += ms[i]->_vtxs[j];
		}

	_num = 0;
	for (int i=0; i<ms.size(); i++)
		_num += ms[i]->_num_tri;

	s_fcenters = new vec3f[_num];
	s_fboxes = new BOX[_num];

	int tri_idx = 0;
	int vtx_offset = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->_num_tri; j++) {
			tri3f &f = ms[i]->_tris[j];
			vec3f &p1 = ms[i]->_vtxs[f.id0()];
			vec3f &p2 = ms[i]->_vtxs[f.id1()];
			vec3f &p3 = ms[i]->_vtxs[f.id2()];

			s_fboxes[tri_idx] += p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;
			s_fboxes[tri_idx].enlarge(tol);

			s_fcenters[tri_idx] = (p1 + p2 + p3) / REAL(3.0);
			tri_idx++;
		}
		vtx_offset += ms[i]->_num_vtx;
	}

	aap pln(total);
	s_idx_buffer = new unsigned int[_num];
	unsigned int left_idx = 0, right_idx = _num;

	tri_idx = 0;
	for (int i=0; i<ms.size(); i++)
		for (int j = 0; j<ms[i]->_num_tri; j++) {
		if (pln.inside(s_fcenters[tri_idx]))
			s_idx_buffer[left_idx++] = tri_idx;
		else
			s_idx_buffer[--right_idx] = tri_idx;

		tri_idx++;
	}

	_nodes = new bvh_node[_num*2-1];
	_nodes[0]._box = total;
	s_current = _nodes+3;

	if (_num == 1)
		_nodes[0]._child = 0;
	else {
		_nodes[0]._child = -1;

		if (left_idx == 0 || left_idx == _num)
			left_idx = _num/2;

		_nodes[0].left()->construct(s_idx_buffer, left_idx);
		_nodes[0].right()->construct(s_idx_buffer+left_idx, _num-left_idx);
	}

	delete [] s_idx_buffer;
	delete [] s_fcenters;

	refit();
	//delete[] s_fboxes;
}

void bvh::refit(std::vector<kmesh*> &ms, REAL tol)
{
	assert(s_fboxes);

	int tri_idx = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->_num_tri; j++) {
			tri3f &f = ms[i]->_tris[j];
			vec3f &p1 = ms[i]->_vtxs[f.id0()];
			vec3f &p2 = ms[i]->_vtxs[f.id1()];
			vec3f &p3 = ms[i]->_vtxs[f.id2()];

			s_fboxes[tri_idx] = p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;
			s_fboxes[tri_idx].enlarge(tol);

			tri_idx++;
		}
	}

	refit();
}

void bvh::resetParents()
{
	root()->resetParents(root());
}

void bvh::refit()
{
	root()->refit();
}

#include <queue>
using namespace std;

void bvh::reorder()
{
	if (true) 
	{
		queue<bvh_node *> q;

		// We need to perform a breadth-first traversal to fill the ids

		// the first pass get idx for each node ...
		int *buffer = new int[_num*2-1];
		int idx = 0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			buffer[node-_nodes] = idx++;
			q.pop();

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		// the 2nd pass, get right nodes ...
		bvh_node *new_nodes = new bvh_node[_num*2-1];
		idx=0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			q.pop();

			new_nodes[idx] = *node;
			if (!node->isLeaf()) {
				int loc = node->left()-_nodes;
				new_nodes[idx]._child = idx-buffer[loc];
			}
			idx++;

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		delete [] buffer;
		delete [] _nodes;
		_nodes = new_nodes;
	}
}

void
bvh_node::refit()
{
	if (isLeaf()) {
		_box = s_fboxes[_child];

	} else {
		left()->refit();
		right()->refit();

		_box = left()->_box + right()->_box;
	}
}

void
bvh_node::resetParents(bvh_node *root)
{
	if (this == root)
		setParent(-1);

	if (isLeaf())
		return;

	left()->resetParents(root);
	right()->resetParents(root);

	left()->setParent(this - root);
	right()->setParent(this - root);
}


void
bvh_node::construct(unsigned int id)
{
	_child = id;
	_box = s_fboxes[id];
}

void
bvh_node::construct(unsigned int *lst, unsigned int num)
{
	for (unsigned int i=0; i<num; i++)
		_box += s_fboxes[lst[i]];

	if (num == 1) {
		_child = lst[0];
		return;
	}

	// try to split them
	_child = int(this-s_current);
	s_current += 2;

	if (num == 2) {
		left()->construct(lst[0]);
		right()->construct(lst[1]);
		return;
	}

	aap pln(_box);
	unsigned int left_idx=0, right_idx=num-1;
	for (unsigned int t=0; t<num; t++) {
		int i=lst[left_idx];

		if (pln.inside( s_fcenters[i]))
			left_idx++;
		else {// swap it
			unsigned int tmp=lst[left_idx];
			lst[left_idx] = lst[right_idx];
			lst[right_idx--] = tmp;
		}
	}

	int half = num/2;

	if (left_idx == 0 || left_idx == num) {
		left()->construct(lst, half);
		right()->construct(lst+half, num-half);
	} else {
		left()->construct(lst, left_idx);
		right()->construct(lst+left_idx, num-left_idx);
	}
}

#if 0
void
bvh_node::sprouting(bvh_node *other, front_list &append, vector<tri_pair> &ret)
{
	if (isLeaf() && other->isLeaf()) {

		if (!covertex(triID(), other->triID())) {
			append.push_back(front_node(this, other, 0));

			if (_box.overlaps(other->_box))
				ret.push_back(tri_pair(triID(), other->triID()));
		}

		return;
	}

	if (!_box.overlaps(other->_box)) {
		append.push_back(front_node(this, other, 0));
		return;
	}

	if (isLeaf()) {
		sprouting(other->left(), append, ret);
		sprouting(other->right(), append, ret);
	}
	else {
		left()->sprouting(other, append, ret);
		right()->sprouting(other, append, ret);
	}
}
#endif

void mesh_id(int id, vector<kmesh *> &m, int &mid, int &fid)
{
	fid = id;
	for (mid=0; mid<m.size(); mid++)
		if (fid < m[mid]->_num_tri) {
			return;
		} else {
			fid -= m[mid]->_num_tri;
		}

	assert(false);
	fid = -1;
	mid = -1;
	printf("mesh_id error!!!!\n");
	abort();
}

bool covertex(int id1, int id2)
{
	if ((*ptCloth).empty())
		return false;

	int mid1, fid1, mid2, fid2;

	mesh_id(id1, *ptCloth, mid1, fid1);
	mesh_id(id2, *ptCloth, mid2, fid2);

	if (mid1 != mid2)
		return false;

	tri3f &f1 = (*ptCloth)[mid1]->_tris[fid1];
	tri3f &f2 = (*ptCloth)[mid2]->_tris[fid2];

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			if (f1.id(i) == f2.id(2))
				return true;

	return false;
}

#if 0
void
front_list::propogate(vector<kmesh *> &c, vector<tri_pair> &ret)
{
	self_mesh(c);

	front_list append;

	for (vector<front_node>::iterator it = begin(); it != end(); it++) {
		(*it).update(append, ret);
	}

	insert(end(), append.begin(), append.end());
}

void front_node::update(front_list &append, vector<tri_pair> &ret)
 {
	 if (_flag != 0)
		 return;

	 if (_left->isLeaf() && _right->isLeaf()) {
		 if (!covertex(_left->triID(), _right->triID()) &&
			 _left->box().overlaps(_right->box()))
			 ret.push_back(tri_pair(_left->triID(), _right->triID()));

		 return;
	 }

	 if (!_left->box().overlaps(_right->box()))
		 return;

	 // need to be spouted
	 _flag = 1; // set to be invalid

	 if (_left->isLeaf()) {
		 _left->sprouting(_right->left(), append, ret);
		 _left->sprouting(_right->right(), append, ret);
	 } else {
		 _left->left()->sprouting(_right, append, ret);
		 _left->right()->sprouting(_right, append, ret);
	 }
 }


extern void refitBVH(bool);
extern void pushBVH(unsigned int length, int *ids, bool isCloth);
extern void pushBVHLeaf(unsigned int length, int *idf, bool isCloth);
extern void pushBVHIdx(int max_level, unsigned int *level_idx, bool isCloth);

void bvh::push2GPU(bool isCloth)
{
#ifdef USE_GPU
	unsigned int length = _num*2-1;
	int *ids = new int[length*2];

	for (unsigned int i=0; i<length; i++) {
		ids[i] = (root()+i)->triID();
		ids[length+i] = (root()+i)->parentID();
	}

	pushBVH(length, ids, isCloth);
	delete [] ids;

	unsigned int leafNum = 0;
	int *idf = new int[_num];
	for (unsigned int i = 0; i < length; i++) {
		if ((root() + i)->isLeaf()) {
			int idx = (root() + i)->triID();
			idf[idx] = i;
			leafNum++;
		}
	}
	assert(leafNum == _num);
	pushBVHLeaf(leafNum, idf, isCloth);
	delete []idf;

	{// push information for refit
		int max_level = 0;
		root()->getLevel(0, max_level);
		max_level++;

		unsigned int *level_idx = new unsigned int [max_level];
		unsigned int *level_buffer = new unsigned int [max_level];
		for (int i=0; i<max_level; i++)
			level_idx[i] = level_buffer[i] = 0;

		root()->getLevelIdx(0, level_buffer);
		for (int i=1; i<max_level; i++)
			for (int j=0; j<i; j++)
				level_idx[i] += level_buffer[j];

		delete [] level_buffer;
		pushBVHIdx(max_level, level_idx, isCloth);
		delete [] level_idx;
	}

	refitBVH(isCloth);
#endif
}

extern void pushFront(bool, int, unsigned int *);

void 
front_list::push2GPU(bvh_node *r1, bvh_node *r2 )
{
#ifdef USE_GPU
	bool self = (r2 == NULL);

	if (r2 == NULL)
		r2 = r1;

	int num = size();
	if (num) {
		int idx = 0;
		unsigned int *buffer = new unsigned int [num*4];
		for (vector<front_node>::iterator it=begin();
			it != end(); it++)
		{
			front_node n = *it;
			buffer[idx++] = n._left - r1;
			buffer[idx++] = n._right-r2;
			buffer[idx++] = 0;
			buffer[idx++] = n._ptr;
		}

		pushFront(self, num, buffer);
		delete [] buffer;
	} else
		pushFront(self, 0, NULL);
#endif
}
#endif

void
bvh_node::visualize(int level)
{
	if (isLeaf()) {
		_box.visualize();
	}
	else
		if ((level > 0)) {
			if (level == 1) {
				_box.visualize();
			}
			else {
				if (left()) left()->visualize(level - 1);
				if (right()) right()->visualize(level - 1);
			}
		}
}

#if 0
void IntersectBVH(Ray& ray)
{
	BVHNode* node = &bvhNode[rootNodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri(ray, tri[triIdx[node->leftFirst + i]]);
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		REAL dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
		REAL dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
		if (dist1 > dist2) { swap(dist1, dist2); swap(child1, child2); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}
#endif

void bvh::rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret)
{
	root()->rayCasting(pt, dir, km, ret);
}

//bvh_node* pR;
void bvh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, kmesh* km, REAL& ret)
{
	//pR = root();
	root()->rayCasting2(pt, dir, dirR, km, ret);
}

bool RayIntersectsTriangle(const vec3f &ro, const vec3f &rv, const vec3f &t0, const vec3f &t1, const vec3f &t2, REAL &ret)
{
	const REAL EPSILON = 0.0000001;
	const vec3f &vertex0 = t0;
	const vec3f &vertex1 = t1;
	const vec3f &vertex2 = t2;
	vec3f edge1, edge2, h, s, q;
	REAL a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = rv.cross(edge2);
	a = edge1.dot(h);

	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.

	f = 1.0 / a;
	s = ro - vertex0;
	u = f * s.dot(h);

	if (u < 0.0 || u > 1.0)
		return false;

	q = s.cross(edge1);
	v = f * rv.dot(q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	REAL t = f * edge2.dot(q);

	if (t > EPSILON) // ray intersection
	{
		ret = t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

//ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
REAL IntersectAABB2(BOX& bx, const vec3f& rayO, const vec3f& rayD, const vec3f &rayRD, const REAL& rayT)
{
	const vec3f& bmin = bx.getMin();
	const vec3f& bmax = bx.getMax();

	REAL tx1 = (bmin.x - rayO.x) * rayRD.x, tx2 = (bmax.x - rayO.x) * rayRD.x;
	REAL tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	REAL ty1 = (bmin.y - rayO.y) * rayRD.y, ty2 = (bmax.y - rayO.y) * rayRD.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	REAL tz1 = (bmin.z - rayO.z) * rayRD.z, tz2 = (bmax.z - rayO.z) * rayRD.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < rayT && tmax > 0) return tmin; else return 1e30f;
}

REAL IntersectAABB(BOX &bx, const vec3f&rayO, const vec3f&rayD, const REAL& rayT)
{
	const vec3f &bmin = bx.getMin();
	const vec3f &bmax = bx.getMax();

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

void getTriangleVtxs(kmesh*km, int fid, vec3f& v0, vec3f& v1, vec3f& v2)
{
	tri3f& f = km->_tris[fid];
	v0 = km->_vtxs[f.id0()];
	v1 = km->_vtxs[f.id1()];
	v2 = km->_vtxs[f.id2()];
}

void bvh_node::rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret)
{
	if (isLeaf())
	{
		int id = triID();
		vec3f v0, v1, v2;
		getTriangleVtxs(km, id, v0, v1, v2);

		REAL t;
		if (RayIntersectsTriangle(pt, dir, v0, v1, v2, t)) {
			if (ret > t)
				ret = t;
		}

		return;
	}

	bvh_node* l = this->left();
	bvh_node* r = this->right();

	REAL distl = IntersectAABB(l->box(), pt, dir, ret);
	REAL distr = IntersectAABB(r->box(), pt, dir, ret);
	if (distl > distr) {
		if (distr != 1e30f)
			r->rayCasting(pt, dir, km, ret);
		if (distl != 1e30f)
			l->rayCasting(pt, dir, km, ret);
	}
	else {
		if (distl != 1e30f)
			l->rayCasting(pt, dir, km, ret);
		if (distr != 1e30f)
			r->rayCasting(pt, dir, km, ret);
	}
}

void bvh_node::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, kmesh* km, REAL& ret)
{
	if (isLeaf())
	{
		int id = triID();
		vec3f v0, v1, v2;
		getTriangleVtxs(km, id, v0, v1, v2);

		REAL t;
		if (RayIntersectsTriangle(pt, dir, v0, v1, v2, t)) {
			if (ret > t)
				ret = t;
		}

		return;
	}

	bvh_node* l = this->left();
	bvh_node* r = this->right();

	REAL distl = IntersectAABB2(l->box(), pt, dir, dirR, ret);
	REAL distr = IntersectAABB2(r->box(), pt, dir, dirR, ret);
	if (distl > distr) {
		if (distr != 1e30f)
			r->rayCasting2(pt, dir, dirR, km, ret);
		if (distl != 1e30f)
			l->rayCasting2(pt, dir, dirR, km, ret);
	}
	else {
		if (distl != 1e30f)
			l->rayCasting2(pt, dir, dirR, km, ret);
		if (distr != 1e30f)
			r->rayCasting2(pt, dir, dirR, km, ret);
	}
}


#define STACK_SIZE 64

void bvh::rayCasting3(const vec3f& pt, const vec3f& dir, const vec3f& dirR, kmesh* km, REAL& ret)
{
	//pR = root();

	bvh_node* stack[STACK_SIZE];
	bvh_node* node = root();
	unsigned int stackPtr = 0;

	while (1) {
		if (node->isLeaf()) {
			int id = node->triID();
			vec3f v0, v1, v2;
			getTriangleVtxs(km, id, v0, v1, v2);

			REAL t;
			if (RayIntersectsTriangle(pt, dir, v0, v1, v2, t)) {
				if (ret > t)
					ret = t;
			}

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];

			continue;
		}

		bvh_node* l = node->left();
		bvh_node* r = node->right();

		REAL distl = IntersectAABB2(l->box(), pt, dir, dirR, ret);
		REAL distr = IntersectAABB2(r->box(), pt, dir, dirR, ret);
		if (distl > distr) {
			std::swap(l, r);
			std::swap(distl, distr);
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