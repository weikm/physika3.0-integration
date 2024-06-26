#pragma once

#include <float.h>
#include <stdlib.h>
#include "real.hpp"
#include "vec3f.h"
//#include "cmesh.h"
#include "crigid.h"
#include "pair.h"
#include "box.h"

#include <vector>
using namespace std;

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

class bvh;
class bvh_node;
class front_list;
class kmesh;
class linear_bvh;
class linear_bvh_node;

class alignas(16) bvh_node {
	BOX _box;
	int _child; // >=0 leaf with tri_id, <0 left & right
	int _parent;

	void setParent(int p) { _parent = p; }

public:
	bvh_node() {
		_child = 0;
		_parent = 0;
	}

	~bvh_node() {
		NULL;
	}

	void collide(bvh_node *other, vector<id_pair> &ret) {
		if (isLeaf() && other->isLeaf()) {
			ret.push_back(id_pair(this->triID(), other->triID()));
			return;
		}

		if (!_box.overlaps(other->box()))
			return;

		if (isLeaf()) {
			collide(other->left(), ret);
			collide(other->right(), ret);
		}
		else {
			left()->collide(other, ret);
			right()->collide(other, ret);
		}
	}

	void construct(unsigned int id);
	void construct(unsigned int *lst, unsigned int num);

	void visualize(int level);
	void refit();
	void resetParents(bvh_node *root);
	void rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret);
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, kmesh* km, REAL& ret);

	FORCEINLINE BOX &box() { return _box; }
	FORCEINLINE bvh_node *left() { return this - _child; }
	FORCEINLINE bvh_node *right() { return this - _child + 1; }
	FORCEINLINE int triID() { return _child; }
	FORCEINLINE int isLeaf() { return _child >= 0; }
	FORCEINLINE int parentID() { return _parent; }

	FORCEINLINE void getLevel(int current, int &max_level) {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current+1, max_level);
		right()->getLevel(current+1, max_level);
	}

	FORCEINLINE void getLevelIdx(int current, unsigned int *idx) {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current+1, idx);
		right()->getLevelIdx(current+1, idx);
	}

	friend class bvh;
};

class kmesh;

class bvh {
	int _num; // all face num
	bvh_node *_nodes;

	void construct(std::vector<kmesh*> &, REAL tol);
	void refit();
	void reorder(); // for breath-first refit
	void resetParents();

public:
	bvh(std::vector<kmesh*> &ms, REAL tol=0.0);

	~bvh() {
		if (_nodes)
			delete [] _nodes;
	}
	
	int num() { return _num; }
	bvh_node *root() { return _nodes; }

	void refit(std::vector<kmesh*> &ms, REAL tol=0.0);
	void collide(bvh* other, vector<id_pair>& ret)
	{
		root()->collide(other->root(), ret);
	}

	void rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret);
	//faster with fewer division
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dir2, kmesh* km, REAL& ret);
	//faster with no recursion
	void rayCasting3(const vec3f& pt, const vec3f& dir, const vec3f& dir2, kmesh* km, REAL& ret);

	void visualize(int);
};
