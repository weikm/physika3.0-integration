#pragma once

#include <float.h>
#include <stdlib.h>
#include "real.h"
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

public:
	bvh_node() {
		_child = 0;
	}

	~bvh_node() {
		NULL;
	}

	void query(const BOX& bx, std::vector<int>& rets) const
	{
		if (!_box.overlaps(bx))
			return;

		if (isLeaf())
			rets.push_back(this->triID());
		else {
			left()->query(bx, rets);
			right()->query(bx, rets);
		}
	}

	void distance(const bvh_node* other, const cbox2boxTrfCache& trfBx, const transf& trfA2B,
		kmesh* mA, kmesh* mB, REAL& minDist, std::vector<vec3f>& ret);

	void collide(const bvh_node* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& ret) const
	{
		if (!overlapping_trans_cache(_box, other->_box, trf, true))
			return;

		if (isLeaf() && other->isLeaf()) {
			ret.push_back(id_pair(this->triID(), other->triID(), false));
			return;
		}

		if (isLeaf()) {
			collide(other->left(), trf, ret);
			collide(other->right(), trf, ret);
		}
		else {
#if 0
			left()->collide(other, trf, ret);
			right()->collide(other, trf, ret);
#else
			if (other->isLeaf()) {
				left()->collide(other, trf, ret);
				right()->collide(other, trf, ret);
			}
			else {
				left()->collide(other->left(), trf, ret);
				right()->collide(other->left(), trf, ret);
				left()->collide(other->right(), trf, ret);
				right()->collide(other->right(), trf, ret);
			}
#endif
		}

	}

	void collide(const bvh_node *other, std::vector<id_pair> &ret) const {
		if (isLeaf() && other->isLeaf()) {
			ret.push_back(id_pair(this->triID(), other->triID(), false));
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
	void getChilds(int level, std::vector<bvh_node*> &rets);
	void getChilds(int level, bvh_node *root, std::vector<int>& rets);

	void refit();
	void resetParents(bvh_node *root);
	void rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret);
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, kmesh* km, REAL& ret);

	FORCEINLINE const BOX &box() const { return _box; }
	FORCEINLINE const bvh_node *left() const { return this - _child; }
	FORCEINLINE const bvh_node *right() const { return this - _child + 1; }
	FORCEINLINE bvh_node* left() { return this - _child; }
	FORCEINLINE bvh_node* right() { return this - _child + 1; }

	FORCEINLINE int triID() const { return _child; }
	FORCEINLINE int isLeaf() const { return _child >= 0; }

	FORCEINLINE void getLevel(int current, int &max_level) const {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current+1, max_level);
		right()->getLevel(current+1, max_level);
	}

	FORCEINLINE void getLevelIdx(int current, unsigned int *idx) const {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current+1, idx);
		right()->getLevelIdx(current+1, idx);
	}

	friend class bvh;
	friend class qbvh_node;
	friend class sbvh_node;
};

class kmesh;

class bvh {
	int _num; // all face num
	bvh_node *_nodes;

	void construct(std::vector<kmesh*> &, REAL tol);
	void refit();
	void reorder(); // for breath-first refit

public:
	bvh(std::vector<kmesh*> &ms, REAL tol=0.0);

	~bvh() {
		if (_nodes)
			delete [] _nodes;
	}
	
	int num() const { return _num; }
	bvh_node *root() const { return _nodes; }

	void refit(std::vector<kmesh*> &ms, REAL tol=0.0);

	void distance(const bvh* other, const cbox2boxTrfCache& trfBx, const transf& trfA2B,
		kmesh* mA, kmesh* mB, REAL& minDist, std::vector<vec3f>& ret, bool parallel);

	void collide(const bvh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& ret)
	{
		root()->collide(other->root(), trf, ret);
	}

	void collideWithStack(const bvh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& ret);

	void collide(bvh* other, std::vector<id_pair>& ret)
	{
		root()->collide(other->root(), ret);
	}

	void query(const BOX& bx, std::vector<int>& rets)
	{
		root()->query(bx, rets);
	}

	void rayCasting(const vec3f& pt, const vec3f& dir, kmesh *km, REAL& ret);
	//faster with fewer division
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dir2, kmesh* km, REAL& ret);
	//faster with no recursion
	void rayCasting3(const vec3f& pt, const vec3f& dir, const vec3f& dir2, kmesh* km, REAL& ret);

	void visualize(int);

	friend class qbvh;
	friend class sbvh;
};


