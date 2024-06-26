#pragma once

#include "quantization.h"


class qbvh;
class bvh_node;
class bvh;

//qbvh_node is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
class alignas(16) qbvh_node {
	//12 bytes
	unsigned short int	bxMin[3];
	unsigned short int	bxMax[3];
	//4 bytes
	int _child; // >=0 leaf with tri_id, <0 left & right

public:
	FORCEINLINE qbvh_node() {}
	FORCEINLINE void set(const qbvh* tree, const bvh_node* ot);

	FORCEINLINE const qbvh_node* left() const { return this - _child; }
	FORCEINLINE const qbvh_node* right() const { return this - _child + 1; }
	FORCEINLINE qbvh_node* left() { return this - _child; }
	FORCEINLINE qbvh_node* right() { return this - _child + 1; }

	FORCEINLINE int triID() const { return _child; }
	FORCEINLINE int isLeaf() const { return _child >= 0; }

	FORCEINLINE bool testQuantizedBoxOverlapp(
		const unsigned short* quantizedMin, const unsigned short* quantizedMax) const
	{
		if (bxMin[0] > quantizedMax[0] ||
			bxMax[0] < quantizedMin[0] ||
			bxMin[1] > quantizedMax[1] ||
			bxMax[1] < quantizedMin[1] ||
			bxMin[2] > quantizedMax[2] ||
			bxMax[2] < quantizedMin[2])
		{
			return false;
		}
		return true;
	}

	void query(const unsigned short* minPt, const unsigned short* maxPt, std::vector<int>& rets) const
	{
		if (!testQuantizedBoxOverlapp(minPt, maxPt))
			return;

		if (isLeaf())
			rets.push_back(this->triID());
		else {
			left()->query(minPt, maxPt, rets);
			right()->query(minPt, maxPt, rets);
		}
	}

};

//Quantized BVH
class qbvh {
	int _num; //all face num
	qbvh_node* _nodes;
	BOX _global_bound;
	vec3f _quantization;

protected:
	FORCEINLINE void calc_quantization(BOX globalBound, REAL boundMargin = REAL(1.0))
	{
		calc_quantization_parameters(_global_bound, _quantization, globalBound, boundMargin);
	}

public:
	qbvh(const bvh* ot);

	FORCEINLINE void quantizePoint(
		unsigned short* quantizedpoint, const vec3f& point) const
	{
		quantize_clamp(quantizedpoint, point, _global_bound, _quantization);
	}

	~qbvh() {
		if (_nodes)
			delete[] _nodes;
	}

	FORCEINLINE void getInfo(vec3f& bMin, vec3f& bMax, vec3f& quan)
	{
		bMin = _global_bound.getMin();
		bMax = _global_bound.getMax();
		quan = _quantization;
	}

	FORCEINLINE int num() const { return _num; }
	FORCEINLINE qbvh_node* root() const { return _nodes; }

	void query(const unsigned short*minPt, const unsigned short *maxPt, std::vector<int>& rets) const
	{
		root()->query(minPt, maxPt, rets);
	}

};


class sbvh;

//qbvh_node is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
class alignas(16) sbvh_node {
	//12 bytes
	unsigned short int	bxMin[3];
	unsigned short int	bxMax[3];
	//4 bytes
	int _child; // >=0 leaf with tri_id, <0 left & right

public:
	FORCEINLINE sbvh_node() {}
	FORCEINLINE void set(const sbvh* tree, const bvh_node* ot);

	const sbvh_node* left(sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum) const; // { return this - _child; }
	const sbvh_node* right(sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum) const; // { return this - _child + 1; }
//	FORCEINLINE sbvh_node* left() { return this - _child; }
//	FORCEINLINE sbvh_node* right() { return this - _child + 1; }

	FORCEINLINE int triID() const { return _child; }
	FORCEINLINE int isLeaf() const { return _child >= 0; }

	FORCEINLINE bool testQuantizedBoxOverlapp(
		const unsigned short* quantizedMin, const unsigned short* quantizedMax) const
	{
		if (bxMin[0] > quantizedMax[0] ||
			bxMax[0] < quantizedMin[0] ||
			bxMin[1] > quantizedMax[1] ||
			bxMax[1] < quantizedMin[1] ||
			bxMin[2] > quantizedMax[2] ||
			bxMax[2] < quantizedMin[2])
		{
			return false;
		}
		return true;
	}

	void query(const unsigned short* minPt, const unsigned short* maxPt, sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum, std::vector<int>& rets) const
	{
		if (!testQuantizedBoxOverlapp(minPt, maxPt))
			return;

		if (isLeaf())
			rets.push_back(this->triID());
		else {
			left(upperPtr, lowerPtr, upperNum)->query(minPt, maxPt, upperPtr, lowerPtr, upperNum, rets);
			right(upperPtr, lowerPtr, upperNum)->query(minPt, maxPt, upperPtr, lowerPtr, upperNum, rets);
		}
	}

};


//Segmented Quantized BVH
class sbvh {
	int _num; //all face num

	int _upperNum, _lowerNum;
	sbvh_node* _upperNodes;
	sbvh_node* _lowerNodes;

	BOX _global_bound;
	vec3f _quantization;

protected:
	FORCEINLINE void calc_quantization(BOX globalBound, REAL boundMargin = REAL(1.0))
	{
		calc_quantization_parameters(_global_bound, _quantization, globalBound, boundMargin);
	}

public:
	sbvh(const bvh* ot, int upperNum);

	FORCEINLINE void quantizePoint(
		unsigned short* quantizedpoint, const vec3f& point) const
	{
		quantize_clamp(quantizedpoint, point, _global_bound, _quantization);
	}

	~sbvh() {
		if (_upperNodes)
			delete[] _upperNodes;
		if (_lowerNodes)
			delete[] _lowerNodes;
	}

	FORCEINLINE void getInfo(vec3f& bMin, vec3f& bMax, vec3f& quan)
	{
		bMin = _global_bound.getMin();
		bMax = _global_bound.getMax();
		quan = _quantization;
	}

	FORCEINLINE int num() const { return _num; }
	FORCEINLINE sbvh_node* root() const { return _upperNodes; }
	FORCEINLINE sbvh_node* upperNodes() const { return _upperNodes; }
	FORCEINLINE sbvh_node* lowerNodes() const { return _lowerNodes; }
	FORCEINLINE int upperNum() const { return _upperNum; }
	FORCEINLINE int lowerNum() const { return _lowerNum; }

	void query(const unsigned short* minPt, const unsigned short* maxPt, std::vector<int>& rets) const
	{
		root()->query(minPt, maxPt, _upperNodes, _lowerNodes, _upperNum, rets);
	}

};