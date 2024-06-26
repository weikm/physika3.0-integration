
#include "tmbvh.h"
#include "qbvh.h"
#include <climits>
#include <utility>


void 
qbvh_node::set(const qbvh* tree, const bvh_node* ot)
{
	_child = ot->_child;

	const BOX& bx = ot->_box;
	tree->quantizePoint(bxMin, bx.getMin());
	tree->quantizePoint(bxMax, bx.getMax());
}

qbvh::qbvh(const bvh* ot)
{
	_num = ot->num();
	BOX bx = ot->root()->box();
	calc_quantization(bx);

	int nodeNum = _num * 2 - 1;
	_nodes = new qbvh_node[nodeNum];
	for (int i = 0; i < nodeNum; i++)
		_nodes[i].set(this, ot->_nodes+i);
}


void
sbvh_node::set(const sbvh* tree, const bvh_node* ot)
{
	_child = ot->_child;

	const BOX& bx = ot->_box;
	tree->quantizePoint(bxMin, bx.getMin());
	tree->quantizePoint(bxMax, bx.getMax());
}

const sbvh_node* sbvh_node::left(sbvh_node *upperPtr, sbvh_node *lowerPtr, int upperNum) const
{
	if ((this - upperPtr) < upperNum) {
		const sbvh_node *r = this - _child;
		if ((r - upperPtr) < upperNum)
			return r;
		else
			return lowerPtr + (r - upperPtr) - upperNum;
	} else
		return this - _child;
}

const sbvh_node* sbvh_node::right(sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum) const
{
	if ((this - upperPtr) < upperNum) {
		const sbvh_node* r = this-_child+1;
		if ((r - upperPtr) < upperNum)
			return r;
		else
			return lowerPtr + (r - upperPtr) - upperNum;
	}
	else
		return this-_child+1;
}

sbvh::sbvh(const bvh* ot, int upperNum)
{
	_num = ot->num();
	BOX bx = ot->root()->box();
	calc_quantization(bx);

	int nodeNum = _num * 2 - 1;

	if (upperNum > nodeNum) {
		_upperNum = nodeNum;
		_lowerNum = 0;
	}
	else {
		_upperNum = upperNum;
		_lowerNum = nodeNum - upperNum;
	}

	_upperNodes = new sbvh_node[_upperNum];
	for (int i = 0; i < _upperNum; i++)
		_upperNodes[i].set(this, ot->_nodes + i);

	if (_lowerNum) {
		_lowerNodes = new sbvh_node[_lowerNum];
		for (int i = 0; i < _lowerNum; i++)
			_lowerNodes[i].set(this, ot->_nodes + i+_upperNum);
	}
	else
		_lowerNodes = nullptr;
}
