#include "collision/internal/collision_qbvh.hpp"

namespace Physika {
qbvh_node::qbvh_node() 
{
}

void qbvh_node::set(const qbvh* tree, const bvh_node* ot)
{
    _child = ot->_child;

    const BOX<REAL>& bx = ot->_box;
    tree->quantizePoint(bxMin, bx.getMin());
    tree->quantizePoint(bxMax, bx.getMax());
}

const qbvh_node* qbvh_node::left() const
{
    return this - _child;
}
const qbvh_node* qbvh_node::right() const
{
    return this - _child + 1;
}
qbvh_node* qbvh_node::left()
{
    return this - _child;
}
qbvh_node* qbvh_node::right()
{
    return this - _child + 1;
}

int qbvh_node::triID() const
{
    return _child;
}
int qbvh_node::isLeaf() const
{
    return _child >= 0;
}

bool qbvh_node::testQuantizedBoxOverlapp(
    const unsigned short* quantizedMin,
    const unsigned short* quantizedMax) const
{
    if (bxMin[0] > quantizedMax[0] || bxMax[0] < quantizedMin[0] || bxMin[1] > quantizedMax[1] || bxMax[1] < quantizedMin[1] || bxMin[2] > quantizedMax[2] || bxMax[2] < quantizedMin[2])
    {
        return false;
    }
    return true;
}

void qbvh_node::query(const unsigned short* minPt, const unsigned short* maxPt, std::vector<int>& rets) const
{
    if (!testQuantizedBoxOverlapp(minPt, maxPt))
        return;

    if (isLeaf())
        rets.push_back(this->triID());
    else
    {
        left()->query(minPt, maxPt, rets);
        right()->query(minPt, maxPt, rets);
    }
}

qbvh::qbvh(const bvh* ot)
{
    _num   = ot->num();

    BOX<REAL> bx = ot->root()->box();
    calc_quantization(bx);

    int nodeNum = _num * 2 - 1;
    _nodes      = new qbvh_node[nodeNum];
    for (int i = 0; i < nodeNum; i++)
        _nodes[i].set(this, ot->_nodes + i);
}

void sbvh_node::set(const sbvh* tree, const bvh_node* ot)
{
    _child = ot->_child;

    const BOX<REAL>& bx = ot->_box;
    tree->quantizePoint(bxMin, bx.getMin());
    tree->quantizePoint(bxMax, bx.getMax());
}

const sbvh_node* sbvh_node::left(sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum) const
{
    if ((this - upperPtr) < upperNum)
    {
        const sbvh_node* r = this - _child;
        if ((r - upperPtr) < upperNum)
            return r;
        else
            return lowerPtr + (r - upperPtr) - upperNum;
    }
    else
        return this - _child;
}

const sbvh_node* sbvh_node::right(sbvh_node* upperPtr, sbvh_node* lowerPtr, int upperNum) const
{
    if ((this - upperPtr) < upperNum)
    {
        const sbvh_node* r = this - _child + 1;
        if ((r - upperPtr) < upperNum)
            return r;
        else
            return lowerPtr + (r - upperPtr) - upperNum;
    }
    else
        return this - _child + 1;
}

sbvh::sbvh(const bvh* ot, int upperNum)
{
    _num   = ot->num();
    BOX<REAL> bx = ot->root()->box();
    calc_quantization(bx);

    int nodeNum = _num * 2 - 1;

    if (upperNum > nodeNum)
    {
        _upperNum = nodeNum;
        _lowerNum = 0;
    }
    else
    {
        _upperNum = upperNum;
        _lowerNum = nodeNum - upperNum;
    }

    _upperNodes = new sbvh_node[_upperNum];
    for (int i = 0; i < _upperNum; i++)
        _upperNodes[i].set(this, ot->_nodes + i);

    if (_lowerNum)
    {
        _lowerNodes = new sbvh_node[_lowerNum];
        for (int i = 0; i < _lowerNum; i++)
            _lowerNodes[i].set(this, ot->_nodes + i + _upperNum);
    }
    else
        _lowerNodes = nullptr;
}
};  // namespace Physika