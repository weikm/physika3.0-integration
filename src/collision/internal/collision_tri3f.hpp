#pragma once
#include <type_traits>
#include <algorithm>
/**
 * a datastructure to store a triangle face by its three vertex indices, the id can only range from 0 to 2
 */

namespace Physika {
// triangle face
struct tri3f
{

    tri3f();

    tri3f(unsigned int id0, unsigned int id1, unsigned int id2);

    void set(unsigned int id0, unsigned int id1, unsigned int id2);

    // i can only be 0, 1, 2
    unsigned int id(int i) const;
    unsigned int id0() const;
    unsigned int id1() const;
    unsigned int id2() const;
    unsigned int ids[3];
    void reverse()
    {
        std::swap(ids[0], ids[2]);
    }
};

class stri3f
{
public:
    unsigned int _ids[3];
    unsigned int _sids[3];

    stri3f()
    {
        set(-1, -1, -1);
    }

    stri3f(unsigned int id0, unsigned int id1, unsigned int id2)
    {
        set(id0, id1, id2);
    }

    void set(unsigned int id0, unsigned int id1, unsigned int id2)
    {
        _sids[0] = _ids[0] = id0;
        _sids[1] = _ids[1] = id1;
        _sids[2] = _ids[2] = id2;

        std::sort(std::begin(_sids), std::end(_sids));
    }

    tri3f getTri() const
    {
        return tri3f(_ids[0], _ids[1], _ids[2]);
    }

    unsigned int id(int i)
    {
        return _ids[i];
    }
     unsigned int id0()
    {
        return _ids[0];
    }
     unsigned int id1()
    {
        return _ids[1];
    }
     unsigned int id2()
    {
        return _ids[2];
    }
     void reverse()
    {
        std::swap(_ids[0], _ids[2]);
    }

    bool operator<(const stri3f& other) const
    {
        if (_sids[0] == other._sids[0])
        {
            if (_sids[1] == other._sids[1])
                return _sids[2] < other._sids[2];
            else
                return _sids[1] < other._sids[1];
        }
        else
            return _sids[0] < other._sids[0];
    }

    bool operator==(const stri3f& other) const
    {
        return (_sids[0] == other._sids[0] && _sids[1] == other._sids[1] && _sids[2] == other._sids[2]);
    }
};
}  // namespace Physika