#include "collision/internal/collision_tri3f.hpp"

namespace Physika
{
tri3f::tri3f()
{
    ids[0] = ids[1] = ids[2] = -1;
}

tri3f::tri3f(unsigned int id0, unsigned int id1, unsigned int id2)
{
    set(id0, id1, id2);
}

void tri3f::set(unsigned int id0, unsigned int id1, unsigned int id2)
{
    ids[0] = id0;
    ids[1] = id1;
    ids[2] = id2;
}

// i can only be 0, 1, 2
unsigned int tri3f::id(int i) const
{
    return ids[i];
}
unsigned int tri3f::id0() const
{
    return ids[0];
}
unsigned int tri3f::id1() const
{
    return ids[1];
}
unsigned int tri3f::id2() const
{
    return ids[2];
}
}