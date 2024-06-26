namespace Physika {
class alignas(16) ManifoldPoint
{
};

#define MANIFOLD_CACHE_SIZE 4

class alignas(16) Manifold
{
    ManifoldPoint m_pointCache[MANIFOLD_CACHE_SIZE];

public:
    Manifold();

    int addManifoldPoint(const ManifoldPoint& newPoint, bool isPredictive = false);

    void removeContactPoint(int index);

    void replaceContactPoint(const ManifoldPoint& newPoint, int insertIndex);

    bool validContactDistance(const ManifoldPoint& pt) const;
};
}  // namespace Physika