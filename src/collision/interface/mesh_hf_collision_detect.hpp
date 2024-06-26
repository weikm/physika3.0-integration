#include "collision/internal/trimesh.hpp"
#include "collision/internal/height_field.hpp"
namespace Physika {
// * MeshHfCollisionDetect detect the collision between heightfield and trimesh
// * it returns the array of collision point id, collision point normal. and the
// * number of collisions
class MeshHfCollisionDetect
{
public:
    MeshHfCollisionDetect(triMesh& mesh, heightField1d& heightField);
    ~MeshHfCollisionDetect() {}

    void execute();

    void setHeightField(heightField1d& heightField);

    void setMesh(triMesh& trimesh);

    void getResult(int* collisionID, vec3f* collisionNormal, int& count) const;

    
    int getMaxCollision()
    {
        return m_maxCollision;
    }

    void setMaxCollision(int maxCollision)
    {
        this->m_maxCollision = maxCollision;
    }

private:
    void executeInternal();

    int m_maxCollision;  // the max num of collisions
    // input
    triMesh*       m_trimesh;
    heightField1d* m_heightField;

    // output
    int*   m_collision_id;      // the id of collision point
    vec3f* m_collision_normal;  // the normal of collision point
    int    m_counter;           // the num of collisions
};
}  // namespace Physika