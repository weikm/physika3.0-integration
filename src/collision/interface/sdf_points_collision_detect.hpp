#pragma once
#include "collision/interface/collidable_sdf.hpp"
#include "collision/internal/distance_field3d.hpp"
namespace Physika {

/**
 * SDFPointsCollisionDetect is a solver to detect the collision
 * between sdf and points 
 * for example a car's sdf and the sand particles
 */ 

class SDFPointsCollisionDetect
{
public:
    /**
     * @brief attach the sdf and points
     * @param[in] points    points to do collision detect
     * @param[in] points    sdf to do collision detect
     * @param[in] radius    collision radius
     * @param[in] num       the num of points
     */
    SDFPointsCollisionDetect(vec3f* points,DistanceField3D* sdf,float radius,int num);

    SDFPointsCollisionDetect();

    ~SDFPointsCollisionDetect();

    /**
     * @brief attach the points
     * @param[in] points    sdf to do collision detect
     */
    void setPoints(vec3f* points);

    /**
     * @brief set the radius of the point
     * @param[in] points    the radius of point
     */
    void setRadius(float radius);

    /**
     * @brief attach the sdf
     * @param[in] points    sdf to do collision detect
     */
    void setDistanceField(DistanceField3D* sdf);

    /**
     * @brief       get all collision result of sdf and ponit collision detect
     * @param[out]  collisionNum     the number of collide point to sdf 
     * @param[out]  collisionIds     the ids of collide point
     * @param[out]  collisionNormals the normals of collide point
     * @param[out]  collisionDistance the distance of collide point
     */
    void getCollisionResult(int& collisionNum, int*& collisionIds, vec3f*& collisionNormals, float*& collisionDistance) const;

    /**
     * @brief   get the collision num of sdf and ponit collision detect
     * @return  the number of collide points
     */
    int getCollisionNums() const;

    /**
     * @brief   get the collision ids of sdf and ponit collision detect
     * @return  the ids of all collide points
     */
    const int* getCollisionIds() const;

    /**
     * @brief   get the collision num of sdf and ponit collision detect
     * @return  the normals of all collide point
     */
    const vec3f* getCollisionNormals() const;

    /**
     * @brief   get the collision distance of sdf and ponit collision detect
     * @return  the distance of all collide point
     */
    const float* getCollisionDistance() const;

    /**
     * @brief get the collision result of points and sdf
     *
     * @return true if run successfully
     */
    bool execute();

private:
    bool m_is_init;
    int  max_collisionNum;  // the max num of collision
    // input
    DistanceField3D* m_distanceField;  //!< signed distance field
    vec3f* m_points;         //!< collision meshes
    float m_radius;         //!< collision radius   
    int m_num_points;      //!< the num of points

    // output
    int    m_num_collisions;      // the num of collisions
    int*   m_collision_id;        // the id of collide particle
    vec3f* m_collision_normal;    // the normal of collision
    float* m_collision_distance;  // the distance of penetration
};
}  // namespace Physika
