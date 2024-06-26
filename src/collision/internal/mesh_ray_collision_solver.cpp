#include "framework/object.hpp"
#include <queue>
#include "collision/internal/collision_mat3f.hpp"
#include "collision/interface/mesh_ray_collision_solver.hpp"
#include "collision/interface/collidable_ray.hpp"
#include "collision/interface/collidable_trianglemesh.hpp"
#include "collision/interface/mesh_ray_collision_detect.hpp"
 namespace Physika {
 MeshRayCollisionSolver::MeshRayCollisionSolver()
     : Solver(), m_is_init(false), m_mesh_ray_object(nullptr)
 {
 }

 MeshRayCollisionSolver::MeshRayCollisionSolver(const vec3f lcsXDir, const vec3f lcsYDir, const vec3f lcsZDir, const vec3f lcsOrigin, const bool transformToLCS)
     : Solver(), m_is_init(false), m_mesh_ray_object(nullptr)
 {
     setLCS(lcsXDir, lcsYDir, lcsZDir, lcsOrigin, transformToLCS);
 }

 MeshRayCollisionSolver::~MeshRayCollisionSolver()
 {
     this->reset();
 }

 bool MeshRayCollisionSolver::initialize()
 {
     std::cout << "MeshRayCollisionSolver::initialize() initializes the solver.\n";
     m_is_init = true;
     return true;
 }

 bool MeshRayCollisionSolver::isInitialized() const
 {
     std::cout << "MeshRayCollisionSolverr::isInitialized() gets the initialization status of the  solver.\n";
     return m_is_init;
 }

 bool MeshRayCollisionSolver::reset()
 {
     std::cout << "MeshRayCollisionSolver::reset() sets the solver to newly constructed state.\n";
     m_is_init = false;
     m_mesh_ray_object->reset();

     return true;
 }

 bool MeshRayCollisionSolver::run()
 {
     return step();
 }

 bool MeshRayCollisionSolver::step()
 {
     std::cout << "MeshRayCollisionSolver::run() updates the solver till termination criteria are met.\n";
     if (!m_is_init)
     {
         std::cout << "error: solver not initialized.\n";
         return false;
     }
     if (!m_mesh_ray_object)
     {
         std::cout << "error: mesh ray object not attached to the solver.\n";
         return false;
     }
     std::cout << "MeshRayCollisionSolver::run() applied to mesh ray object " << m_mesh_ray_object->id() << ".\n";
     doCollision();
     return true;
 }

 bool MeshRayCollisionSolver::isApplicable(const Object* object) const
 {
     std::cout << "MeshRayCollisionSolver::isApplicable() checks if object has CollidableMeshComponent.\n";
     if (!object)
         return false;

     return object->hasComponent<MeshRayCollisionComponent>();
 }

 bool MeshRayCollisionSolver::attachObject(Object* object)
 {
     std::cout << "MeshRayCollisionSolver::attachObject() set the target of the solver.\n";
     if (!object)
         return false;

     if (!this->isApplicable(object))
     {
         std::cout << "    error: object is not applicable.\n";
         return false;
     }

     if (object->hasComponent<MeshRayCollisionComponent>())
     {
         std::cout << "    object attached .\n";
         m_mesh_ray_object = object;
     }

     return true;
 }

 bool MeshRayCollisionSolver::detachObject(Object* object)
 {
     std::cout << "MeshRayCollisionSolver::detachObject() remove the object from target list of the solver.\n";
     if (!object)
         return false;

     if (m_mesh_ray_object == object)
     {
         m_mesh_ray_object = nullptr;
         std::cout << "    mesh ray object detached.\n";
         return true;
     }
     std::cout << "    error: object is not attached to the solver.\n";
     return false;
 }

 void MeshRayCollisionSolver::clearAttachment()
 {
     std::cout << "MeshRayCollisionSolver::clearAttachment() clears the target list of the solver.\n";
     m_mesh_ray_object = nullptr;
 }

 bool MeshRayCollisionSolver::doCollision()
 {
     auto                   ray  = m_mesh_ray_object->getComponent<MeshRayCollisionComponent>()->m_ray;
     auto                   mesh = m_mesh_ray_object->getComponent<MeshRayCollisionComponent>()->m_mesh;
     MeshRayCollisionDetect solver(*ray, *mesh);
     solver.execute();
     solver.getIntersectPoint(m_mesh_ray_object->getComponent<MeshRayCollisionComponent>()->m_intersectPoint);

     bool intersectState = false;
     solver.getIntersectState(intersectState);
     if (m_mesh_ray_object->getComponent<MeshRayCollisionComponent>()->m_transformToLCS)
     {
         transform();
     }
     return intersectState;
 }

 bool MeshRayCollisionSolver::transform()
 {
     auto component = m_mesh_ray_object->getComponent<MeshRayCollisionComponent>();
     matrix3f localTransform(
         component->m_lcsXDir.x, component->m_lcsYDir.x, component->m_lcsZDir.x, component->m_lcsXDir.y, component->m_lcsYDir.y, component->m_lcsZDir.y, component->m_lcsXDir.z, component->m_lcsYDir.z, component->m_lcsZDir.z);
     matrix3f inverseLocalTransform = localTransform.getInverse();
     component->m_lcsIntersectPoint  = inverseLocalTransform * (component->m_intersectPoint - component->m_lcsOrigin);
     component->m_lcsIntersectNormal = inverseLocalTransform * component->m_intersectNormal;
     component->m_lcsRayDir          = inverseLocalTransform * component->m_ray->m_dir;
     std::cout << "MeshRayCollisionSolver::transform() transforms the position of ray mesh intersection point and normal and the ray direction from world space to local space.\n";
     return true;
 }


 void MeshRayCollisionSolver::setLCS(const vec3f lcsXDir, const vec3f lcsYDir, const vec3f lcsZDir, const vec3f lcsOrigin, const bool transformToLCS)
 {
     auto component   = m_mesh_ray_object->getComponent<MeshRayCollisionComponent>();
     component->m_lcsOrigin = lcsOrigin;
     component->m_lcsXDir   = lcsXDir;
     component->m_lcsYDir   = lcsYDir;
     component->m_lcsZDir   = lcsZDir;
     component->m_transformToLCS = transformToLCS;
 }

 void MeshRayCollisionSolver::getResult(vec3f& intersectPoint, vec3f& rayDir) const
 {
     auto component = m_mesh_ray_object->getComponent<MeshRayCollisionComponent>();
     if (component->m_transformToLCS)
     {
         intersectPoint = component->m_lcsIntersectPoint;
         rayDir         = component->m_lcsRayDir;
     }
     else
     {
         intersectPoint = component->m_intersectPoint;
         rayDir         = component->m_ray->m_dir;
     }
 }
 }  // namespace Physika