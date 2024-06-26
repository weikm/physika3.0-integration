#include <iostream>

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "framework/solver.hpp"

#include "collision/interface/collidable_ray.hpp"
#include "collision/interface/collidable_sdf.hpp"
#include "collision/interface/collidable_points.hpp"
#include "collision/interface/mesh_ray_collision_solver.hpp"
#include "collision/interface/mesh_hf_collision_solver.hpp"
#include "collision/interface/sdf_points_collision_solver.hpp"
#include "collision/internal/height_field.hpp"
#include "collision/interface/mesh_mesh_collision_solver.hpp"
#include "collision/interface/mesh_mesh_collision_detect.hpp"
#include "collision/interface/mesh_mesh_discrete_collision_detect.hpp"
#include "collision/interface/mesh_ray_collision_detect.hpp"

using namespace Physika;

 //mesh-mesh collision solver for car and car

//int main() 
//{
//    auto scene_car_car = World::instance().createScene();
//    auto obj_car1   = World::instance().createObject();
//    auto obj_car2   = World::instance().createObject();
//    auto car_car_collision_solver = World::instance().createSolver<MeshMeshCollisionSolver>();
//
//    obj_car1->addComponent<CollidableTriangleMeshComponent>();
//    obj_car2->addComponent<CollidableTriangleMeshComponent>();
//
//    scene_car_car->addObject(obj_car1);
//    scene_car_car->addObject(obj_car2);
//    scene_car_car->addSolver(car_car_collision_solver);
//
//    car_car_collision_solver->attachObject(obj_car1);
//    car_car_collision_solver->attachObject(obj_car2);
//
//    std::cout << "Scene A: id(" << scene_car_car->id() << ").\n";
//    std::cout << "    Object A: id(" << obj_car1->id() << "), with CollidableMeshComponent.\n";
//    std::cout << "    Object B: id(" << obj_car2->id() << "), with CollidableMeshComponent.\n";
//    std::cout << "    MeshMeshCollisionSolver: id(" << car_car_collision_solver->id() << ").\n";
//    std::cout << "End\n";
//
//    int numVtxa = 4, numVtxb = 3;
//    int numTria = 2, numTrib = 1;
//    // mesh data for obj_a and obj_b
//    vec3f* vtxa  = new vec3f[numVtxa];
//    vec3f* ovtxa = new vec3f[numVtxa];
//    vtxa[0]      = vec3f(0.0f, 0.0f, 0.0f);
//    vtxa[1]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtxa[2]      = vec3f(1.0f, 0.0f, 0.0f);
//    vtxa[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    ovtxa[0]     = vec3f(0.0f, 0.0f, 2.0f);
//    ovtxa[1]     = vec3f(0.0f, 1.0f, 2.0f);
//    ovtxa[2]     = vec3f(1.0f, 0.0f, 2.0f);
//    ovtxa[3]     = vec3f(-1.0f, 0.0f, 2.0f);
//    vec3f* vtxb  = new vec3f[numVtxb];
//    vec3f* ovtxb = new vec3f[numVtxb];
//    vtxb[0]      = vec3f(0.0f, -1.0f, 0.0f);
//    vtxb[1]      = vec3f(0.0f, 0.0f, 1.0f);
//    vtxb[2]      = vec3f(0.0f, 1.0f, 0.0f);
//    ovtxb[0]     = vec3f(2.0f, -1.0f, 0.0f);
//    ovtxb[1]     = vec3f(2.0f, 0.0f, 1.0f);
//    ovtxb[2]     = vec3f(2.0f, 1.0f, 0.0f);
//    tri3f* tria = new tri3f[numTria];
//    tria[0].set(0, 1, 2);
//    tria[1].set(0, 1, 3);
//    tri3f* trib = new tri3f[numTrib];
//    trib[0].set(0, 1, 2);
//    bool selfCollisionOption = false;
//    triMesh mesha                                                              = triMesh(numVtxa, numTria, tria, vtxa, false);
//    triMesh meshb															  = triMesh(numVtxb, numTrib, trib, vtxb, false);
//    obj_car1->getComponent<CollidableTriangleMeshComponent>()->m_mesh    = &mesha;
//    obj_car1->getComponent<CollidableTriangleMeshComponent>()->m_mesh->_ovtxs   = ovtxa;
//
//
//    obj_car2->getComponent<CollidableTriangleMeshComponent>()->m_mesh           = &meshb;
//    obj_car2->getComponent<CollidableTriangleMeshComponent>()->m_mesh->_ovtxs   = ovtxb;
//    
//    car_car_collision_solver->doCollision();
//}

 //mesh-ray collision solver for car-bullet scene
//using namespace Physika;
//int main()
//{
//    auto scene_car_bullet    = World::instance().createScene();
//    auto obj_car    = World::instance().createObject();
//    auto obj_bullet = World::instance().createObject();
//
//    obj_car->addComponent<CollidableTriangleMeshComponent>();
//    obj_bullet->addComponent<CollidableRayComponent>();
//
//    // init for obj car
//    int numVtx = 4;
//    int numTri = 2;
//    // mesh data for obj_a and obj_b
//    vec3f* vtx  = new vec3f[numVtx];
//    vec3f* ovtx = new vec3f[numVtx];
//    Physika::tri3f* tri  = new Physika::tri3f[numTri];
//    vtx[0]      = vec3f(0.0f, 0.0f, 0.0f);
//    vtx[1]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtx[2]      = vec3f(1.0f, 0.0f, 0.0f);
//    vtx[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    tri[0].set(0, 1, 2);
//    tri[1].set(0, 1, 3);
//    
//    triMesh mesh(numVtx, numTri, tri, vtx, false);
//    obj_car->getComponent<CollidableTriangleMeshComponent>()->m_mesh = &mesh;
//
//    Ray ray(vec3f(0, 0, 1), vec3f(0, 1, -1));
//
//    obj_bullet->getComponent<CollidableRayComponent>()->m_ray = &ray;
//
//    scene_car_bullet->addObject(obj_car);
//    scene_car_bullet->addObject(obj_bullet);
//
//    auto car_bullet_collision_solver = World::instance().createSolver<MeshRayCollisionSolver>();
//    car_bullet_collision_solver->setLCS(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1), vec3f(1,1,1), true);
//    car_bullet_collision_solver->attachObject(obj_car);
//    car_bullet_collision_solver->attachObject(obj_bullet);
//    scene_car_bullet->addSolver(car_bullet_collision_solver);
//
//    car_bullet_collision_solver->initialize();
//
//    car_bullet_collision_solver->run();
//    vec3f intersectPoint, rayDir;
//    car_bullet_collision_solver->getResult(intersectPoint, rayDir);
//    std::cout << intersectPoint.x << " " << intersectPoint.y << " " << intersectPoint.z << std::endl;
//}

 //sdf-points collision solver for car-sand scene
//int main()
//{
//    auto scene_car_sand = World::instance().createScene();
//    auto obj_car          = World::instance().createObject();
//    auto obj_sand       = World::instance().createObject();
//    auto car_sand_collision_solver = World::instance().createSolver<SDFPointsCollisionSolver>();
//
//    //obj_car->addComponent<TriangleMeshComponent>();
//    obj_car->addComponent<CollidableTriangleMeshComponent>();
//    obj_car->addComponent<CollidableSDFComponent>();
//
//    obj_sand->addComponent<CollidablePointsComponent>();
//
//    scene_car_sand->addObject(obj_car);
//    scene_car_sand->addObject(obj_sand);
//    scene_car_sand->addSolver(car_sand_collision_solver);
//
//    car_sand_collision_solver->attachObject(obj_car);
//    car_sand_collision_solver->attachObject(obj_sand);
//
//    car_sand_collision_solver->doCollision();
//}

//int main()
//{
//    // init for obj car
//    int numVtx = 4;
//    int numTri = 2;
//    // mesh data for obj_a and obj_b
//    vec3f* vtx  = new vec3f[numVtx];
//    vec3f* ovtx = new vec3f[numVtx];
//    tri3f* tri  = new tri3f[numTri];
//    vtx[0]      = vec3f(0.0f, 0.0f, 0.0f);
//    vtx[1]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtx[2]      = vec3f(1.0f, 0.0f, 0.0f);
//    vtx[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    tri[0].set(0, 1, 2);
//    tri[1].set(0, 1, 3);
//
//    TriangleMeshComponent*  mesh = new TriangleMeshComponent;
//    CollidableRayComponent* ray  = new CollidableRayComponent;
//    mesh->setTris(tri, numTri);
//    mesh->setVtxs(vtx, numVtx);
//
//    ray->m_dir    = vec3f(0, 1, -1);
//    ray->m_origin = vec3f(0, 0, 1);
//
//    auto car_bullet_collision_solver = MeshRayCollisionSolver(vec3f(1, 0, 0), vec3f(0, 1, 0), vec3f(0, 0, 1), vec3f(1, 1, 1), mesh, ray);
//    car_bullet_collision_solver.run();
//    vec3f intersectPoint, rayDir;
//    car_bullet_collision_solver.getResult(intersectPoint, rayDir);
//    std::cout << intersectPoint.x << " " << intersectPoint.y << " " << intersectPoint.z << std::endl;
//}

//int main()
//{
//    auto scene_car_sand            = World::instance().createScene();
//    auto obj_car                   = World::instance().createObject();
//    auto obj_sand                  = World::instance().createObject();
//    auto car_sand_collision_solver = World::instance().createSolver<MeshHeightFieldCollisionSolver>();
//
//    obj_car->addComponent<TriangleMeshComponent>();
//    obj_car->addComponent<CollidableTriangleMeshComponent>();
//    obj_car->addComponent<CollidableHeightFieldComponent>();
//
//    obj_sand->addComponent<CollidableHeightFieldComponent>();
//
//    int numVtx = 4;
//    int numTri = 2;
//    // mesh data for obj_a and obj_b
//    vec3f* vtx  = new vec3f[numVtx];
//    vec3f* ovtx = new vec3f[numVtx];
//    tri3f* tri  = new tri3f[numTri];
//    vtx[0]      = vec3f(0.0f, 0.0f, 0.0f);
//    vtx[1]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtx[2]      = vec3f(1.0f, 0.0f, 0.0f);
//    vtx[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    tri[0].set(0, 1, 2);
//    tri[1].set(0, 1, 3);
//    obj_car->getComponent<TriangleMeshComponent>()->setVtxs(vtx, numVtx);
//    obj_car->getComponent<TriangleMeshComponent>()->setTris(tri, numTri);
//    car_sand_collision_solver->attachObject(obj_car);
//    car_sand_collision_solver->attachObject(obj_sand);
//    car_sand_collision_solver->isInitialized();
//
//    car_sand_collision_solver->doCollision();
//}

//mesh -mesh collide detection
int main() {
    
    int numVtxa = 4, numVtxb = 3;
    int numTria = 2, numTrib = 1;
    // mesh data for obj_a and obj_b
    vec3f* vtxa  = new vec3f[numVtxa];
    vec3f* ovtxa = new vec3f[numVtxa];
    vtxa[0]      = vec3f(0.0f, -2.0f, 1.0f);
    vtxa[1]      = vec3f(1.0f, -2.0f, 0.0f);
    vtxa[2]      = vec3f(2.0f, -2.0f, 1.0f);
    vtxa[3]      = vec3f(-1.0f, 0.0f, 1.0f);
    ovtxa[0]     = vec3f(0.0f, 2.0f, 1.0f);
    ovtxa[1]     = vec3f(1.0f, 2.0f, 0.0f);
    ovtxa[2]     = vec3f(2.0f, 2.0f, 1.0f);
    ovtxa[3]     = vec3f(-1.0f, 0.0f, 1.0f);
    vec3f* vtxb  = new vec3f[numVtxb];
    vec3f* ovtxb = new vec3f[numVtxb];
    vtxb[0]      = vec3f(0.0f, -1.0f, 0.0f);
    vtxb[1]      = vec3f(0.0f, 0.0f, 1.0f);
    vtxb[2]      = vec3f(0.0f, 1.0f, 0.0f);
    ovtxb[0]     = vec3f(2.0f, -1.0f, 0.0f);
    ovtxb[1]     = vec3f(2.0f, 0.0f, 1.0f);
    ovtxb[2]     = vec3f(2.0f, 1.0f, 0.0f);
    tri3f* tria = new tri3f[numTria];
    tria[0].set(0, 1, 2);
    tria[1].set(0, 1, 3);
    tri3f* trib = new tri3f[numTrib];
    trib[0].set(0, 1, 2);
    bool selfCollisionOption = false;
    triMesh mesha                                                              = triMesh(numVtxa, numTria, tria, vtxa, false);
    triMesh meshb															  = triMesh(numVtxb, numTrib, trib, vtxb, false);
    mesha._ovtxs = ovtxa;
    meshb._ovtxs = ovtxb;
    std::vector<triMesh*> m_meshes;
    m_meshes.push_back(&mesha);
    m_meshes.push_back(&meshb);
    auto    mesh_mesh_collision_detector                                       = MeshMeshCollisionDetect(m_meshes);
    
    mesh_mesh_collision_detector.execute();
    std::vector<ImpactInfo> contact_info;
    mesh_mesh_collision_detector.getImpactInfo(contact_info);
    for (int i = 0; i < contact_info.size(); i++)
    {
        std::cout << "dist: " << contact_info[i].m_dist << "   "
                  << "face id: " << contact_info[i].m_faceId[0] << " " << contact_info[i].m_faceId[1] << "\n";
	};
}

//// mesh -mesh dcd collide detection
//int main()
//{
//    int numVtxa = 4, numVtxb = 4;
//    int numTria = 2, numTrib = 2;
//    // mesh data for obj_a and obj_b
//    vec3f* vtxa  = new vec3f[numVtxa];
//    vtxa[0]      = vec3f(0.0f, 0.0f, 0.0f);
//    vtxa[1]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtxa[2]      = vec3f(1.0f, 0.0f, 0.0f);
//    vtxa[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    vec3f* vtxb  = new vec3f[numVtxb];
//    vtxb[0]      = vec3f(0.0f, -1.0f, 0.0f);
//    vtxb[1]      = vec3f(0.0f, 0.0f, 1.0f);
//    vtxb[2]      = vec3f(0.0f, 1.0f, 0.0f);
//    vtxb[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//    tri3f* tria  = new tri3f[numTria];
//    tria[0].set(0, 1, 2);
//    tria[1].set(0, 1, 3);
//    tri3f* trib = new tri3f[numTrib];
//    trib[0].set(0, 1, 3);
//    trib[1].set(0, 1, 2);
//    bool    selfCollisionOption = false;
//    triMesh mesha               = triMesh(numVtxa, numTria, tria, vtxa, false);
//    triMesh meshb               = triMesh(numVtxb, numTrib, trib, vtxb, false);
//    std::vector<triMesh*> m_meshes;
//    m_meshes.push_back(&mesha);
//    m_meshes.push_back(&meshb);
//    auto mesh_mesh_dcd_collision_detector = MeshMeshDiscreteCollisionDetect(m_meshes);
//
//    mesh_mesh_dcd_collision_detector.execute();
//
//    std::vector<std::vector<id_pair>> pairs;
//    mesh_mesh_dcd_collision_detector.getContactPairs(pairs);
//    for (int i = 0; i < pairs.size(); i++)
//    {
//        std::cout << "Pair " << i << ":\n";
//        std::cout << "    mesh id: " << pairs[i][0].id0()
//                  << " face id : " << pairs[i][0].id1() << "\n ";
//        std::cout << "    mesh id: " << pairs[i][1].id0()
//                  << " face id : " << pairs[i][1].id1() << "\n ";
//    };
//}

//mesh-ray collision detection
// int main()
//{
//     // init for obj car
//     int numVtx = 4;
//     int numTri = 2;
//     // mesh data for obj_a and obj_b
//     vec3f* vtx  = new vec3f[numVtx];
//     vec3f* ovtx = new vec3f[numVtx];
//     tri3f* tri  = new tri3f[numTri];
//     vtx[0]      = vec3f(0.0f, 0.0f, 0.0f);
//     vtx[1]      = vec3f(0.0f, 1.0f, 0.0f);
//     vtx[2]      = vec3f(1.0f, 0.0f, 0.0f);
//     vtx[3]      = vec3f(-1.0f, 0.0f, 0.0f);
//     tri[0].set(0, 1, 2);
//     tri[1].set(0, 1, 3);
//
//     triMesh*                mesh = new triMesh(numVtx,numTri,tri,vtx,false);
//     Ray*     ray  = new Ray(vec3f(0, 0, 1), vec3f(0, 1, -1));
//
//
//     auto car_bullet_collision_detect = MeshRayCollisionDetect(*ray, *mesh);
//     car_bullet_collision_detect.execute();
//     vec3f intersectPoint, rayDir;
//     car_bullet_collision_detect.getIntersectPoint(intersectPoint);
//     std::cout << intersectPoint.x << " " << intersectPoint.y << " " << intersectPoint.z << std::endl;
// }