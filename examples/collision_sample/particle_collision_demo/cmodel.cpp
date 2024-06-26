//#define B150 1

using namespace std;
#include <omp.h>
#include "crigid.h"

#include "particles.h"
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "collision/internal/collision_transf.hpp"
#include "collision/interface/collidable_points.hpp"
#include "collision/interface/particles_collision_solver.hpp"

auto             particles_scene  = Physika::World::instance().createScene();
Physika::Object* particles = Physika::World::instance().createObject();
auto             particles_solver = Physika::World::instance().createSolver<Physika::ParticlesCollisionSolver>();

class myTimer
{
	double t0;
	char msg[512];
public:
	myTimer(const char* msgIn) {
		t0 = omp_get_wtime();
		strcpy(msg, msgIn);
	}
	~myTimer() {
		double tdelta = omp_get_wtime() - t0;
		printf("%s: %2.5f s\n", msg, tdelta);
	}
};

class myTimer2
{
	double dt;
	char msg[512];
public:
	myTimer2(const char* msgIn) {
		dt = 0;
		strcpy(msg, msgIn);
	}

	void print() {
		printf("%s: %2.5f s\n", msg, dt);
	}

	void inc(double delta) {
		dt += delta;
	}
};

#define BUNNY_SCALE 1.f

#pragma warning(disable: 4996)

extern void drawSdfPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void drawMinPair(crigid* r0, crigid* r1, std::vector<vec3f>&pairs);
extern void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs);
extern void drawRigid(crigid*, bool cyl, int level, vec3f &);

#define MAX_TRI_CLIPPING 16

class cscene
{
public:
    ParticlePool _particles;

public:

    void draw(int level, bool showCD, bool showBody, bool showOnly)
    {
        _particles.draw();
    }

    // for collision detection
    std::vector<id_pair> cdPairs;
} g_scene;

void initModel(const char* c1file, const char* c2file)
{
	// init data for physika solver
    particles_scene->addObject(particles);
	particles->addComponent<Physika::CollidablePointsComponent>();
    particles_scene->addSolver(particles_solver);
	particles_solver->attachObject(particles);

	// init particles data
	particles->getComponent<Physika::CollidablePointsComponent>()->m_num = g_scene._particles.numOfParticles;
    particles->getComponent<Physika::CollidablePointsComponent>()->m_radius = g_scene._particles.particles[0].radius;
    particles->getComponent<Physika::CollidablePointsComponent>()->m_pos    = new vec3f[g_scene._particles.numOfParticles];

	// init solver
    particles_solver->initialize();
	return;
}


bool exportModel(const char* cfile)
{
    return true;
}

bool importModel(const char* cfile)
{
    return true;
}

void quitModel()
{

}

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	g_scene.draw(level, tri, true, edge);
}

extern double totalQuery;

bool dynamicModel(char*, bool, bool)
{
    g_scene._particles.update();
    for (int i = 0 ; i < g_scene._particles.numOfParticles; i++)
	{
        particles->getComponent<Physika::CollidablePointsComponent>()->m_pos[i] = g_scene._particles.particles[i].x;
	}
    
	return true;
}

void checkCollision()
{
    for (int i = 0 ;i < g_scene._particles.numOfParticles; i++)
	{
		g_scene._particles.flags[i] = 0;
	}

	std::vector<Physika::id_pair> res;
    particles_solver->run();
    particles_solver->getResult(res);
    g_scene.cdPairs.clear();
	g_scene.cdPairs = res;
    for (int i = 0; i < g_scene.cdPairs.size(); i++)
	{
        g_scene._particles.flags[g_scene.cdPairs[i].id0()] = 1;
        g_scene._particles.flags[g_scene.cdPairs[i].id1()] = 1;
	}
	return;
}