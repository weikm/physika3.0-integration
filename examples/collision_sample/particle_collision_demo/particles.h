#pragma once

#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/collision_pair.hpp"
#include "collision/internal/collision_aabb.hpp"

using Physika::vec3f;
using Physika::aabb;
using Physika::id_pair;

class Particle
{
public:
    vec3f x, v;
    REAL radius;

    // Constructor declaration
    Particle();

    void update();
    void draw(int);
    void setRandomMovement();
    void setColor(long c);
    aabb<REAL> bound();

private:
    vec3f color;
};

class ParticlePool
{
public:
    int numOfParticles = 300;
    std::vector<Particle> particles;
    std::vector<int> flags;

    // Contructor declaration
    ParticlePool();

    void update();
    void draw();
    aabb<REAL> bound();
    void checkCollision(std::vector<id_pair>& pairs);
};

