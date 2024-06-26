#pragma once
#include "vec3f.h"
#include "box.h"
#include "pair.h"

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
    BOX bound();

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
    BOX bound();
    void checkCollision(std::vector<id_pair>& pairs);
};

