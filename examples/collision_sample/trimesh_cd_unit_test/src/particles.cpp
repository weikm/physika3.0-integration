#include <random>
#include <cmath>
#include <iostream>
#include "particles.h"

// Constructor definition
Particle::Particle()
{
    setRandomMovement();
    radius = 12.0;
}

float screenWidth = 300.f;
float screenHeight = 300.f;
float screenDepth = 300.f;

void Particle::update()
{

    if (std::round(x.x + v.x) >= screenWidth || std::round(x.x + v.x) <= 0)
        v.x = -v.x;
    if (std::round(x.y + v.y) >= screenHeight || std::round(x.y + v.y) <= 0)
        v.y = -v.y;
    if (std::round(x.z + v.z) >= screenDepth || std::round(x.z + v.z) <= 0)
        v.z = -v.z;

    // Add friction
    //v *= 0.998;
    x += v;
}

BOX Particle::bound()
{
    BOX bx;
    bx += x;
    bx.enlarge(radius);
    return bx;
}

inline float getRandT()
{
    return rand() / float(RAND_MAX);
}

inline float getRandNT()
{
    return (getRandT() - 1) * 2;
}

void Particle::setRandomMovement()
{
    // Set random movement between 1 and 10
    float randX = getRandT() * screenWidth;
    float randY = getRandT() * screenHeight;
    float randZ = getRandT() * screenDepth;

    x = vec3f(randX, randY, randZ);

    // If randX or randY < 10 then make negative
    v = vec3f(getRandNT() * 3, getRandNT() * 3, getRandNT() * 3);
}

void Particle::setColor(long c)
{
    color = vec3f(rand() / float(RAND_MAX), rand() / float(RAND_MAX), rand() / float(RAND_MAX));
}

ParticlePool::ParticlePool()
{
    std::srand(std::clock());

    for (int i = 0; i < this->numOfParticles; i++) {
        Particle p = Particle();
        p.setColor(0);
        particles.push_back(p);
    }

    flags.resize(numOfParticles);
}

void ParticlePool::update()
{
    for (auto& p : particles)
        p.update();
}

BOX ParticlePool::bound()
{
    BOX bx;
    for (auto p : particles) {
        bx += p.bound();
    }
    return bx;
}

void
ParticlePool::checkCollision(std::vector<id_pair>& pairs)
{
    int num = particles.size();
    for (int i = 0; i < num; i++)
        flags[i] = false;

    for (int i = 0; i < num; i++) {
        Particle& pa = particles[i];
        for (int j = i + 1; j < num; j++) {
            Particle& pb = particles[j];

            if ((pa.x - pb.x).length() < (pa.radius + pb.radius)) {
                flags[i] = flags[j] = true;
                pairs.push_back(id_pair(i, j, false));
            }
        }
    }
}
