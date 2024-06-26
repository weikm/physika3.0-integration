#include "collision/interface/particles_collision_detect.hpp"

namespace Physika {
ParticlesCollisonDetect::ParticlesCollisonDetect() {}

ParticlesCollisonDetect::~ParticlesCollisonDetect() {}

ParticlesCollisonDetect::ParticlesCollisonDetect(std::vector<vec3f> points, int num, float radius) {
	m_points = points;
    m_num_points = num;
	m_radius = radius;
}

void ParticlesCollisonDetect::setPoints(std::vector<vec3f> points)
{
		m_points = points;
}

void ParticlesCollisonDetect::setRadius(float radius)
{
	m_radius = radius;
}

void ParticlesCollisonDetect::getCollisionPairs(std::vector<id_pair>& pairs) const
{
	pairs = m_pairs;
}

bool ParticlesCollisonDetect::execute()
{
    for (int i = 0; i < m_points.size(); i++)
    {
        vec3f p = m_points[i];
		for (int j = i + 1; j < m_points.size(); j++)
		{
			vec3f q = m_points[j];
			if ((p-q).length() < m_radius)
			{
                m_pairs.push_back(id_pair(i, j,false));
			}
		}

    }
    return true;
}
}