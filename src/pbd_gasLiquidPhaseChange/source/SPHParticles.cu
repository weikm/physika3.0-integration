#include "SPHParticles.hpp"

#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <memory>

#include "Util/helper_math.h"
namespace Physika {
void SPHParticles::advect(float dt)
{	
	thrust::transform(thrust::device,
		m_pos.addr(), m_pos.addr() + size(),
		m_vel.addr(),
		m_pos.addr(),
		[dt]__host__ __device__(const float3& lhs, const float3& rhs) { return lhs + dt*rhs; }	
	);
}

float3* SPHParticles::getMaxVelocity() {
	return thrust::max_element(m_vel.addr(), m_vel.addr() + size(),
		[]__host__ __device__(const float3 & lhs, const float3 & rhs) { return length(lhs) < length(rhs); });
}

float3* SPHParticles::getMinVelocity() {
	return thrust::min_element(m_vel.addr(), m_vel.addr() + size(),
		[]__host__ __device__(const float3 & lhs, const float3 & rhs) { return length(lhs) < length(rhs); });
}

int SPHParticles::check_size() {
	size_t particle_index = 0;
	size_t* particle_index_ptr = &particle_index;
	checkCudaErrors(cudaMemcpy(particle_index_ptr, this->m_particle_index, sizeof(size_t), cudaMemcpyDeviceToHost));
	if (particle_index <= m_capacity && particle_index > 0) {
		this->m_particle_num = particle_index;
		return 1;
	}
	return 0;
}

int SPHParticles::reduce_size(size_t num) {
	if (num < 1) return 0;
	if (m_particle_num < num)
		return -1;
	m_particle_num -= num;
	size_t particle_index = m_particle_num;
	size_t* particle_index_ptr = &particle_index;
	checkCudaErrors(cudaMemcpy(this->m_particle_index, particle_index_ptr, sizeof(size_t), cudaMemcpyHostToDevice));
	return 0;
}

int SPHParticles::add_size(size_t num) {
	if (num + m_particle_num > m_capacity)
		return -1;
	m_particle_num += num;
	checkCudaErrors(cudaMemcpy(this->m_particle_index, &m_particle_num, sizeof(size_t), cudaMemcpyHostToDevice));
	return 0;
}

ThermoParticles::ThermoParticles(const std::vector<float3>& p) :SPHParticles(p),
m_temp(p.size()), m_latent(p.size()), m_type(p.size()), m_humidity(p.size())
{
	// init temperature: 20 degrees centigrade
	thrust::fill(thrust::device, m_temp.addr(), m_temp.addr() + size(), 20.0f);
	// init fluid type: all liquid
	thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
}

ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t) :SPHParticles(p),
m_temp(p.size()), m_latent(p.size()), m_type(p.size()), m_humidity(p.size())
{
	// copy temperatures host to device
	checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice)); 
	// init type: all liquid
	thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float3>& v, const std::vector<float>& t)
    : SPHParticles(p), m_temp(p.size()), m_latent(p.size()), m_type(p.size()), m_humidity(p.size())
{
	// copy temperatures host to device
	checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice)); 
	// init type: all liquid
	thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
    checkCudaErrors(cudaMemcpy(m_vel.addr(), &v[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));  // copy vel
}

ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<int>& n) :SPHParticles(p),
m_temp(p.size()), m_latent(p.size()), m_type(p.size()), m_humidity(p.size())
{
	// init type
	checkCudaErrors(cudaMemcpy(m_type.addr(), &n[0], sizeof(int) * n.size(), cudaMemcpyHostToDevice)); 
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const std::vector<int>& n) :
SPHParticles(p),
m_temp(p.size()), m_latent(p.size()), m_type(p.size()), m_humidity(p.size())
{
	// copy temperatures host to device
	checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice));
	// init type
	checkCudaErrors(cudaMemcpy(m_type.addr(), &n[0], sizeof(int) * n.size(), cudaMemcpyHostToDevice));
}

// initialize with a capacity
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const size_t _capacity):
	SPHParticles(p, _capacity),
	m_temp(_capacity), m_latent(_capacity), m_type(_capacity), m_humidity(_capacity)
{
	// init temperature: 20 degrees centigrade
	thrust::fill(thrust::device, m_temp.addr(), m_temp.addr() + size(), 20.0f);
	// init fluid type: all liquid
	thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const size_t _capacity):
	SPHParticles(p, _capacity),
	m_temp(_capacity), m_latent(_capacity), m_type(_capacity), m_humidity(_capacity) 
{
	// copy temperatures host to device
	checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice));
	// init type: all liquid
	thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float3>& v, const std::vector<float>& t, const size_t _capacity)
    : SPHParticles(p, _capacity), m_temp(_capacity), m_latent(_capacity), m_type(_capacity), m_humidity(_capacity)
{
    // copy temperatures host to device
    checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice));
    // init type: all liquid
    thrust::fill(thrust::device, m_type.addr(), m_type.addr() + size(), 1);
    checkCudaErrors(cudaMemcpy(m_vel.addr(), &v[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));  // copy vel
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<int>& n, const size_t _capacity) :
	SPHParticles(p, _capacity),
	m_temp(_capacity), m_latent(_capacity), m_type(_capacity), m_humidity(_capacity) 
{
	// init type
	checkCudaErrors(cudaMemcpy(m_type.addr(), &n[0], sizeof(int) * n.size(), cudaMemcpyHostToDevice));
}
ThermoParticles::ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const std::vector<int>& n, const size_t _capacity) :
	SPHParticles(p, _capacity),
	m_temp(_capacity), m_latent(_capacity), m_type(_capacity), m_humidity(_capacity) 
{
	// copy temperatures host to device
	checkCudaErrors(cudaMemcpy(m_temp.addr(), &t[0], sizeof(float) * t.size(), cudaMemcpyHostToDevice));
	// init type
	checkCudaErrors(cudaMemcpy(m_type.addr(), &n[0], sizeof(int) * n.size(), cudaMemcpyHostToDevice));
}

}