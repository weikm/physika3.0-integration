#pragma once
#include <vector>

#include "DataArray.cuh"
namespace Physika {
struct SPHParticles
{
	DataArray<float3> m_pos;
	DataArray<float3> m_vel;
    DataArray<float3> m_exforce;
	DataArray<float> m_mass;
	DataArray<float> m_density;
	DataArray<float> m_pressure;
	DataArray<int> m_valid; // indicate valid or not
	DataArray<uint32_t> m_p2cell; // Look-up key using space filling curves such as Morton(z-index), Hilbert or simple XYZ.
	
	size_t m_capacity; // max num of particles
	size_t m_particle_num; // num of live particles 

	size_t* m_particle_index; // A CUDA pointer, indicates current particle index.

	// initialize with invariant particles num 
    explicit SPHParticles(const std::vector<float3>& p)
        :
		m_pos(p.size()),
		m_vel(p.size()), 
		m_exforce(p.size()),
		m_mass(p.size()),
		m_density(p.size()),
		m_pressure(p.size()),
		m_valid(p.size()),
		m_p2cell(p.size()),
		m_capacity(p.size()),
		m_particle_num(p.size()) {
		checkCudaErrors(cudaMemcpy(m_pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice)); // copy positions
		checkCudaErrors(cudaMalloc((void**)&m_particle_index, sizeof(size_t)));
		checkCudaErrors(cudaMemcpy(m_particle_index, &m_particle_num, sizeof(size_t), cudaMemcpyHostToDevice));
	}
	// initialize with a capacity number
	explicit SPHParticles(const std::vector<float3>& p, const size_t _capacity) :
		m_pos(_capacity),
		m_vel(_capacity), 
		m_exforce(_capacity),
		m_mass(_capacity),
		m_density(_capacity),
		m_pressure(_capacity),
		m_valid(_capacity),
		m_p2cell(_capacity),
		m_capacity(_capacity),
		m_particle_num(p.size()){
		checkCudaErrors(cudaMemcpy(m_pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice)); // copy positions
		checkCudaErrors(cudaMalloc((void**)&m_particle_index, sizeof(size_t)));
		checkCudaErrors(cudaMemcpy(m_particle_index, &m_particle_num, sizeof(size_t), cudaMemcpyHostToDevice));
	}
    SPHParticles(const SPHParticles&) = delete;
    SPHParticles &operator=(const SPHParticles&) = delete;
	unsigned int getCapacity() const {
		return m_capacity;
	}
    unsigned int size() const{
		return m_particle_num;
    }
	
	size_t* getIndexPtr() const {
		return m_particle_index;
	}
    float3* getPosPtr() const {
		return m_pos.addr();
	}
	float3* getVelPtr() const {
		return m_vel.addr();
	}
	float3* getExternalForcePtr() const {
        return m_exforce.addr();
	}
	const DataArray<float3>& getPos() const {
		return m_pos;
	}
    float* getPressurePtr() const {
		return m_pressure.addr();
	}
	const DataArray<float>& getPressure() const {
		return m_pressure;
	}
	int* getValidPtr() const {
		return m_valid.addr();
	}
	const DataArray<int>& getValid()const {
		return m_valid;
	}
	float* getDensityPtr() const {
		return m_density.addr();
	}
	const DataArray<float>& getDensity() const {
		return m_density;
	}
	uint32_t* getP2Cell() const {
		return m_p2cell.addr();
	}
	float* getMassPtr() const {
		return m_mass.addr();
	}
	void advect(float dt);
	float3* getMaxVelocity();
	float3* getMinVelocity();

	int check_size();
	int add_size(size_t num);
	int reduce_size(size_t num);
	virtual ~SPHParticles() noexcept { 
		cudaFree(m_particle_index); 
	}
};
struct ThermoParticles: public SPHParticles // Thermo-Particles with temperature & different rest_density
{
	DataArray<float> m_temp; // Store temperature, for water: 0~100 degrees centigrade. Over heat should be transfered to latent heat. 
	DataArray<float> m_latent; // Store latent heat for phase-change, upper bound is 2260kJ for 1kg water. 
	DataArray<float> m_humidity; // Store humidity for gas condensation. particles or thermal boundary particles 
	DataArray<int>	 m_type; // '1': liquidity / boundary, '0' gaseous / nucleation point (when phase-change judge)
	//DataArray<float> m_humidity; // Relative humidity
	explicit ThermoParticles(const std::vector<float3>& p);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float3>& v, const std::vector<float>& t);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<int>& n);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const std::vector<int>& n);

	explicit ThermoParticles(const std::vector<float3>& p, const size_t _capacity);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const size_t _capacity);
    explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float3>& v, const std::vector<float>& t, const size_t _capacity);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<int>& n, const size_t _capacity);
	explicit ThermoParticles(const std::vector<float3>& p, const std::vector<float>& t, const std::vector<int>& n, const size_t _capacity);
	float* getTempPtr() const {
		return m_temp.addr();
	}
	float* getLatentPtr() const {
		return m_latent.addr();
	}
	float* getHumidityPtr() const {
		return m_humidity.addr();
	}
	int* getTypePtr() const {
		return m_type.addr();
	}
};
}