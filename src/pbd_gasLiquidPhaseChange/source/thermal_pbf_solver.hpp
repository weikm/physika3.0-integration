#pragma once
#include "SPHParticles.hpp"
namespace Physika {
class ThermalPBFSolver {

public:
	void normalStep(std::shared_ptr<ThermoParticles>& fluids,
					const std::shared_ptr<ThermoParticles>& boundaries,
					const std::shared_ptr<ThermoParticles>& heaten_boundaries,
					const DataArray<uint32_t>& cellStartFluid,
					const DataArray<uint32_t>& cellStartBoundary,
					const DataArray<uint32_t>& cellStartHeaten,
					float3 spaceSize, int3 cellSize, float cellLength, float smoothRadius,
					float dt, float rho0, float rhoB, float visc, float3 G,
					float surfaceTensionIntensity, float airPressure);
	void thermalStep(std::shared_ptr<ThermoParticles>& fluids, 
					const std::shared_ptr<ThermoParticles>& boundaries, 
					const std::shared_ptr<ThermoParticles>& heaten_boundaries,
					const DataArray<uint32_t>& cellStartFluid, 
					const DataArray<uint32_t>& cellStartBoundary, 
					const DataArray<uint32_t>& cellStartHeaten, 
					float3 spaceSize, int3 cellSize, float cellLength, float smoothRadius, 
					float dt, float rho0, float rhoB, float visc, float3 G,
					float surfaceTensionIntensity, float airPressure);

	explicit ThermalPBFSolver(int num,
		int defaultMaxIter = 10, 
		int scene_type = 1,
		float defaultXSPH_c = 0.05f,
		float defaultRelaxation = 0.75f,
		float defaultConductivityFluid = 0.620f,
		float defaultConductivityBoundary = 1.5f)
        :
		m_maxIter(defaultMaxIter),
		m_xSPH_c(defaultXSPH_c),
		m_relaxation(defaultRelaxation),
		m_conductivityFluid(defaultConductivityFluid),
		m_conductivityBoundary(defaultConductivityBoundary),
		bufferInt(num),
		bufferInt2(num),
		fluidPosLast(num), bufferFloat3(num), 
		surfaceBufferFloat3(num),
		bufferFloat(num),
		outColorGradient(num),
		scene_type(scene_type) {}

	explicit ThermalPBFSolver(const std::shared_ptr<ThermoParticles>& particles,
		int defaultMaxIter = 10,
		int scene_type = 1,
		float defaultXSPH_c = 0.1f,
		float defaultRelaxation = 1.0f,
		float defaultConductivityFluid = 0.620f,
		float defaultConductivityBoundary = 1.5f)
        :
		m_maxIter(defaultMaxIter),
		m_xSPH_c(defaultXSPH_c),
		m_relaxation(defaultRelaxation),
		m_conductivityFluid(defaultConductivityFluid),
		m_conductivityBoundary(defaultConductivityBoundary),
		bufferInt(particles->getCapacity()),
		bufferInt2(particles->getCapacity()),
		fluidPosLast(particles->getCapacity()),
		bufferFloat3(particles->getCapacity()), 
		surfaceBufferFloat3(particles->getCapacity()),
		bufferFloat(particles->getCapacity()),
		outColorGradient(particles->getCapacity()),
		scene_type(scene_type)
	{
		initializePosLast(particles->getPos());
	}

	virtual ~ThermalPBFSolver() {	}
	void initializePosLast(const DataArray<float3>& posFluid) {
		checkCudaErrors(cudaMemcpy(fluidPosLast.addr(), posFluid.addr(), sizeof(float3) * fluidPosLast.length(), cudaMemcpyDeviceToDevice));
		posLastInitialized = true;
	}
	float* getColorGradient() {
		return outColorGradient.addr();
	}
	// heat transfer
	void heatTransfer(float dt, std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rhoBoundary,
		int3 cellSize, float cellLength, float smoothRadius);
	// mass transfer: fluid particles -> nucleation sites
	void massTransferVaporization(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
		float rho0, float trans_heat, int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius);
	void massTransferCondensation(float dt, std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
		float rho0, float trans_heat, int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius);

protected:
	void bubbleExternalForce(float dt, std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
		float rho0, float3 G, int3 cellSize, float3 spaceSize, float cellLength, float smoothRadius);

	void convectionForce(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries, float dt, float3 G,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, int3 cellSize, float cellLength, float smoothRadius);

	void predict(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize);
    void force(std::shared_ptr<SPHParticles>& fluids, float dt, float3 G);
    void advect(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize);
	// overwrite and hide the project function in BasicSPHSolver
	virtual int project(std::shared_ptr<ThermoParticles>& fluids, const std::shared_ptr<ThermoParticles>& boundaries,
		const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary,
		float rho0, int3 cellSize, float3 spaceSize, float cellLength,
		float radius, int maxIter);

	// overwrite and hide the diffuse function in BasicSPHSolver, apply XSPH viscosity
	virtual void diffuse(std::shared_ptr<ThermoParticles>& fluids, const DataArray<uint32_t>& cellStartFluid,
		int3 cellSize, float cellLength, float rho0,
		float radius, float visc);

	void applySurfaceEffects(std::shared_ptr<SPHParticles>& fluids, const DataArray<float3>& colorGrad, const DataArray<uint32_t>& cellStartFluid, float rho0, int3 cellSize, float cellLength, float radius, float dt, float surfaceTensionIntensity, float airPressure);
    void surfaceDetection(DataArray<float3>& colorGrad, const std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries, const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rho0, float rhoB, int3 cellSize, float cellLength, float radius);
    void handleSurface(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries, const DataArray<uint32_t>& cellStartFluid, const DataArray<uint32_t>& cellStartBoundary, float rho0, float rhoB, int3 cellSize, float cellLength, float radius, float dt, float surfaceTensionIntensity, float airPressure);


private:
	bool posLastInitialized = false;
	int scene_type; // 1: normal. 2: convection. 3: condensation.
	const int m_maxIter;
	const float m_xSPH_c;
	const float m_relaxation;
	const float m_conductivityFluid;
	const float m_conductivityBoundary;
	const float m_rest_volume = 76.596750762082e-6f;
	const float m_latent_heat = 2260.0f;
	DataArray<int> bufferInt;
	DataArray<int> bufferInt2;
	DataArray<float3> fluidPosLast;
    DataArray<float3> bufferFloat3;
    DataArray<float3> surfaceBufferFloat3;
	DataArray<float> bufferFloat;
	DataArray<float> outColorGradient;
	void updateNeighborhood(const std::shared_ptr<SPHParticles>& particles);
};
}