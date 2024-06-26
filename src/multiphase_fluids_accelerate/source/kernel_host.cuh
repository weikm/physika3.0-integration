#ifndef DEF_HOST_CUDA
#define DEF_HOST_CUDA

#include <vector_types.h>
#include <driver_types.h>
#include <chrono>


extern "C"
{
	void cudaInit(int argc, char **argv);
	void cudaExit(int argc, char **argv);

	void AllocateThreads(int numParticles, int blockSize, int& numBlocks, int& numThreads);

	void TransferDataToCUDA(float* position, float* velocity, float* acceleration, float* pressure, float* density, 
							int* type, int* explosion, int* lock, bool* active, bool* render, bool* rotate, float* center,
							float* smooth_radius, int* grid_ver_idx, float* concentration, bool* mix,float* eforce);
	void TransferDataFromCUDA(float* position, float* velocity, float* acceleration, float* pressure, float* density, int* type, bool* render, float* particle_radius, float* concentration,
							float* scalar_field, float* color, float* grid_scalar_field, float* grid_color_field);
	void TransferSumDataFromCUDA(int& number, float* result);
	void TransferGridDataFromCUDA(uint* grid_particle_cellindex, uint* num_particle_grid, uint* grid_particle_table, uint* next_particle_index);

	void TransferApplyDataToCUDA(bool* active, float* rest_mass, float* mix_rest_density, int current_num, int* lock);
	void TransferLockDataToCUDA(int* lock);
	void TransferVelocityDataToCUDA(float* mix_velocity, float* acceleration);

	void SetupParticlesCUDA(int numParticles, int3 grid_resolution, int grid_number, float kernel_self, int container_size, int3 mc_grid_resolution, int mc_grid_number, int max_num);
	void SetParametersCUDA(float3 boundary_min, float3 boundary_max, float3 grid_boundary_offset, float3 gravity, float mass, float rest_density,
						   float gas_constant, float viscosity, float damp, float simscale, float smooth_radius, float particle_radius,
						   float grid_radius, bool _explicit, int test_index, float3 center, float smoothing_factor, float k_n, float k_r, float k_s,
						   int neighbor_threshold, float surface_tension_factor, float mc_grid_radius);
	void SetupMarchingCubeCUDA(int3 mc_grid_resolution, int mc_grid_number, float mc_grid_radius);
	void SetGeneralParametersCUDA(float gas_constant);
	void SetMcParametersCUDA(float itp_radius, float anisotropic_radius, float k_n, float k_r, float k_s);
	void ClearParticlesCUDA();

	void AllocateGridCUDA(int numParticles);

	// bubbles
	void TransferBubbleDataFromCUDA(float* position, float* particle_radius, int* type);

	// peridynamics
	void ComputeTensorKCUDA(int numParticles);
	void ComputeTensorFCUDA(float time_step, int numParticles);

	// muti-phase
	void TransferMFDataToCUDA(float* mforce, float* restmass, float* mrest_density, float* mpressure, float* alpha_advanced, int* phase, float* mix_velocity, float* rest_mass_k);
	void TransferMFDataFromCUDA(float* alpha_advanced, float* restmass, float* _eff_V, float* _delta_mass, float* _delta_mass_k, float* _delta_alpha, float* _mix_density, float* _mix_velocity);
	void SetMFParametersCUDA(float* density, float* mass, float* visc, int phase_number, float tau, float sigma, bool miscible, float3* phase_color);
	void ComputeMFDensityCUDA(int numParticles);
	void ApplyAlphaCUDA(int numParticles);
	void ComputeDriftVelocityCUDA(int numParticles);
	void ComputeAlphaAdvanceCUDA(float time_step, int numParticles);
	void ComputeCorrectionCUDA(int numParticles);
	void ComputeCorrectionVolumeCUDA(float time_step, int numParticles);
	void AlphaBanlanceUsingMassFractionCUDA(float time_step, int numParticles);
	void ComputeTDMCUDA(int numParticles);
	void UpdateMFParticlesCUDA(float time_step, int numParticles);

	// muti-phase in peridynamics
	void ComputeDriftVelocityPeridynamicsCUDA(int numParticles, float bound_vel, float factor);
	void ComputeSdVPeridynamicsCUDA(int numParticles);
	void ComputeAlphaAdvancePeridynamicsCUDA(float time_step, int numParticles);
	void ComputeTDMPeridynamicsCUDA(float time_step, int numParticles, float surface_factor);
	void ComputeMFItemPeridynamicsCUDA(float time_step, int numParticles);
	void MFPressureSolvePeridynamicsCUDA(float time_step, int numParticles, int numFluids);

	// phase transport
	void ComputeAlphaTransportCUDA(float time_step, int numParticles, float factor);
	void ContributeAlphaCorrectionCUDA(int numParticles);

	// particle bound
	void ComputePhiParticleBoundCUDA(int numParticles);
	void ComputeMFDensityParticleBoundCUDA(int numParticles, bool transferAlpha);
	void ContributePressureCUDA(int numParticles);
	void ComputeBoundPressureCUDA(int numParticles);
	void AdvanceParticleBoundCUDA(float time_step, int numParticles);

	// generate mf particles
	void GenerateParticlesCUDA(float time_step, int begin, int N, int numParticles, int Generate_pos, float3 start_point, float GenFrameRate);
	void ChemicalReactionCUDA(int numParticles, float ReactionSpeed);
	void ConcentrationDecayCUDA(int numParticles);

	// rigid body
	void UpdateRigidBodyCUDA(float time_step, int numParticles, float omega);
	void UpdateRigidBodyDriftCUDA(float time_step, int numParticles);
	void RigidBodyTransitionCUDA(int numParticles);
	void UpdateUpBoundCUDA(float time_step, int numParticles);


	// density scalar field
	void GetAverageKernelPosCUDA(int numParticles);
	void ComputePosCovarianceCUDA(int numParticle);
	void ComputeShapeMatrixGCUDA(int numParticles);
	void CalDensityScalarFieldParticleCUDA(int numParticles);

	// Mcube grid structure
	void AllocateMcGridCUDA(int numParticles);
	void ParticleScalarvalueToGridCUDA(int numMcGridver);

	// Reset particles
	void ResetParticleAttribCUDA(int numParticles);

	// Secondary bubbles
	void ComputeBubbleAlphaCUDA(int numParticles);
	void InsertBubbleParticleCUDA(int numParticles);
	void UpdatePosBubbleCUDA(float time_step, int numParticles);
	void DeleteBubbleParticleCUDA(int numParticles);
	void ResortBubbleList(int* target, float3* target_pos, float3* position, int* idxToListIdx, int& num_bubble);

	// Air-liquid surface tension
	void ComputeSurfaceTensionCUDA(int numParticles);
	void ComputeVorticityCUDA(int numParticles);

	// change:
	void InsertParticlesCUDA(int numParticles);
	void PrefixSumCellsCUDA(int zero_offsets);
	void CountingSortFullCUDA(int numParticles);
	void ComputeShapeCUDA(float time_step, int numParticles);
	void ComputeAlphaCUDA(float time_step, int numParticles);
	void ComputeForceCUDA(float time_step, int numParticles);
	void AdvanceParticleCUDA(float time_step, int numParticles, int frame);
    void CopyToComponent(float*pos,float*color,int numParticles);
	void TestFunc(float time_step, int numParticles);
}

#endif // !DEF_HOST_CUDA
