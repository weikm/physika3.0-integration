#ifndef DEF_KERN_CUDA
#define DEF_KERN_CUDA
#include <cstdio>
#include <cmath>
#include "common_defs.h"

typedef unsigned int		uint;
typedef unsigned short int	ushort;

struct tensor
{
	float tensor[9];
};

struct neighbor_value_container
{
	int index;
	float value[MAX_PHASE_NUMBER];
};

struct bufList {
	float3*		_position;
    float3*     _external_force;
    float3*     _render_pos;
	float3*		_velocity;
	float3*		_acceleration;
	float*		_density;
	float*		_pressure;
	int*		_type;
	int*		_phase;
	int*		_explosion;
	int*		_lock;
	bool*		_active;
	bool*		_render;
	bool*		_rotate;
	bool*		_noted;
	bool*		_test;
	bool*		_mix;
	// change:
	float*		_inv_dens;

	// temp container
	neighbor_value_container* _container;

	// peridynamics
	tensor*		_tensor_K;
	tensor*		_tensor_F;
	float*		_SdV;

	// muti-phase
	float3*		_mix_velocity;				// u_m						num_particle * 1
	float3*		_phase_velocity;			// u_k						num_particle * num_phase
	float3*		_drift_velocity;			// u_mk						num_particle * num_phase
	float3*		_force;						// f						num_particle * 1
	float3*		_pressure_force;			// pressure part of f		num_particle * 1
	float3*		_visc_force;				// visc part of f			num_particle * 1
	float3*		_phase_force;				// phase part of f			num_particle * 1
	float3*		_color;						// color of a particle		num_particle * 1
	float*		_mix_density;				// ~rho_m (interpolation)	num_particle * 1
	float*		_mix_pressure;				// p_m						num_particle * 1
	float*		_alpha;						// alpha_k					num_particle * num_phase
	float*		_delta_alpha;				// d_alpha_k				num_particle * num_phase
	float*		_alpha_advanced;			// alpha_k + d_alpha_k		num_particle * num_phase
	float*		_alpha_sum;					// sum of alpha_k			num_particle * 1
	float*		_viscosity;					// mu						num_particle * 1
	float*		_rest_mass;					// m						num_particle * 1
	float*		_mix_rest_density;			// rho_m (rest)				num_particle * 1
	float*		_lambda;					// lambda					num_particle * 1
	float*		_dV;						// dV / V					num_particle * 1
	float*		_eff_V;						// V_eff					num_particle * 1
	float*		_smooth_radius;				// h						num_particle * 1
	float*		_particle_radius;			// r						num_particle * 1
	float*		_delta_mass;				// d_m						num_particle * 1
	float*		_delta_mass_k;				// d_m_k					num_particle * num_phase
	float*		_rest_mass_k;				// m_k						num_particle * num_phase
	float*		_concentration;				// c						num_particle * 1
	float3*		_sf_k;						// sf_k						num_particle * num_phase        // for air and fluid only
	float3*		_surface_tension;			// SF						num_particle * 1
	float*		_mass_fraction;				// c_k						num_particle * num_phase
	float3*		_nabla_c;					// nabla_c_k				num_particle * num_phase

	// surface tracking
	float3*		_ave_position;				// pos_bar					num_particle * 1
	float3*		_weighted_position;			// pos_weighted				num_particle * 1
	float*		_surface_scalar_field;		// sf						num_particle * 1
	tensor*		_C;							// covariance matrix		num_particle * 1
	tensor*		_G;							// shape matrix				num_particle * 1

	// rigid
	float3*		_center;

	// pressure-bound
	float3*		_pre_velocity;
	float3*		_acceleration_p;
	float*		_rest_volume;
	float*		_volume;
	float*		_source_term;
	float*		_diag_term;
	float*		_residual;

	float*		_bound_phi;					// effective mass of the bound particle

	// data structure
	uint*		neighbortable;
	float*		neighbordist;
	uint*		neighbor_search_index;
	int*		particle_neighbor_number;

	uint*		grid_particle_table;
	uint*		next_particle_index;
	uint*		num_particle_grid;
	uint*		particle_grid_cellindex;
	int*		grid_search_offset;

	// mcube data structure
	uint*		mc_grid_particle_table;
	uint*		mc_next_particle_index;
	uint*		num_particle_mc_grid;
	uint*		particle_mc_grid_cellindex;
	int*		mc_grid_search_offset;

	float*		scalar_field_value_grid;
	float3*		color_field_grid;
	int*		grid_ver_idx;

	// sum function
	float*		sum_result;
	float*		min_data;

	// bubbles
	float*		inter_alpha;				// interpolate volume fraction
	float*		bubble_volume;				
	bool*		bubble;						// whether the particle is a bubble
	int*		bubble_type;				// 0 means small bubbles, and 1 means middle bubbles
	int*		attached_id;				// index of attached particle
	int*		idxToListIdx;				// map the idx of particle and idx of it in bubbleList
	int*		bubbleList;
	float3*		bubblePosList;				// bubblist that store bubble pos
	int*		bubbleNum;

	// vorticity
	float3*		_vorticity;
	float3*		_vorticity_force;

	// change:
	uint* sorted_grid_map;			// global_offset-to-particle_index
	uint* particle_grid_index;		// index of a particle in the grid
	uint* grid_offset;

	// prefix sums auxiliary buffers
	uint* aux_array1;
	uint* aux_scan1;
	uint* aux_array2;
	uint* aux_scan2;
};

struct ParticleParams {
	float3		boundary_min;
	float3		boundary_max;
	float3		initial_volume_min;
	float3		initial_volume_max;
	float3		grid_boundary_offset;
	float3		rotate_center;
	float3		gravity;

	float		particle_spacing;
	float		rest_density;
	float		simscale;
	float		particle_radius;
	float		smooth_radius;
	float		kernel_self;
	float		mass;
	float		gas_constant;
	float		viscosity;
	float		surface_tension_factor;
	float		damp;
	float		poly6kern;
	float		spikykern;
	float		lapkern;
	float		CubicSplineKern;
	float		GradCubicSplineKern;


	int			num_particle;
	int			max_num_particle;


	int			current_num_neighbor;
	int			max_num_neighbor;
	int			mean_neighbor_per_particle;
	int			max_neighbor_per_particle;

	int3		grid_resolution;
	
	// change:
	int3		grid_scan_max;

	int			grid_number;
	float		grid_radius;
	int			grid_search_offset[27];

	int3		mc_grid_resolution;
	int3		mc_grid_ver_resolution;
	float		mc_grid_radius;
	int			mc_grid_number;
	int			grid_ver_number;
	int			mc_grid_search_offset[729];


	int			particle_blocks;
	int			particle_threads;
	int			grid_blocks;
	int			grid_threads;
	int			mc_grid_blocks;
	int			mc_grid_threads;
	int			grid_ver_blocks;
	int			grid_ver_threads;
	int			size_Points;
	int			size_Grids;
	int			size_McGrids;
	int			size_GridVers;

	bool		_explicit;

	// temp container
	int			container_size;

	// miti-phase
	float		phase_density[MAX_PHASE_NUMBER];			// rho_k			rest density for each single phase
	float		phase_mass[MAX_PHASE_NUMBER];				// m_k				rest mass for each single phase
	float		phase_visc[MAX_PHASE_NUMBER];				// mu_k				phase viscosity for each single phase
	float3		phase_color[MAX_PHASE_NUMBER];				// color_k			phase color for rendering
	float		tau;										// tau				drift velocity coefficient
	float		sigma;										// sigma			drift velocity coefficient
	int			phase_number;								//					number of phases
	bool		miscible;									//					miscible or not

	// pressure-bound
	float		gamma;
	float		beta;
	float		omega;

	// test
	int			test_index;

	// surface factor
	float		smoothing_factor;
	float		diag_k_s;
	float		diag_k_n;
	float		diag_k_r;
	int			search_num_threshold;
	float		itp_radius;
	float		anisotropic_radius;

	// bubble
	float		theta1;
	float		theta2;
};

#ifndef CUDA_KERNEL
__global__ void AllocateGrid(bufList buf, int numParticles);
// muti-phase
__global__ void ComputeMFDensity(bufList buf, int numParticles);
__global__ void ApplyAlpha(bufList buf, int numParticles);
__global__ void ComputeDriftVelocity(bufList buf, int numParticles);
__global__ void ComputeAlphaAdvance(bufList buf, int numParticles, float time_step);
__global__ void ComputeCorrection(bufList buf, int numParticles);
__global__ void ComputeTDM(bufList buf, int numParticles);
__global__ void UpdateMFParticles(bufList buf, int numParticles, float time_step);
// peridynamics
__global__ void ComputeTensorK(bufList buf, int numParticles);
__global__ void ComputeTensorF(bufList buf, int numParticles, float time_step);
// muti-phase in peridynamics
__global__ void ComputeDriftVelocityPeridynamics(bufList buf, int numParticles, float bound_vel, float factor);
__global__ void ComputeSdVPeridynamics(bufList buf, int numParticles);
__global__ void ComputeAlphaAdvancePeridynamics(bufList buf, int numParticles, float time_step);
__global__ void ComputeAlphaTransportIter(bufList buf, int numParticles, float time_step, float factor);
__global__ void UpdateLambda(bufList buf, int numParticles);
__global__ void AlphaCorrection(bufList buf, int numParticles);
__global__ void ComputeTDMPeridynamics(bufList buf, int numParticles, float time_step, float surface_factor);
__global__ void ComputeMFSourceTermPeridynamics(bufList buf, int numParticles, float time_step);
__global__ void ComputeMFDiagElementPeridynamics(bufList buf, int numParticles, float time_step);
__global__ void ComputeMFPressureAccelPeridynamics(bufList buf, int numParticles);
__global__ void MFPressureUpdatePeridynamics(bufList buf, int numParticles, float time_step);
__global__ void ComputeCorrectionVolume(bufList buf, int numParticles, float time_step);
__global__ void AlphaBanlanceUsingMassFraction(bufList buf, int numParticles, float time_step);
// particle bound
__global__ void ComputePhiParticleBound(bufList buf, int numParticles);
__global__ void ComputeMFDensityParticleBound(bufList buf, int numParticles, bool transferAlpha);
__global__ void ContributePressure(bufList buf, int numParticles);
__global__ void ComputeBoundPressure(bufList buf, int numParticles);
__global__ void AdvanceParticleBound(bufList buf, int numParticles, float time_step);
// generate mf particles
__global__ void GenerateParticles(bufList buf, int numParticles, int begin, int N, float time_step, int Generate_pos, float3 state_point, float GenFrameRate);
__global__ void ChemicalReaction(bufList buf, int numParticles, float ReactionSpeed);
__global__ void ConcentrationDecay(bufList buf, int numParticles);
// rigid body
__global__ void UpdateRigidBody(bufList buf, int numParticles, float time_step, float omega);
__global__ void UpdateRigidBodyDrift(bufList buf, int numParticles, float time_step);
__global__ void RigidBodyTransition(bufList buf, int numParticles);
__global__ void UpdateUpBound(bufList buf, int numParticles, float time_step);
// density scalar field
__global__ void GetAverageKernelPos(bufList buf, int numParticles);
__global__ void ComputePosCovariance(bufList buf, int numParticles);
__global__ void ComputeShapeMatG(bufList buf, int numParticles);
__global__ void ComputeShapeMatGSVD(bufList buf, int numParticles);
__global__ void CalDensityScalarFieldParticle(bufList buf, int numParticles);
// mc grid
__global__ void AllocateMcGrid(bufList buf, int numParticles);
__global__ void ParticleScalarvalueToGrid(bufList buf, int numMcGridver);
// reset particles
__global__ void ResetParticles(bufList buf, int numParticles);
// secondary bubbles
__global__ void ComputeBubbleAlpha(bufList buf, int numParticles);
__global__ void InsertBubbleParticle(bufList buf, int numParticles);
__global__ void UpdatePosBubble(bufList buf, int numParticles, float time_step);
__global__ void DeleteBubbleParticle(bufList buf, int numParticles);
// surface tension
__global__ void ComputeSfParticlePhase(bufList buf, int numParticles);
__global__ void ComputeSurfaceTensionParticle(bufList buf, int numParticles);
// vorticity
__global__ void ComputeVorticityParticle(bufList buf, int numParticles);
__global__ void ComputeForceVorticity(bufList buf, int numParticles);

// change:
__global__ void test(bufList buf, int numParticles);
__global__ void insertParticles(bufList buf, int numParticles);
__global__ void prefixFixup(uint* input, uint* aux, int len);
__global__ void prefixSum(uint* input, uint* output, uint* aux, int len, int zeroff);
__global__ void countingSortFull(bufList buf, bufList temp, int numParticles);

__global__ void computePhi(bufList buf, int numParticles);
__global__ void computeDensity(bufList buf, int numParticles);
__global__ void computeK(bufList buf, int numParticles);
__global__ void computeF(bufList buf, int numParticles, float time_step);
__global__ void setAlpha(bufList buf, int numParticles);
__global__ void computeDriftVel(bufList buf, int numParticles);
__global__ void computeDelAlpha(bufList buf, int numParticles, float time_step);
__global__ void computeLambda(bufList buf, int numParticles);
__global__ void computeCorrection(bufList buf, int numParticles, float time_step);
__global__ void normalizeAlpha(bufList buf, int numParticles, float time_step);
__global__ void computeForce(bufList buf, int numParticles, float time_step);
__global__ void advanceParticles(bufList buf, int numParticles, float time_step, int frame);
#endif

#define EPSILON				0.00001f
#define GRID_UCHAR			0xFF
#define GRID_UNDEF			0xFFFFFFFF

#endif // !DEF_KERN_CUDA
