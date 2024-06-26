#define MAX_PARAM_NUM 80  // max param num

// particle system
// ---------------
#define GRID_UCHAR 0xFF
#define GRID_UNDEF 0xFFFFFFFF

// float param
// -----------
#define PSPACINGREALWORLD 1      // particle spacing in real world
#define PSPACINGGRAPHICSWORLD 2  // particle spacing in graphicworld
#define PRESTDENSITY 3           // rest density of particle
#define MAXNUMBER 4              // max particle number
#define SIMSCALE 5               // simscale of simulation
#define PARTICLERADIUS 6         // particle radius
#define TIMESTEP 7               // time step of simulation
#define GRIDRADIUS 8             // grid radius
#define MAXGRIDNUMBER 9          // max number of the grid
#define MINNEIGHBORNUMBER 10     // the minimum number of neighbor particles
#define SMOOTHRADIUS 11          // smooth radius used in kernel function
#define MAXNEIGHBORPARTICLE 12   // the maximum number of neighbor particles
#define PKERNELSELF 13           // the value of kernelm4
#define PMASS 14                 // particle mass
#define PGASCONSTANT 15          // gas particle constant
#define PVISC 16                 // particle viscosity
#define FMASS 17                 // particle mass for the first phase
#define SMASS 18                 // particle mass for the second phase
#define TMASS 19                 // particle mass for the third phase
#define FDENSITY 20              // first-phase-density
#define SDENSITY 21              // second-phase-density
#define TDENSITY 22              // third-phase-density
#define FVISC 23                 // first-phase-viscosity
#define SVISC 24                 // second-phase-viscosity
#define TVISC 25                 // third-phase-viscosity
#define PHASENUMBER 26           // number of phases
#define DRIFTVELTAU 27           // drift velocity coefficient tau
#define DRIFTVELSIGMA 28         // drift velocity coefficient sigma
#define MODE 29                  // 0 - Standard explicit PDMF, 1 - PDMF without driftvel [2021 Y.Jiang]
#define BOUNDOFFSET 30           // boundary particle width
#define BOUNDGAMMA 31            // pressure-bound parameter
#define BOUNDBETA 32             // pressure-bound parameter
#define BOUNDOMEGA 33            // pressure-bound parameter
#define TESTINDEX 34             // param for test
#define KD 35                    // muti-phase drag force
#define MAXLOADNUMBER 36         // load fluid particle used for generating
#define POINTNUMBER 37           // model point cloud number
#define RIGIDSCALE 38            // scale of the load rigid
#define CONTAINERSIZE 39         // size of the temp container storing neighbor values
#define SMOOTHFACTOR 40          // smoothing factor for surface smoothing
#define FACTORKN 41              // factor k_n of calculating shape matrix G
#define FACTORKR 42              // factor k_r of calculating shape matrix G
#define FACTORKS 43              // factor k_s of calculating shape matrix G
#define NEIGHBORTHRESHOLD 44     // neighbor search threshold for matrix G
#define SURFACETENSIONFACTOR 45  // factor of calculating surface tension
#define MCUBETHRESHOLD 46        // marching cube threshold
#define MCUBEVOXEL 47            // marching cube voxel
#define POINTNUMBERROTATE 48     // model point cloud number 2
#define RIGIDSCALEROTATE 49      // point cloud scale factor 2
#define INTERPOLATERADIUS 50     // interpolate radius for marching cube
#define ANISOTROPICRADIUS 51     // anisotropic radius for ellipsoid kernel
#define GENERATEFRAMERATE 52     // frame rate of particle generation
#define GENERATENUM 53           // unit num of particle generation in a frame
#define GENERATEPOSENUM 54       // enum of generate pos
#define RIGIDROTATEOMEGA 55      // angular velocity of the rotate rigid
#define REACTIONRATE 56          // chemical reaction rate

// vec param
// ---------
#define BOUNDARYMIN 1              // the boundary coordinate at left-bottom corner
#define BOUNDARYMAX 2              // the boundary coordinate at right-top corner
#define INITIALVOLUMEMIN 3         // the initial volume coordinate at left-bottom corner
#define INITIALVOLUMEMAX 4         // the initial volume coordinate at right-top corner
#define GRIDBOUNDARYOFFSET 5       // the offset of grid boundary
#define GRAVITY 6                  // the gravity used in simulation
#define INITIALVOLUMEMINPHASE 7    // the initial volume coordinate at left-bottom corner of the multi_fluid bound
#define INITIALVOLUMEMAXPHASE 8    // the initial volume coordinate at right-top corner of the multi_fluid bound
#define RIGIDVOLUMEMIN 9           // the initial volume coordinate at left-bottom corner of the rigid bound
#define RIGIDVOLUMEMAX 10          // the initial volume coordinate at right-top corner of the rigid bound
#define INITIALVOLUMEMINPHASES 11  // the initial volume coordinate at left-bottom corner of the another multi_fluid bound
#define INITIALVOLUMEMAXPHASES 12  // the initial volume coordinate at right-top corner of the another multi_fluid bound
#define RIGIDCENTER 13             // rigid center set for rigid particle
#define RIGIDDRIFT 14              // rigid drift set for rigid particle
#define RIGIDDRIFTROTATE 15        // rigid drift rotate set for rigid particle
#define RIGIDSCALE 17              // scale of rigid object
#define RIGIDSCALEROTATE 18        // scale rotate of rigid object

// vec_int param
// -------------
#define GRIDRESOLUTION 1  // grid resolution

// bool param
// ----------
#define LOADRIGID 1        // if load rigid
#define LOADBOUND 2        // if load boundary
#define MUTIPHASE 3        // if simulate multiphase scene
#define IMPLICIT 4         // if use implicit timestep
#define MISCIBLE 5         // if is miscible
#define LOADRIGIDROTATE 6  // if load rotate rigid
#define HIGHRESOLUTION 13
#define LOADGENERATEFORM 19
#define RENDERRIGID 28
#define DISABELTRANSFER 29

// m_multi_phase
#define MFDENSITY 1    // for multiphase fluid density
#define MFVISCOSITY 2  // for multiphase fluid viscosity
#define MFRESTMASS 3   // for multiphase fluid rest mass

// m_multi_phase_vec
#define MFCOLOR 1  // for multiphase fluid particle color