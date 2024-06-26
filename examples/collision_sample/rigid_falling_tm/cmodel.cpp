//#define B150 1

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "crigid.h"
#include "pair.h"
#include "plane.h"
#include "tmbvh.h"
#include "qbvh.h"
#include "contact.h"
#include <stdio.h>
#include <omp.h>

#include <mutex>
#include "cmodel.h"

//#include "collision/interface/mesh_mesh_collision_detect.hpp"
// mutex to lock critical region
std::mutex mylock;
int maxTriPairs = 0;

id_pair* sPairSet = nullptr;
id_pair* sRigSet = nullptr;
bool gFlat = false;

#ifdef PROF
//#if defined(PROF) || defined(GPU)

class myTimer {
public:
	myTimer(const char* msgIn) {}
};


class myTimer2
{
public:
	myTimer2(const char* msgIn) {}

	void print() {}

	void inc(double delta) {}
};

#else

class myTimer
{
	double t0;
	char msg[512];
public:
	myTimer(const char* msgIn) {
		t0 = omp_get_wtime();
		strcpy(msg, msgIn);
	}
	~myTimer() {
		double tdelta = omp_get_wtime() - t0;
		printf("%s: %2.5f s\n", msg, tdelta);
	}
};

class myTimer2
{
	double dt;
	char msg[512];
public:
	myTimer2(const char* msgIn) {
		dt = 0;
		strcpy(msg, msgIn);
	}

	void print() {
		printf("%s: %2.5f s\n", msg, dt);
	}

	void inc(double delta) {
		dt += delta;
	}
};

#endif

#define BUNNY_SCALE 4.f

#pragma warning(disable: 4996)

extern void drawRigid(crigid*, bool cyl, int level, vec3f &);
extern void drawPlanes(bool);
extern float getLargestVelocityNorm(crigid* body1, crigid* body2);

#define SIMDSQRT12 float(0.7071067811865475244008443621048490)
#define RecipSqrt(x) ((float)(float(1.0)/sqrtf(float(x))))		/* reciprocal square root */
static int stepIdx = 0;

enum AnisotropicFrictionFlags
{
	CF_ANISOTROPIC_FRICTION_DISABLED = 0,
	CF_ANISOTROPIC_FRICTION = 1,
	CF_ANISOTROPIC_ROLLING_FRICTION = 2
};

__forceinline float restitutionCurve(float rel_vel, float restitution)
{
	return restitution * -rel_vel;
}

__forceinline void PlaneSpace1(const vec3f& n, vec3f& p, vec3f& q)
{
	float a = fabs(n[2]);
	float b = SIMDSQRT12;

	if (fabs(n[2]) > SIMDSQRT12) {
		// choose p in y-z plane
		float a = n[1] * n[1] + n[2] * n[2];
		float k = RecipSqrt(a);
		p[0] = 0;
		p[1] = -n[2] * k;
		p[2] = n[1] * k;
		// set q = n x p
		q[0] = a * k;
		q[1] = -n[0] * p[2];
		q[2] = n[0] * p[1];
  }
	else {
		// choose p in x-y plane
		float a = n[0] * n[0] + n[1] * n[1];
		float k = RecipSqrt(a);
		p[0] = -n[1] * k;
		p[1] = n[0] * k;
		p[2] = 0;
		// set q = n x p
		q[0] = -n[2] * p[1];
		q[1] = n[2] * p[0];
		q[2] = a * k;
	}
}

#ifdef GPU
extern void pushMesh1(int);
extern void pushRigids(int, void*);
extern void getPairsGPU(void* data, int num, void *data2);
extern void pushRigidPairsGPU(int num, void* data);

extern int checkTriTriCDGPU(const void* offIn, const void* rotIn, REAL margin,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2);

extern int checkTriTriCDFusedGPU(REAL margin,
	REAL x0, REAL y0, REAL z0,
	REAL x1, REAL y1, REAL z1,
	REAL x2, REAL y2, REAL z2);

extern int checkRigidRigidCDGPU(REAL margin);
extern void clearGPU();

//extern void pushVtxSet(int num, void* data);
//extern void pushRigid1(int);
//extern void checkBVH();

extern void gpuTimerBegin();
extern float gpuTimerEnd(char*, bool);

#endif

BOX g_box;
BOX g_projBx;
REAL g_time = 0.0f;

extern bool verb;

vec3f projDir(0.0f, -1.0f, 0.0f);
REAL maxDist = 20.0;
static int sidx = 0;


/// Calc a plane from a triangle edge an a normal. plane is a vec4f
__forceinline void get_edge_plane2(const vec3f& e1, const vec3f& e2, const vec3f& normal, vec4f& plane)
{
	vec3f planenormal = (e2 - e1).cross(normal);
	planenormal.normalize();
	plane.setValue(planenormal[0], planenormal[1], planenormal[2], e2.dot(planenormal));
}

__forceinline float distance_point_plane(const vec4f& plane, const vec3f& point)
{
	return point.dot(plane.xyz()) - plane.w();
}


/*! Vector blending
Takes two vectors a, b, blends them together*/
__forceinline void get_vec_blend(vec3f& vr, const vec3f& va, const vec3f& vb, float blend_factor)
{
	vr = (1 - blend_factor) * va + blend_factor * vb;
}

//! This function calcs the distance from a 3D plane
__forceinline void get_plane_clip_polygon_collect(
	const vec3f& point0,
	const vec3f& point1,
	float dist0,
	float dist1,
	vec3f* clipped,
	int& clipped_count)
{
	bool _prevclassif = (dist0 > FLT_EPSILON);
	bool _classif = (dist1 > FLT_EPSILON);
	if (_classif != _prevclassif)
	{
		float blendfactor = -dist0 / (dist1 - dist0);
		get_vec_blend(clipped[clipped_count], point0, point1, blendfactor);
		clipped_count++;
	}
	if (!_classif)
	{
		clipped[clipped_count] = point1;
		clipped_count++;
	}
}


//! Clips a polygon by a plane
/*!
*\return The count of the clipped counts
*/
__forceinline int get_plane_clip_polygon(
	const vec4f& plane,
	const vec3f* polygon_points,
	int polygon_point_count,
	vec3f* clipped)
{
	int clipped_count = 0;


	//clip first point
	float firstdist = distance_point_plane(plane, polygon_points[0]);;
	if (!(firstdist > FLT_EPSILON))
	{
		clipped[clipped_count] = polygon_points[0];
		clipped_count++;
	}

	float olddist = firstdist;
	for (int i = 1; i < polygon_point_count; i++)
	{
		float dist = distance_point_plane(plane, polygon_points[i]);

		get_plane_clip_polygon_collect(
			polygon_points[i - 1], polygon_points[i],
			olddist,
			dist,
			clipped,
			clipped_count);


		olddist = dist;
	}

	//RETURN TO FIRST  point

	get_plane_clip_polygon_collect(
		polygon_points[polygon_point_count - 1], polygon_points[0],
		olddist,
		firstdist,
		clipped,
		clipped_count);

	return clipped_count;
}

//! Clips a polygon by a plane
/*!
*\param clipped must be an array of 16 points.
*\return The count of the clipped counts
*/
__forceinline int get_plane_clip_triangle(
	const vec4f& plane,
	const vec3f& point0,
	const vec3f& point1,
	const vec3f& point2,
	vec3f* clipped // an allocated array of 16 points at least
)
{
	int clipped_count = 0;

	//clip first point0
	float firstdist = distance_point_plane(plane, point0);;
	if (!(firstdist > FLT_EPSILON))
	{
		clipped[clipped_count] = point0;
		clipped_count++;
	}

	// point 1
	float olddist = firstdist;
	float dist = distance_point_plane(plane, point1);

	get_plane_clip_polygon_collect(
		point0, point1,
		olddist,
		dist,
		clipped,
		clipped_count);

	olddist = dist;


	// point 2
	dist = distance_point_plane(plane, point2);

	get_plane_clip_polygon_collect(
		point1, point2,
		olddist,
		dist,
		clipped,
		clipped_count);
	olddist = dist;



	//RETURN TO FIRST  point0
	get_plane_clip_polygon_collect(
		point2, point0,
		olddist,
		firstdist,
		clipped,
		clipped_count);

	return clipped_count;
}

#define MAX_TRI_CLIPPING 16

//! Structure for collision
struct ctriangleContact
{
	float m_penetration_depth;
	int m_point_count;
	vec4f m_separating_normal;
	vec3f m_points[MAX_TRI_CLIPPING];

	__forceinline void copy_from(const ctriangleContact& other)
	{
		m_penetration_depth = other.m_penetration_depth;
		m_separating_normal = other.m_separating_normal;
		m_point_count = other.m_point_count;
		int i = m_point_count;
		while (i--)
		{
			m_points[i] = other.m_points[i];
		}
	}

	ctriangleContact()
	{
	}

	ctriangleContact(const ctriangleContact& other)
	{
		copy_from(other);
	}

	//! classify points that are closer
	void merge_points(const vec4f& plane, float margin, const vec3f* points, int point_count)
	{
		m_point_count = 0;
		m_penetration_depth = -1000.0f;

		int point_indices[MAX_TRI_CLIPPING];

		int _k;

		for (_k = 0; _k < point_count; _k++)
		{
			float _dist = -distance_point_plane(plane, points[_k]) + margin;

			if (_dist >= 0.0f)
			{
				if (_dist > m_penetration_depth)
				{
					m_penetration_depth = _dist;
					point_indices[0] = _k;
					m_point_count = 1;
				}
				else if ((_dist + FLT_EPSILON) >= m_penetration_depth)
				{
					point_indices[m_point_count] = _k;
					m_point_count++;
				}
			}
		}

		for (_k = 0; _k < m_point_count; _k++)
		{
			m_points[_k] = points[point_indices[_k]];
		}
	}
};

class cprimitiveTriangle
{
public:
	vec3f m_vertices[3];
	vec4f m_plane;
	float m_margin;
	float m_dummy;

	cprimitiveTriangle() :m_margin(0.01f)
	{
	}


	__forceinline void buildTriPlane()
	{
		vec3f normal = (m_vertices[1] - m_vertices[0]).cross(m_vertices[2] - m_vertices[0]);
		normal.normalize();
		m_plane.setValue(normal[0], normal[1], normal[2], m_vertices[0].dot(normal));
	}

	//! Test if triangles could collide
	bool overlap_test_conservative(const cprimitiveTriangle& other)
	{
		float total_margin = m_margin + other.m_margin;
		// classify points on other triangle
		float dis0 = distance_point_plane(m_plane, other.m_vertices[0]) - total_margin;

		float dis1 = distance_point_plane(m_plane, other.m_vertices[1]) - total_margin;

		float dis2 = distance_point_plane(m_plane, other.m_vertices[2]) - total_margin;

		if (dis0 > 0.0f && dis1 > 0.0f && dis2 > 0.0f) return false;

		// classify points on this triangle
		dis0 = distance_point_plane(other.m_plane, m_vertices[0]) - total_margin;

		dis1 = distance_point_plane(other.m_plane, m_vertices[1]) - total_margin;

		dis2 = distance_point_plane(other.m_plane, m_vertices[2]) - total_margin;

		if (dis0 > 0.0f && dis1 > 0.0f && dis2 > 0.0f) return false;

		return true;
	}

	//! Calcs the plane which is paralele to the edge and perpendicular to the triangle plane
	/*!
	\pre this triangle must have its plane calculated.
	*/
	__forceinline void get_edge_plane(int edge_index, vec4f& plane)  const
	{
		const vec3f& e0 = m_vertices[edge_index];
		const vec3f& e1 = m_vertices[(edge_index + 1) % 3];
		get_edge_plane2(e0, e1, m_plane.xyz(), plane);
	}

	void applyTransform(const transf & t)
	{
		m_vertices[0] = t(m_vertices[0]);
		m_vertices[1] = t(m_vertices[1]);
		m_vertices[2] = t(m_vertices[2]);
	}

	//! Clips the triangle against this
	/*!
	\pre clipped_points must have MAX_TRI_CLIPPING size, and this triangle must have its plane calculated.
	\return the number of clipped points
	*/
	int clip_triangle(cprimitiveTriangle& other, vec3f* clipped_points)
	{
		// edge 0

		vec3f temp_points[MAX_TRI_CLIPPING];


		vec4f edgeplane;

		get_edge_plane(0, edgeplane);


		int clipped_count = get_plane_clip_triangle(
			edgeplane, other.m_vertices[0], other.m_vertices[1], other.m_vertices[2], temp_points);

		if (clipped_count == 0) return 0;

		vec3f temp_points1[MAX_TRI_CLIPPING];


		// edge 1
		get_edge_plane(1, edgeplane);


		clipped_count = get_plane_clip_polygon(edgeplane, temp_points, clipped_count, temp_points1);

		if (clipped_count == 0) return 0;

		// edge 2
		get_edge_plane(2, edgeplane);

		clipped_count = get_plane_clip_polygon(
			edgeplane, temp_points1, clipped_count, clipped_points);

		return clipped_count;
	}


	//! Find collision using the clipping method
	/*!
	\pre this triangle and other must have their triangles calculated
	*/
	bool find_triangle_collision_clip_method(cprimitiveTriangle& other, ctriangleContact& contacts)
	{
		float margin = m_margin + other.m_margin;

		vec3f clipped_points[MAX_TRI_CLIPPING];
		int clipped_count;
		//create planes
		// plane v vs U points

		ctriangleContact contacts1;

		contacts1.m_separating_normal = m_plane;


		clipped_count = clip_triangle(other, clipped_points);

		if (clipped_count == 0)
		{
			return false;//Reject
		}

		//find most deep interval face1
		contacts1.merge_points(contacts1.m_separating_normal, margin, clipped_points, clipped_count);
		if (contacts1.m_point_count == 0) return false; // too far
		//Normal pointing to this triangle
		contacts1.m_separating_normal *= -1.f;


		//Clip tri1 by tri2 edges
		ctriangleContact contacts2;
		contacts2.m_separating_normal = other.m_plane;

		clipped_count = other.clip_triangle(*this, clipped_points);

		if (clipped_count == 0)
		{
			return false;//Reject
		}

		//find most deep interval face1
		contacts2.merge_points(contacts2.m_separating_normal, margin, clipped_points, clipped_count);
		if (contacts2.m_point_count == 0) return false; // too far




		////check most dir for contacts
		if (contacts2.m_penetration_depth < contacts1.m_penetration_depth)
		{
			contacts.copy_from(contacts2);
		}
		else
		{
			contacts.copy_from(contacts1);
		}
		return true;
	}

};

inline void get_primitive_triangle(crigid *rig, int prim_index, cprimitiveTriangle& triangle)
{
	tri3f* tris = rig->getMesh()->getTris();
	vec3f* vtxs = rig->getMesh()->getVtxs();

	tri3f& t = tris[prim_index];
	triangle.m_vertices[0] = vtxs[t.id0()];
	triangle.m_vertices[1] = vtxs[t.id1()];
	triangle.m_vertices[2] = vtxs[t.id2()];
	triangle.m_margin = 0.01;
}

///The csolverBody is an internal datastructure for the constraint solver. Only necessary data is packed to increase cache coherence/performance.
struct csolverBody
{
	transf		m_worldTransform;
	vec3f		m_deltaLinearVelocity;
	vec3f		m_deltaAngularVelocity;
	vec3f		m_invMass;
	vec3f		m_pushVelocity;
	vec3f		m_turnVelocity;
	vec3f		m_linearVelocity;
	vec3f		m_angularVelocity;

	crigid* m_originalBody;

	void	setWorldTransform(const transf& worldTransform)
	{
		m_worldTransform = worldTransform;
	}

	const transf& getWorldTransform() const
	{
		return m_worldTransform;
	}

	__forceinline void	getVelocityInLocalPointObsolete(const vec3f& rel_pos, vec3f& velocity) const
	{
		if (m_originalBody)
			velocity = m_linearVelocity + m_deltaLinearVelocity + (m_angularVelocity + m_deltaAngularVelocity).cross(rel_pos);
		else
			velocity = vec3f(0, 0, 0);
	}

	__forceinline void	getAngularVelocity(vec3f& angVel) const
	{
		if (m_originalBody)
			angVel = m_angularVelocity + m_deltaAngularVelocity;
		else
			angVel = vec3f(0, 0, 0);
	}


	//Optimization for the iterative solver: avoid calculating constant terms involving inertia, normal, relative position
	__forceinline void applyImpulse(const vec3f& linearComponent, const vec3f& angularComponent, const float impulseMagnitude)
	{
		if (m_originalBody)
		{
			m_deltaLinearVelocity += linearComponent * impulseMagnitude;
			m_deltaAngularVelocity += angularComponent * impulseMagnitude;
		}
	}

	__forceinline void internalApplyPushImpulse(const vec3f& linearComponent, const vec3f& angularComponent, float impulseMagnitude)
	{
		if (m_originalBody)
		{
			m_pushVelocity += linearComponent * impulseMagnitude;
			m_turnVelocity += angularComponent * impulseMagnitude;
		}
	}



	const vec3f& getDeltaLinearVelocity() const
	{
		return m_deltaLinearVelocity;
	}

	const vec3f& getDeltaAngularVelocity() const
	{
		return m_deltaAngularVelocity;
	}

	const vec3f& getPushVelocity() const
	{
		return m_pushVelocity;
	}

	const vec3f& getTurnVelocity() const
	{
		return m_turnVelocity;
	}


	////////////////////////////////////////////////
	///some internal methods, don't use them

	vec3f& internalGetDeltaLinearVelocity()
	{
		return m_deltaLinearVelocity;
	}

	vec3f& internalGetDeltaAngularVelocity()
	{
		return m_deltaAngularVelocity;
	}

	const vec3f& internalGetInvMass() const
	{
		return m_invMass;
	}

	void internalSetInvMass(const vec3f& invMass)
	{
		m_invMass = invMass;
	}

	vec3f& internalGetPushVelocity()
	{
		return m_pushVelocity;
	}

	vec3f& internalGetTurnVelocity()
	{
		return m_turnVelocity;
	}

	__forceinline void	internalGetVelocityInLocalPointObsolete(const vec3f& rel_pos, vec3f& velocity) const
	{
		velocity = m_linearVelocity + m_deltaLinearVelocity + (m_angularVelocity + m_deltaAngularVelocity).cross(rel_pos);
	}

	__forceinline void	internalGetAngularVelocity(vec3f& angVel) const
	{
		angVel = m_angularVelocity + m_deltaAngularVelocity;
	}


	//Optimization for the iterative solver: avoid calculating constant terms involving inertia, normal, relative position
	__forceinline void internalApplyImpulse(const vec3f& linearComponent, const vec3f& angularComponent, const float impulseMagnitude)
	{
		if (m_originalBody)
		{
			m_deltaLinearVelocity += linearComponent * impulseMagnitude ;
			m_deltaAngularVelocity += angularComponent * impulseMagnitude;
		}
	}




	void	writebackVelocity()
	{
		if (m_originalBody)
		{
			m_linearVelocity += m_deltaLinearVelocity;
			m_angularVelocity += m_deltaAngularVelocity;

			//m_originalBody->setCompanionId(-1);
		}
	}


	void	writebackVelocityAndTransform(float timeStep, float splitImpulseTurnErp)
	{
		(void)timeStep;
		if (m_originalBody)
		{
			m_linearVelocity += m_deltaLinearVelocity;
			m_angularVelocity += m_deltaAngularVelocity;

			//correct the position/orientation based on push/turn recovery
			transf newTransform;
			if (m_pushVelocity[0] != 0.f || m_pushVelocity[1] != 0 || m_pushVelocity[2] != 0 || m_turnVelocity[0] != 0.f || m_turnVelocity[1] != 0 || m_turnVelocity[2] != 0)
			{
				//	btQuaternion orn = m_worldTransform.getRotation();
				TransformUtil::integrateTransform(m_worldTransform, m_pushVelocity, m_turnVelocity * splitImpulseTurnErp, timeStep, newTransform);
				m_worldTransform = newTransform;
			}
			//m_worldTransform.setRotation(orn);
			//m_originalBody->setCompanionId(-1);
		}
	}
};


enum	csolverMode
{
	SOLVER_RANDMIZE_ORDER = 1,
	SOLVER_FRICTION_SEPARATE = 2,
	SOLVER_USE_WARMSTARTING = 4,
	SOLVER_USE_2_FRICTION_DIRECTIONS = 16,
	SOLVER_ENABLE_FRICTION_DIRECTION_CACHING = 32,
	SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION = 64,
	SOLVER_CACHE_FRIENDLY = 128,
	SOLVER_SIMD = 256,
	SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS = 512,
	SOLVER_ALLOW_ZERO_LENGTH_FRICTION_DIRECTIONS = 1024
};

struct	csolverConstraint
{
	vec3f m_relpos1CrossNormal;
	vec3f m_contactNormal;

	vec3f m_relpos2CrossNormal;
	//vec3f		m_contactNormal2;//usually m_contactNormal2 == -m_contactNormal

	vec3f m_angularComponentA;
	vec3f m_angularComponentB;

	float m_appliedPushImpulse;
	float m_appliedImpulse;

	float	m_friction;
	float	m_jacDiagABInv;
	float	m_rhs;
	float	m_cfm;

	float	m_lowerLimit;
	float	m_upperLimit;
	float	m_rhsPenetration;
	union
	{
		void* m_originalContactPoint;
		float	m_unusedPadding4;
	};

	int m_overrideNumSolverIterations;
	int m_frictionIndex;
	int m_solverBodyIdA;
	int m_solverBodyIdB;

	enum csolverConstraintType
	{
		SOLVER_CONTACT_1D = 0,
		SOLVER_FRICTION_1D
	};
};

class cscene {
	std::vector<kmesh*> _meshs;
	std::vector<crigid*> _rigids;
	std::vector<cplane*> _plns;
	std::vector<manifold*> _predictiveManifolds;
	std::vector<id_pair> _rigid_pairs, _rigid_pln_pairs;
    //Physika::MeshMeshCollisionDetect solver;

	//for GPU updating...
	std::vector< transf> _trfs;
public:
	~cscene() { clear(); }

	void clear() {
		for (auto r : _rigids)
			delete r;

		for (auto p : _plns)
			delete p;

		for (auto m : _meshs)
			delete m;

		_meshs.clear();
		_plns.clear();
		_rigids.clear();
	}



	void draw(int level) {
		for (auto r : _rigids) {
			drawRigid(r, false, level, vec3f());
		}
		drawPlanes(gFlat);
	}

	void addMesh(kmesh* km) {
		_meshs.push_back(km);
	}

	void addPlane(cplane* pl) {
		_plns.push_back(pl);
	}

	void addRigid(crigid* rig) {
		_rigids.push_back(rig);
	}

	bool output(const char* fname) {
		FILE* fp = fopen(fname, "wt");
		if (fp == NULL)
			return false;

		fprintf(fp, "%zd\n", _rigids.size());
		for (int i = 0; i < _rigids.size(); i++) {
			transf& trf = _rigids[i]->getWorldTransform();
			vec3f& off = trf.getOrigin();
			quaternion q = trf.getRotation();
			fprintf(fp, "%lf, %lf, %lf\n", off.x, off.y, off.z);
			fprintf(fp, "%lf, %lf, %lf, %lf\n", q.x(), q.y(), q.z(), q.w());
		}
		fclose(fp);
		return true;
	}

	bool input(const char* fname) {
		FILE* fp = fopen(fname, "rt");
		if (fp == NULL)
			return false;

		int num = 0;
		char buffer[512];
		fgets(buffer, 512, fp);
		sscanf(buffer, "%d", &num);
		if (num != _rigids.size())
			return false;

		for (int i = 0; i < _rigids.size(); i++) {
			transf& trf = _rigids[i]->getWorldTransform();

			fgets(buffer, 512, fp);
			double x, y, z, w;
			sscanf(buffer, "%lf, %lf, %lf", &x, &y, &z);
			vec3f off(x, y, z);
			fgets(buffer, 512, fp);
			sscanf(buffer, "%lf, %lf, %lf, %lf", &x, &y, &z, &w);
			quaternion q(x, y, z, w);

			trf.setOrigin(off);
			trf.setRotation(q);
		}
		fclose(fp);
		return true;
	}

#ifdef GPU
	void push2GPU()
	{
		printf("cpu size = %zd, %zd, %zd, %zd, %zd, %zd\n",
			sizeof(BOX), sizeof(vec3f), sizeof(matrix3f), sizeof(bvh_node), sizeof(transf), sizeof(qbvh_node));

		pushMesh1(_meshs.size());

		for (int i = 0; i < _meshs.size(); i++) {
			_meshs[i]->push2G(i);
		}

		update2GPU();
	}

	void update2GPU()
	{
		_trfs.resize(_rigids.size());
		for (int i = 0; i < _rigids.size(); i++) {
			_trfs[i] = _rigids[i]->getWorldTransform();
		}
		pushRigids(_rigids.size(), _trfs.data());
	}

#if 0
	void checkVtxRayGPU(crigid* rig, int i, REAL& dMin);
	void checkVtxRayInvGPU(crigid* rig, crigid* obj, int i, REAL& dMin);

	void sweepQueryGPU() {
		_qryRets.clear();

		//broad phrase
		int num = _rigids.size();
		crigid* o = _rigids[0];
		for (int i = 1; i < num; i++) {
			crigid* r = _rigids[i];

			BOX bx = r->bound();
			if (!bx.overlaps(g_projBx)) {
				//broad prhase culling
				continue;
			}

			//narrow phrase
			REAL dMin = maxDist;

			//dMin = checkVtxRay(r);
			checkVtxRayGPU(r, i, dMin);
			checkVtxRayInvGPU(r, o, i, dMin);

			if (dMin < maxDist) {
				if (verb)
					printf("hit distance = %lf\n", dMin);

				_qryRets.push_back(dMin);//unsafe for multiThread
			}
		}
	}
#endif
#endif

	//////////////////////////////////////////////////////////////////////////////
	int	stepSimulation(float timeStep, int maxSubSteps = 1, float fixedTimeStep = float(1.) / float(60.))
	{
		int numSimulationSubSteps = timeStep / fixedTimeStep;
		//clamp the number of substeps, to prevent simulation grinding spiralling down to a halt
		int clampedSimulationSteps = (numSimulationSubSteps > maxSubSteps) ? maxSubSteps : numSimulationSubSteps;

		saveKinematicState(fixedTimeStep * clampedSimulationSteps);
		applyGravity();

		for (int i = 0; i < clampedSimulationSteps; i++)
		{
			internalSingleStepSimulation(fixedTimeStep);
			synchronizeMotionStates();
		}
		clearForces();

#ifdef GPU
		update2GPU();
#endif
		return clampedSimulationSteps;
	}

	//only for knematic bodies, ignore now....
	void saveKinematicState(float step) {
		NULL;
	}

	///apply gravity, call this once per timestep
	void	applyGravity()
	{
		for (auto body : _rigids)
		{
			if (body->isActive())
			{
				body->applyGravity();
			}
		}
	}

	void	clearForces()
	{
		for (auto body : _rigids)
		{
			body->clearForces();
		}
	}

	//no motion states now, just ignore...
	void	synchronizeMotionStates()
	{
#if 0
		//iterate over all active rigid bodies
		for (auto body : _rigids)
		{
			if (body->isActive())
			{
				synchronizeSingleMotionState(body);
			}
		}
#endif
	}

#if 0
	void	synchronizeSingleMotionState(crigid* body)
	{
		transf interpolatedTransform;

		btTransformUtil::integrateTransform(body->getInterpolationWorldTransform(),
			body->getInterpolationLinearVelocity(), body->getInterpolationAngularVelocity(), m_localTime * body->getHitFraction(), interpolatedTransform);
		body->getMotionState()->setWorldTransform(interpolatedTransform);
	}
#endif

	void	internalSingleStepSimulation(float timeStep)
	{
		{
			myTimer tr("#predictUnconstraintMotion");
			///apply gravity, predict motion
			predictUnconstraintMotion(timeStep);
		}

		//all contact related...
		{
			myTimer tr("#Cotact related");
			{
				myTimer tr("\t#createPredictiveContacts");
				createPredictiveContacts(timeStep);
			}
			///perform collision detection
			{
				myTimer tr("\t#performDiscreteCollisionDetection");
				performDiscreteCollisionDetection();
			}
			{
				myTimer tr("\t#solveConstraints");
				calculateSimulationIslands();
				///solve contact and other joint constraints
				solveConstraints(timeStep);
			}
		}

		{
			myTimer tr("#Itegration related");

			///integrate transforms
			integrateTransforms(timeStep);

			///update vehicle simulation
			updateActions(timeStep);

			updateActivationState(timeStep);

			if (verb)
				outputRigids();
		}

	}

	void outputRigids()
	{
		for (auto body : _rigids)
		{
			body->output(stepIdx++);
			//if (stepIdx >= 33)
			//	printf("here!");
		}
	}

	///solve contact and other joint constraints
	void solveConstraints(float timeStep)
	{
		/// solve all the constraints for this island
		//m_islandManager->buildAndProcessIslands(getCollisionWorld()->getDispatcher(), getCollisionWorld(), m_solverIslandCallback);
		//buildAndProcessIslands();

		//m_solverIslandCallback->processConstraints();
		processConstraints(timeStep);
	}

	void buildAndProcessIslands()
	{
		//only 1 island only, process all the manifolds
		processIsland();
	}

	//not really called solveGroup, so leave empty here!
	void	processIsland()
	{
		///we don't split islands, so all constraints/contact manifolds/bodies are passed into the solver regardless the island id
		//m_solver->solveGroup(bodies, numBodies, manifolds, numManifolds, &m_sortedConstraints[0], m_numConstraints, *m_solverInfo, m_debugDrawer, m_stackAlloc, m_dispatcher);
	}

	void processConstraints(float timeStep)
	{
		//m_solver->solveGroup(bodies, m_bodies.size(), manifold, m_manifolds.size(), constraints, m_constraints.size(), *m_solverInfo, m_debugDrawer, m_stackAlloc, m_dispatcher);
		std::vector<csolverBody> bodyPool;
		std::vector<csolverConstraint> contactConstraintPool, frictionConstraintPool;

		solveGroupSetup(bodyPool, contactConstraintPool, frictionConstraintPool, timeStep);
		solveGroupIterations(bodyPool, contactConstraintPool, frictionConstraintPool, timeStep);
		solveGroupFinish(bodyPool, contactConstraintPool, frictionConstraintPool, timeStep);
	}

	void	initSolverBody(csolverBody * solverBody, crigid * rb)
	{
		solverBody->internalGetDeltaLinearVelocity() = vec3f();
		solverBody->internalGetDeltaAngularVelocity() = vec3f();
		solverBody->internalGetPushVelocity() = vec3f();
		solverBody->internalGetTurnVelocity() = vec3f();

		if (rb)
		{
			solverBody->m_worldTransform = rb->getWorldTransform();
			solverBody->internalSetInvMass(vec3f(rb->getInvMass(), rb->getInvMass(), rb->getInvMass()));
			solverBody->m_originalBody = rb;
			solverBody->m_linearVelocity = rb->getLinearVelocity();
			solverBody->m_angularVelocity = rb->getAngularVelocity();
		}
		else
		{
			solverBody->m_worldTransform.setIdentity();
			solverBody->internalSetInvMass(vec3f(0, 0, 0));
			solverBody->m_originalBody = 0;
			solverBody->m_linearVelocity = vec3f();
			solverBody->m_angularVelocity = vec3f();
		}
	}

	void solveGroupSetup(std::vector<csolverBody> &bodyPool,
		std::vector<csolverConstraint> &contactConstraintPool,
		std::vector<csolverConstraint>& frictionConstraintPool, 
		float timeStep)
	{
		int numBodies = _rigids.size();
		int numManifolds = _predictiveManifolds.size();
		if (verb)
			printf("%d: %d, %d\n", sidx++, numBodies, numManifolds);

		bodyPool.resize(numBodies + 1);
		csolverBody& fixedBody = bodyPool[0];
		initSolverBody(&fixedBody, 0);
		for (int i = 0; i < numBodies; i++) {
			csolverBody& solverBody = bodyPool[i+1];
			crigid* body = _rigids[i];

			body->setCompanionId(i + 1);
			initSolverBody(&solverBody, body);
			if (_rigids[i]->getInvMass()) {
				vec3f gyroForce(0, 0, 0);
				solverBody.m_linearVelocity += body->getTotalForce() * body->getInvMass() * timeStep;
				solverBody.m_angularVelocity += (body->getTotalTorque() - gyroForce) * body->getInvInertiaTensorWorld() * timeStep;
			}
		}

		for (int i = 0; i < numManifolds; i++) {
			manifold* mf = _predictiveManifolds[i];
			const crigid* rig = mf->getBody1();
			const crigid* rig0 = mf->getBody0();

			int solverBodyIdA = (rig0 == nullptr) ? 0 : rig0->getCompanionId();
			int solverBodyIdB = rig->getCompanionId();
			csolverBody* solverBodyA = &bodyPool[solverBodyIdA];
			csolverBody* solverBodyB = &bodyPool[solverBodyIdB];
			
			int rollingFriction = 1;
			for (int j = 0; j < mf->getNumContacts(); j++) {
				manifoldPoint& cp = mf->getContactPoint(j);

				if (cp.getDistance() <= mf->getContactProcessingThreshold()) {
					addContactConstraints(mf, cp, contactConstraintPool, frictionConstraintPool, timeStep,
						solverBodyIdA, solverBodyIdB, solverBodyA, solverBodyB, bodyPool, rig0, rig, rollingFriction);
				}
			}
		}
	}
	
	void addContactConstraints(manifold* mf, manifoldPoint& cp,
		std::vector<csolverConstraint>& contactConstraintPool,
		std::vector<csolverConstraint>& frictionConstraintPool, 
		float timeStep,
		int solverBodyIdA, int solverBodyIdB,
		csolverBody *solverBodyA, csolverBody *solverBodyB,
		std::vector<csolverBody>& bodyPool,
		const crigid *bodyA, const crigid *bodyB, int &rollingFriction)
	{
		vec3f rel_pos1;
		vec3f rel_pos2;
		float relaxation;
		float rel_vel;
		vec3f vel;

		int frictionIndex = contactConstraintPool.size();
		csolverConstraint contactConstraint;

		//			btRigidBody* rb0 = btRigidBody::upcast(colObj0);
		//			btRigidBody* rb1 = btRigidBody::upcast(colObj1);
		contactConstraint.m_solverBodyIdA = solverBodyIdA;
		contactConstraint.m_solverBodyIdB = solverBodyIdB;
		contactConstraint.m_originalContactPoint = &cp;

		setupContactConstraint(contactConstraint, solverBodyIdA, solverBodyIdB, cp, //infoGlobal,
			vel, rel_vel, relaxation, rel_pos1, rel_pos2, bodyPool, timeStep);

		//			const vec3f& pos1 = cp.getPositionWorldOnA();
		//			const vec3f& pos2 = cp.getPositionWorldOnB();

					/////setup the friction constraints

		contactConstraint.m_frictionIndex = frictionConstraintPool.size();

		vec3f angVelA, angVelB;
		solverBodyA->getAngularVelocity(angVelA);
		solverBodyB->getAngularVelocity(angVelB);
		vec3f relAngVel = angVelB - angVelA;

		if ((cp.m_combinedRollingFriction > 0.f) && (rollingFriction > 0))
		{
			//only a single rollingFriction per manifold
			rollingFriction--;
			assert(0);
#if 0 //no need now...
			if (relAngVel.length() > infoGlobal.m_singleAxisRollingFrictionThreshold)
			{
				relAngVel.normalize();
				applyAnisotropicFriction(colObj0, relAngVel, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				applyAnisotropicFriction(colObj1, relAngVel, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				if (relAngVel.length() > 0.001)
					addRollingFrictionConstraint(relAngVel, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);

			}
			else
			{
				addRollingFrictionConstraint(cp.m_normalWorldOnB, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
				vec3f axis0, axis1;
				btPlaneSpace1(cp.m_normalWorldOnB, axis0, axis1);
				applyAnisotropicFriction(colObj0, axis0, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				applyAnisotropicFriction(colObj1, axis0, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				applyAnisotropicFriction(colObj0, axis1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				applyAnisotropicFriction(colObj1, axis1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
				if (axis0.length() > 0.001)
					addRollingFrictionConstraint(axis0, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
				if (axis1.length() > 0.001)
					addRollingFrictionConstraint(axis1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);

			}
#endif
		}

		int solverMode = 260;
#if 1
		///Bullet has several options to set the friction directions
		///By default, each contact has only a single friction direction that is recomputed automatically very frame 
		///based on the relative linear velocity.
		///If the relative velocity it zero, it will automatically compute a friction direction.

		///You can also enable two friction directions, using the SOLVER_USE_2_FRICTION_DIRECTIONS.
		///In that case, the second friction direction will be orthogonal to both contact normal and first friction direction.
		///
		///If you choose SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION, then the friction will be independent from the relative projected velocity.
		///
		///The user can manually override the friction directions for certain contacts using a contact callback, 
		///and set the cp.m_lateralFrictionInitialized to true
		///In that case, you can set the target relative motion in each friction direction (cp.m_contactMotion1 and cp.m_contactMotion2)
		///this will give a conveyor belt effect
		///
		//if (!(infoGlobal.m_solverMode & SOLVER_ENABLE_FRICTION_DIRECTION_CACHING) || !cp.m_lateralFrictionInitialized)
		if (!cp.m_lateralFrictionInitialized)
		{
			cp.m_lateralFrictionDir1 = vel - cp.m_normalWorldOnB * rel_vel;
			float lat_rel_vel = cp.m_lateralFrictionDir1.length2();
			if (!(solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION) && lat_rel_vel > FLT_EPSILON)
			{
				cp.m_lateralFrictionDir1 *= 1.f / sqrtf(lat_rel_vel);
				if ((solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
				{
					cp.m_lateralFrictionDir2 = cp.m_lateralFrictionDir1.cross(cp.m_normalWorldOnB);
					cp.m_lateralFrictionDir2.normalize();//??
					applyAnisotropicFriction(bodyA, cp.m_lateralFrictionDir2, CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(bodyB, cp.m_lateralFrictionDir2, CF_ANISOTROPIC_FRICTION);
					addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, 
						frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, 0, 0, frictionConstraintPool, bodyPool);

				}

				applyAnisotropicFriction(bodyA, cp.m_lateralFrictionDir1, CF_ANISOTROPIC_FRICTION);
				applyAnisotropicFriction(bodyB, cp.m_lateralFrictionDir1, CF_ANISOTROPIC_FRICTION);
				addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB,
					frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, 0, 0, frictionConstraintPool, bodyPool);

			}
			else
			{
				PlaneSpace1(cp.m_normalWorldOnB, cp.m_lateralFrictionDir1, cp.m_lateralFrictionDir2);

				if ((solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
				{
					applyAnisotropicFriction(bodyA, cp.m_lateralFrictionDir2, CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(bodyB, cp.m_lateralFrictionDir2, CF_ANISOTROPIC_FRICTION);
					addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB,
						frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, 0, 0, frictionConstraintPool, bodyPool);
				}

				applyAnisotropicFriction(bodyA, cp.m_lateralFrictionDir1, CF_ANISOTROPIC_FRICTION);
				applyAnisotropicFriction(bodyB, cp.m_lateralFrictionDir1, CF_ANISOTROPIC_FRICTION);
				addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB,
					frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, 0, 0, frictionConstraintPool, bodyPool);

				if ((solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS) && (solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))
				{
					cp.m_lateralFrictionInitialized = true;
				}
			}

		}
		else
		{
#if 0 //ignored now...
			addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB,
				frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, cp.m_contactMotion1, cp.m_contactCFM1, frictionConstraintPool, bodyPool);

			if ((solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
				addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB,
					frictionIndex, cp, rel_pos1, rel_pos2, bodyA, bodyB, relaxation, cp.m_contactMotion2, cp.m_contactCFM2, frictionConstraintPool, bodyPool);

			setFrictionConstraintImpulse(solverConstraint, solverBodyIdA, solverBodyIdB, cp, infoGlobal);
#endif
			assert(0);
		}
#endif

		contactConstraintPool.push_back(contactConstraint);

	}


	static void	applyAnisotropicFriction(const crigid * colObj, vec3f & frictionDirection, int frictionMode)
	{
#if 0
		if (colObj && colObj->hasAnisotropicFriction(frictionMode))
		{
			// transform to local coordinates
			vec3f loc_lateral = frictionDirection * colObj->getWorldTransform().getBasis();
			const vec3f& friction_scaling = colObj->getAnisotropicFriction();
			//apply anisotropic friction
			loc_lateral *= friction_scaling;
			// ... and transform it back to global coordinates
			frictionDirection = colObj->getWorldTransform().getBasis() * loc_lateral;
		}
#endif
	}


	void addFrictionConstraint(const vec3f& normalAxis, int solverBodyIdA, int solverBodyIdB, int frictionIndex,
		manifoldPoint& cp, const vec3f& rel_pos1, const vec3f& rel_pos2, const crigid * colObj0, const crigid * colObj1,
		float relaxation, float desiredVelocity, float cfmSlip, 
		std::vector<csolverConstraint>& frictionConstraintPool,
		std::vector<csolverBody>& bodyPool)
	{
		csolverConstraint solverConstraint;
		solverConstraint.m_frictionIndex = frictionIndex;
		setupFrictionConstraint(solverConstraint, normalAxis, solverBodyIdA, solverBodyIdB, cp, rel_pos1, rel_pos2,
			colObj0, colObj1, relaxation, desiredVelocity, cfmSlip, bodyPool);
		frictionConstraintPool.push_back(solverConstraint);
	}


	void setupFrictionConstraint(csolverConstraint& solverConstraint, const vec3f& normalAxis, int  solverBodyIdA, int solverBodyIdB,
		manifoldPoint& cp, const vec3f& rel_pos1, const vec3f& rel_pos2, const crigid * colObj0, const crigid* colObj1,
		float relaxation, float desiredVelocity, float cfmSlip, std::vector<csolverBody>& bodyPool)
	{
		solverConstraint.m_contactNormal = normalAxis;
		csolverBody& solverBodyA = bodyPool[solverBodyIdA];
		csolverBody& solverBodyB = bodyPool[solverBodyIdB];

		crigid* body0 = bodyPool[solverBodyIdA].m_originalBody;
		crigid* body1 = bodyPool[solverBodyIdB].m_originalBody;

		solverConstraint.m_solverBodyIdA = solverBodyIdA;
		solverConstraint.m_solverBodyIdB = solverBodyIdB;

		solverConstraint.m_friction = cp.m_combinedFriction;
		solverConstraint.m_originalContactPoint = 0;

		solverConstraint.m_appliedImpulse = 0.f;
		solverConstraint.m_appliedPushImpulse = 0.f;

		{
			vec3f ftorqueAxis1 = rel_pos1.cross(solverConstraint.m_contactNormal);
			solverConstraint.m_relpos1CrossNormal = ftorqueAxis1;
			solverConstraint.m_angularComponentA = body0 ? body0->getInvInertiaTensorWorld() * ftorqueAxis1: vec3f(0, 0, 0);
		}
		{
			vec3f ftorqueAxis1 = rel_pos2.cross(-solverConstraint.m_contactNormal);
			solverConstraint.m_relpos2CrossNormal = ftorqueAxis1;
			solverConstraint.m_angularComponentB = body1 ? body1->getInvInertiaTensorWorld() * ftorqueAxis1 : vec3f(0, 0, 0);
		}

		{
			vec3f vec;
			float denom0 = 0.f;
			float denom1 = 0.f;
			if (body0)
			{
				vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
				denom0 = body0->getInvMass() + normalAxis.dot(vec);
			}
			if (body1)
			{
				vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
				denom1 = body1->getInvMass() + normalAxis.dot(vec);
			}
			float denom = relaxation / (denom0 + denom1);
			solverConstraint.m_jacDiagABInv = denom;
		}

		{


			float rel_vel;
			float vel1Dotn = solverConstraint.m_contactNormal.dot(body0 ? solverBodyA.m_linearVelocity : vec3f(0, 0, 0))
				+ solverConstraint.m_relpos1CrossNormal.dot(body0 ? solverBodyA.m_angularVelocity : vec3f(0, 0, 0));
			float vel2Dotn = -solverConstraint.m_contactNormal.dot(body1 ? solverBodyB.m_linearVelocity : vec3f(0, 0, 0))
				+ solverConstraint.m_relpos2CrossNormal.dot(body1 ? solverBodyB.m_angularVelocity : vec3f(0, 0, 0));

			rel_vel = vel1Dotn + vel2Dotn;

			//		float positionalError = 0.f;

			float velocityError = desiredVelocity - rel_vel;
			float	velocityImpulse = velocityError * float(solverConstraint.m_jacDiagABInv);
			solverConstraint.m_rhs = velocityImpulse;
			solverConstraint.m_cfm = cfmSlip;
			solverConstraint.m_lowerLimit = 0;
			solverConstraint.m_upperLimit = 1e10f;

		}
	}

	void setupContactConstraint(csolverConstraint& solverConstraint,
		int solverBodyIdA, int solverBodyIdB,
		manifoldPoint& cp, //const btContactSolverInfo& infoGlobal,
		vec3f& vel, float& rel_vel, float& relaxation,
		vec3f& rel_pos1, vec3f& rel_pos2,
		std::vector<csolverBody>& bodyPool, float timeStep)
	{

		const vec3f& pos1 = cp.getPositionWorldOnA();
		const vec3f& pos2 = cp.getPositionWorldOnB();

		csolverBody* bodyA = &bodyPool[solverBodyIdA];
		csolverBody* bodyB = &bodyPool[solverBodyIdB];

		crigid* rb0 = bodyA->m_originalBody;
		crigid* rb1 = bodyB->m_originalBody;

		//			vec3f rel_pos1 = pos1 - colObj0->getWorldTransform().getOrigin(); 
		//			vec3f rel_pos2 = pos2 - colObj1->getWorldTransform().getOrigin();
		rel_pos1 = pos1 - bodyA->getWorldTransform().getOrigin();
		rel_pos2 = pos2 - bodyB->getWorldTransform().getOrigin();

		relaxation = 1.f;

		vec3f torqueAxis0 = rel_pos1.cross(cp.m_normalWorldOnB);
		solverConstraint.m_angularComponentA = rb0 ? rb0->getInvInertiaTensorWorld() * torqueAxis0 : vec3f(0, 0, 0);
		vec3f torqueAxis1 = rel_pos2.cross(cp.m_normalWorldOnB);
		solverConstraint.m_angularComponentB = rb1 ? rb1->getInvInertiaTensorWorld() * -torqueAxis1 : vec3f(0, 0, 0);

		{
#ifdef COMPUTE_IMPULSE_DENOM
			float denom0 = rb0->computeImpulseDenominator(pos1, cp.m_normalWorldOnB);
			float denom1 = rb1->computeImpulseDenominator(pos2, cp.m_normalWorldOnB);
#else							
			vec3f vec;
			float denom0 = 0.f;
			float denom1 = 0.f;
			if (rb0)
			{
				vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
				denom0 = rb0->getInvMass() + cp.m_normalWorldOnB.dot(vec);
			}
			if (rb1)
			{
				vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
				denom1 = rb1->getInvMass() + cp.m_normalWorldOnB.dot(vec);
			}
#endif //COMPUTE_IMPULSE_DENOM		

			float denom = relaxation / (denom0 + denom1);
			solverConstraint.m_jacDiagABInv = denom;
		}

		solverConstraint.m_contactNormal = cp.m_normalWorldOnB;
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
		solverConstraint.m_relpos2CrossNormal = -torqueAxis1;

		float restitution = 0.f;
		float penetration = cp.getDistance() + 0.0f; // infoGlobal.m_linearSlop;

		{
			vec3f vel1, vel2;

			vel1 = rb0 ? rb0->getVelocityInLocalPoint(rel_pos1) : vec3f(0, 0, 0);
			vel2 = rb1 ? rb1->getVelocityInLocalPoint(rel_pos2) : vec3f(0, 0, 0);

			//			vec3f vel2 = rb1 ? rb1->getVelocityInLocalPoint(rel_pos2) : vec3f(0,0,0);
			vel = vel1 - vel2;
			rel_vel = cp.m_normalWorldOnB.dot(vel);



			solverConstraint.m_friction = cp.m_combinedFriction;


			restitution = restitutionCurve(rel_vel, cp.m_combinedRestitution);
			if (restitution <= float(0.))
			{
				restitution = 0.f;
			};
		}


		///warm starting (or zero if disabled)
		if (true) //infoGlobal.m_solverMode & SOLVER_USE_WARMSTARTING)
		{
			float warmstartingFactor = 0.85f;
			solverConstraint.m_appliedImpulse = cp.m_appliedImpulse * warmstartingFactor;
			if (rb0)
				bodyA->internalApplyImpulse(solverConstraint.m_contactNormal * bodyA->internalGetInvMass(), solverConstraint.m_angularComponentA, solverConstraint.m_appliedImpulse);
			if (rb1)
				bodyB->internalApplyImpulse(solverConstraint.m_contactNormal * bodyB->internalGetInvMass(), -solverConstraint.m_angularComponentB, -(float)solverConstraint.m_appliedImpulse);
		}
		else
		{
			solverConstraint.m_appliedImpulse = 0.f;
		}

		solverConstraint.m_appliedPushImpulse = 0.f;

		{
			float vel1Dotn = solverConstraint.m_contactNormal.dot(rb0 ? bodyA->m_linearVelocity : vec3f(0, 0, 0))
				+ solverConstraint.m_relpos1CrossNormal.dot(rb0 ? bodyA->m_angularVelocity : vec3f(0, 0, 0));
			float vel2Dotn = -solverConstraint.m_contactNormal.dot(rb1 ? bodyB->m_linearVelocity : vec3f(0, 0, 0))
				+ solverConstraint.m_relpos2CrossNormal.dot(rb1 ? bodyB->m_angularVelocity : vec3f(0, 0, 0));
			float rel_vel = vel1Dotn + vel2Dotn;

			float positionalError = 0.f;
			float	velocityError = restitution - rel_vel;// * damping;


			float erp = 0.8f; //infoGlobal.m_erp2;

			float splitImpulsePenetrationThreshold = -0.04;
			//if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
			if (penetration > splitImpulsePenetrationThreshold)
			{
				erp = 0.2f; // infoGlobal.m_erp;
			}

			if (penetration > 0)
			{
				positionalError = 0;

				velocityError -= penetration / timeStep;
			}
			else
			{
				positionalError = -penetration * erp / timeStep;
			}

			float  penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
			float velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;

			//if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
			if (penetration > splitImpulsePenetrationThreshold)
			{
				//combine position and velocity into rhs
				solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
				solverConstraint.m_rhsPenetration = 0.f;

			}
			else
			{
				//split position and velocity into rhs and m_rhsPenetration
				solverConstraint.m_rhs = velocityImpulse;
				solverConstraint.m_rhsPenetration = penetrationImpulse;
			}
			solverConstraint.m_cfm = 0.f;
			solverConstraint.m_lowerLimit = 0;
			solverConstraint.m_upperLimit = 1e10f;
		}
	}

	void solveGroupIterations(std::vector<csolverBody>& bodyPool,
		std::vector<csolverConstraint>& contactConstraintPool,
		std::vector<csolverConstraint>& frictionConstraintPool, 
		float timeStep)
	{
		///this is a special step to resolve penetrations (just for contacts)
		solveGroupCacheFriendlySplitImpulseIterations(bodyPool, contactConstraintPool);

		//int maxIterations = m_maxOverrideNumSolverIterations > infoGlobal.m_numIterations ? m_maxOverrideNumSolverIterations : infoGlobal.m_numIterations;
		int maxIterations = 10;

		for (int iteration = 0; iteration < maxIterations; iteration++)
			//for ( int iteration = maxIterations-1  ; iteration >= 0;iteration--)
		{
			solveSingleIteration(bodyPool, contactConstraintPool, frictionConstraintPool, timeStep, iteration);
		}
	}

	void solveGroupCacheFriendlySplitImpulseIterations(
		std::vector<csolverBody>& bodyPool,
		std::vector<csolverConstraint>& contactConstraintPool)
	{
		int iteration;
		int numIterations = 10; //infoGlobal.m_numIterations
		for (iteration = 0; iteration < numIterations; iteration++)
		{
			{
				int numPoolConstraints = contactConstraintPool.size();
				int j;
				for (j = 0; j < numPoolConstraints; j++)
				{
					csolverConstraint& solveManifold = contactConstraintPool[j]; // m_orderTmpConstraintPool[j]];
					resolveSplitPenetrationImpulseCacheFriendly(
						bodyPool[solveManifold.m_solverBodyIdA], bodyPool[solveManifold.m_solverBodyIdB], solveManifold);
				}
			}
		}
	}

	void	resolveSplitPenetrationImpulseCacheFriendly(
		csolverBody& body1,
		csolverBody& body2,
		csolverConstraint& c)
	{
		if (c.m_rhsPenetration)
		{
			float deltaImpulse = c.m_rhsPenetration - float(c.m_appliedPushImpulse) * c.m_cfm;
			const float deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetPushVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetTurnVelocity());
			const float deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetPushVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetTurnVelocity());

			deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
			deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
			const float sum = float(c.m_appliedPushImpulse) + deltaImpulse;
			if (sum < c.m_lowerLimit)
			{
				deltaImpulse = c.m_lowerLimit - c.m_appliedPushImpulse;
				c.m_appliedPushImpulse = c.m_lowerLimit;
			}
			else
			{
				c.m_appliedPushImpulse = sum;
			}
			body1.internalApplyPushImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
			body2.internalApplyPushImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
		}
	}

	void solveSingleIteration(std::vector<csolverBody>& bodyPool,
		std::vector<csolverConstraint>& contactConstraintPool,
		std::vector<csolverConstraint>& frictionConstraintPool,
		float timeStep, int iteration)
	{
		int numIterations = 10;
		if (iteration < numIterations)
		{
			///solve all contact constraints
			int numPoolConstraints = contactConstraintPool.size();
			for (int j = 0; j < numPoolConstraints; j++)
			{
				csolverConstraint& solveManifold = contactConstraintPool[j]; // m_orderTmpConstraintPool[j]];
				resolveSingleConstraintRowLowerLimit(bodyPool[solveManifold.m_solverBodyIdA], bodyPool[solveManifold.m_solverBodyIdB], solveManifold);
			}
			///solve all friction constraints
			int numFrictionPoolConstraints = frictionConstraintPool.size();
			for (int j = 0; j < numFrictionPoolConstraints; j++)
			{
				csolverConstraint& solveManifold = frictionConstraintPool[j]; // m_orderFrictionConstraintPool[j]];
				float totalImpulse = contactConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;

				if (totalImpulse > float(0))
				{
					solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
					solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

					resolveSingleConstraintRowGeneric(bodyPool[solveManifold.m_solverBodyIdA], bodyPool[solveManifold.m_solverBodyIdB], solveManifold);
				}
			}

#if 0
			int numRollingFrictionPoolConstraints = m_tmpSolverContactRollingFrictionConstraintPool.size();
			for (int j = 0; j < numRollingFrictionPoolConstraints; j++)
			{
				csolverConstraint& rollingFrictionConstraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
				float totalImpulse = contactConstraintPool[rollingFrictionConstraint.m_frictionIndex].m_appliedImpulse;
				if (totalImpulse > float(0))
				{
					float rollingFrictionMagnitude = rollingFrictionConstraint.m_friction * totalImpulse;
					if (rollingFrictionMagnitude > rollingFrictionConstraint.m_friction)
						rollingFrictionMagnitude = rollingFrictionConstraint.m_friction;

					rollingFrictionConstraint.m_lowerLimit = -rollingFrictionMagnitude;
					rollingFrictionConstraint.m_upperLimit = rollingFrictionMagnitude;

					resolveSingleConstraintRowGeneric(bodyPool[rollingFrictionConstraint.m_solverBodyIdA], bodyPool[rollingFrictionConstraint.m_solverBodyIdB], rollingFrictionConstraint);
				}
			}
#endif
		}
	}


	// Project Gauss Seidel or the equivalent Sequential Impulse
	void resolveSingleConstraintRowGeneric(csolverBody& body1, csolverBody& body2, csolverConstraint& c)
	{
		float deltaImpulse = c.m_rhs - float(c.m_appliedImpulse) * c.m_cfm;
		const float deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetDeltaAngularVelocity());
		const float deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetDeltaAngularVelocity());

		//	const float delta_rel_vel	=	deltaVel1Dotn-deltaVel2Dotn;
		deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
		deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;

		const float sum = float(c.m_appliedImpulse) + deltaImpulse;
		if (sum < c.m_lowerLimit)
		{
			deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
			c.m_appliedImpulse = c.m_lowerLimit;
		}
		else if (sum > c.m_upperLimit)
		{
			deltaImpulse = c.m_upperLimit - c.m_appliedImpulse;
			c.m_appliedImpulse = c.m_upperLimit;
		}
		else
		{
			c.m_appliedImpulse = sum;
		}

		body1.internalApplyImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
		body2.internalApplyImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
	}


		// Project Gauss Seidel or the equivalent Sequential Impulse
	void resolveSingleConstraintRowLowerLimit(csolverBody& body1, csolverBody& body2, csolverConstraint& c)
	{
		float deltaImpulse = c.m_rhs - float(c.m_appliedImpulse) * c.m_cfm;
		const float deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetDeltaAngularVelocity());
		const float deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetDeltaAngularVelocity());

		deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
		deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
		const float sum = float(c.m_appliedImpulse) + deltaImpulse;
		if (sum < c.m_lowerLimit)
		{
			deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
			c.m_appliedImpulse = c.m_lowerLimit;
		}
		else
		{
			c.m_appliedImpulse = sum;
		}
		body1.internalApplyImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);

		body2.internalApplyImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
	}

	void solveGroupFinish(std::vector<csolverBody>& bodyPool, 
		std::vector<csolverConstraint> &contactConstraintPool,
		std::vector<csolverConstraint>& frictionConstraintPool, 
		float timeStep)
	{

		int numPoolConstraints = contactConstraintPool.size();
		int i, j;
		int solverMode = 260;

		if (solverMode & SOLVER_USE_WARMSTARTING)
		{
			for (j = 0; j < numPoolConstraints; j++)
			{
				const csolverConstraint& solveManifold = contactConstraintPool[j];
				manifoldPoint* pt = (manifoldPoint*)solveManifold.m_originalContactPoint;
				assert(pt);
				pt->m_appliedImpulse = solveManifold.m_appliedImpulse;
				//	float f = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
					//	printf("pt->m_appliedImpulseLateral1 = %f\n", f);
				pt->m_appliedImpulseLateral1 = frictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
				//printf("pt->m_appliedImpulseLateral1 = %f\n", pt->m_appliedImpulseLateral1);
				if ((solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
				{
					pt->m_appliedImpulseLateral2 = frictionConstraintPool[solveManifold.m_frictionIndex + 1].m_appliedImpulse;
				}
				//do a callback here?
			}
		}

		//infoGlobal.m_splitImpulseTurnErp = 0.1f;

		for (int i = 0; i < bodyPool.size(); i++)
		{
			crigid * body = bodyPool[i].m_originalBody;
			if (body)
			{
				bodyPool[i].writebackVelocityAndTransform(timeStep, 0.1f);
				bodyPool[i].m_originalBody->setLinearVelocity(bodyPool[i].m_linearVelocity);
				bodyPool[i].m_originalBody->setAngularVelocity(bodyPool[i].m_angularVelocity);
				bodyPool[i].m_originalBody->setWorldTransform(bodyPool[i].m_worldTransform);
			}
		}
	}

	//don't know what is really going on here.
	//My guess is split contacts into different (connected) groups? 
	//So I just ignore it now.
	void calculateSimulationIslands()
	{
		NULL;
	}

	//DCD with current rigid body, not the predicted ...
	void	performDiscreteCollisionDetection()
	{
		{
		myTimer tr("\t\t#updateAabbs");
		updateAabbs();
		}

		// TODO(wangwei) use mesh collision
		{
			myTimer tr("\t\t#computeOverlappingPairs");
		computeOverlappingPairs();
		}
		{
			myTimer tr("\t\t#processAllOverlappingPairs");
			processAllOverlappingPairs();
		}
	}

	void	updateAabbs()
	{
		for (auto body : _rigids)
		{
			body->updateContactBx(gContactBreakingThreshold);
		}
	}

	//a naive broadphase culling, should be imporved later
	void computeOverlappingPairs()
	{
		float threshold = 0.25;
		int culled = 0;
		_rigid_pairs.clear();
		_rigid_pln_pairs.clear();

		int rnum = _rigids.size();
		for (int i=0; i<rnum; i++)
			for (int j = i + 1; j < rnum; j++) {
				if (_rigids[i]->bound().overlaps(_rigids[j]->bound()))
#if 0
					if (getLargestVelocityNorm(_rigids[i], _rigids[j]) < threshold)
						culled++;
					else
#endif
						_rigid_pairs.push_back(id_pair(i, j, true));
			}

#ifdef PROF
#else
		printf("\t\t\t$$$rigid-rigid-culled = %d\n", culled);
#endif

		int pnum = _plns.size();
		for (int i = 0; i < rnum; i++)
			for (int j = 0; j <pnum; j++) {
				if (_rigids[i]->bound().overlaps(*_plns[j]))
					_rigid_pln_pairs.push_back(id_pair(i, j, false));
			}
	}

	// void btGImpactCollisionAlgorithm::processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
	void processAllOverlappingPairs()
	{
#ifdef GPU
#define OLD_GPU
#ifdef OLD_GPU
		if (false) //_rigid_pairs.size() < 10)
		{
			myTimer tr("\t\t\t#RigidRigidCollision");
			myTimer2 tr1("\t\t\t\t#RigidRigid-CD");
			myTimer2 tr2("\t\t\t\t#RigidRigid-HD");

			int maxL = 0;
			for (auto pair : _rigid_pairs) {
				unsigned int id0, id1;
				pair.get(id0, id1);
				RigidRigidCollision(_rigids[id0], _rigids[id1], tr1, tr2, maxL, maxTriPairs);
			}

#ifdef PROF
#else
			printf("\t\t\t$$$rigid-pair-size = %zd\n", _rigid_pairs.size());
			printf("\t\t\t$$$ret--maxsize = %d\n", maxL);
			printf("\t\t\t$$$maxTriPairs = %d\n", maxTriPairs);
#endif

			tr1.print();
			tr2.print();
		}
		else {
			pushRigidPairsGPU(_rigid_pairs.size(), _rigid_pairs.data());
			RigidRigidCollisionFusedGPU(maxTriPairs);
#ifdef PROF
#else
			printf("\t\t\t$$$rigid-pair-size = %zd\n", _rigid_pairs.size());
			printf("\t\t\t$$$maxTriPairs = %d\n", maxTriPairs);
#endif
		}
#else

		pushRigidPairsGPU(_rigid_pairs.size(), _rigid_pairs.data());
		RigidRigidCollisionGPU(maxTriPairs);
#ifdef PROF
#else
		printf("\t\t\t$$$rigid-pair-size = %zd\n", _rigid_pairs.size());
		printf("\t\t\t$$$maxTriPairs = %d\n", maxTriPairs);
#endif

#endif

#else
		{
			myTimer tr("\t\t\t#RigidRigidCollision");
			myTimer2 tr1("\t\t\t\t#RigidRigid-CD");
			myTimer2 tr2("\t\t\t\t#RigidRigid-HD");

			int maxL = 0;
			int maxTriPairs = 0;

			int pairNum = _rigid_pairs.size();

#pragma omp parallel for
			for (int i = 0; i < pairNum; i++) {
				id_pair& pair = _rigid_pairs[i];
				unsigned int id0, id1;
				pair.get(id0, id1);
				RigidRigidCollision(_rigids[id0], _rigids[id1], tr1, tr2, maxL, maxTriPairs);
			}

#ifdef PROF
#else
			printf("\t\t\t$$$rigid-pair-size = %zd\n", _rigid_pairs.size());
			//printf("\t\t\t$$$ret--maxsize = %d\n", maxL);
			//printf("\t\t\t$$$maxTriPairs = %d\n", maxTriPairs);
			printf("\t\t\t$$$triangle-pair-size = %d\n", maxTriPairs);

#endif

			tr1.print();
			tr2.print();
		}
#endif

		{
			myTimer tr("\t\t\t#RigidPlaneCollision");
			int pairNum = _rigid_pln_pairs.size();

#pragma omp parallel for
			for (int i = 0; i < pairNum; i++) {
				id_pair &pair = _rigid_pln_pairs[i];
				unsigned int id0, id1;
				pair.get(id0, id1);
				RigidPlaneCollision(_rigids[id0], _plns[id1]);
			}
		}
	}

#ifdef GPU
	void RigidRigidCollisionGPU(int& maxP)
	{
		//std::vector<id_pair> pairset;
		//std::vector<id_pair> rigset;
		if (sPairSet == nullptr) {
			sPairSet = new id_pair[MAX_PAIR_NUM];
			sRigSet = new id_pair[MAX_PAIR_NUM];
		}

		int pairNum;
		{
			myTimer tr1("\t\t\t\t#RigidRigid-CD");
			//should be optimized as static allocated memory...
			float margin = 0.01;
			gpuTimerBegin();

			pairNum = checkRigidRigidCDGPU(margin);
			if (pairNum) {
				getPairsGPU(sPairSet, pairNum, sRigSet);
			}
			gpuTimerEnd("checkTriTriCDGPU", true);

		}

		if (pairNum == 0)
			return;

#if  1
#ifdef PROF
#else
		printf("\t\t\t$$$triangle-pair-size = %zd\n", pairNum);
#endif

		if (pairNum > maxP)
			maxP = pairNum;
#endif

		{
			myTimer tr("\t\t\t\t#RigidRigid-HD");

#pragma omp parallel for
			for (int i = 0; i < pairNum; i++)
			{
				id_pair& p = sPairSet[i];
				id_pair& r = sRigSet[i];
				unsigned int id0, id1, r0, r1;
				p.get(id0, id1);
				r.get(r0, r1);
				collide_sat_triangles(_rigids[r0], _rigids[r1], id0, id1);
			}
		}
	}

	void RigidRigidCollisionFusedGPU(int &maxP)
	{
		//std::vector<id_pair> pairset;
		//std::vector<id_pair> rigset;
		if (sPairSet == nullptr) {
			sPairSet = new id_pair[MAX_PAIR_NUM];
			sRigSet = new id_pair[MAX_PAIR_NUM];
		}

		int pairNum;
		{
			myTimer tr1("\t\t\t\t#RigidRigid-CD");
			//should be optimized as static allocated memory...
			float margin = 0.01;
			kmesh* m = _meshs[0];
			vec3f bMin, bMax, quan;
			m->_qbvh->getInfo(bMin, bMax, quan);

			pairNum = checkTriTriCDFusedGPU(margin,
				bMin.x, bMin.y, bMin.z,
				bMax.x, bMax.y, bMax.z,
				quan.x, quan.y, quan.z);

			if (pairNum) {
				getPairsGPU(sPairSet, pairNum, sRigSet);
			}
		}

		if (pairNum == 0)
			return;

#if  1
#ifdef PROF
#else
		printf("\t\t\t$$$triangle-pair-size = %zd\n", pairNum);
#endif

		if (pairNum > maxP)
			maxP = pairNum;
#endif

		{
			myTimer tr("\t\t\t\t#RigidRigid-HD");

#pragma omp parallel for
			for (int i=0; i< pairNum; i++)
			{
				id_pair& p = sPairSet[i];
				id_pair& r = sRigSet[i];
				unsigned int id0, id1, r0, r1;
				p.get(id0, id1);
				r.get(r0, r1);
				collide_sat_triangles(_rigids[r0], _rigids[r1], id0, id1);
			}
		}
	}

	void RigidRigidCollision(crigid* ra, crigid* rb, myTimer2& tr1, myTimer2& tr2, int& maxL, int& maxP)
	{
		std::vector<id_pair> pairset;
		{
			double t0 = omp_get_wtime();
			const transf& trfA = ra->getTrf();
			const transf& trfB = rb->getTrf();
			const transf trfA2B = trfB.inverse() * trfA;
			float margin = 0.01;

			gpuTimerBegin();

			vec3f bMin, bMax, quan;
			ra->getMesh()->_qbvh->getInfo(bMin, bMax, quan);

			int pairNum = checkTriTriCDGPU(&trfA2B.getOrigin(), &trfA2B.getBasis(), margin,
				bMin.x, bMin.y, bMin.z,
				bMax.x, bMax.y, bMax.z,
				quan.x, quan.y, quan.z);
			
			if (pairNum) {
				pairset.resize(pairNum);
				getPairsGPU(pairset.data(), pairNum, nullptr);
			}
			gpuTimerEnd("checkTriTriCDGPU", true);
		}
		
		if (pairset.size() == 0)
			return;

#if  1
		//printf("\t\t\t$$$triangle-pair-size = %zd\n", pairset.size());
		if (pairset.size() > maxP)
			maxP = pairset.size();
#endif

		{
			double t0 = omp_get_wtime();

			for (auto p : pairset)
			{
				unsigned int id0, id1;
				p.get(id0, id1);
				collide_sat_triangles(ra, rb, id0, id1);
			}
			tr2.inc(omp_get_wtime() - t0);
		}
	}
#else
	void RigidRigidCollision(crigid* ra, crigid* rb, myTimer2 &tr1, myTimer2 &tr2, int &maxL, int &maxP)
	{
#if 0
		btPairSet pairset;
		gimpact_vs_gimpact_find_pairs(orgtrans0, orgtrans1, meshshape0part, meshshape1part, pairset);

		if (pairset.size() != 0) {
			//specialized function
			collide_sat_triangles(body0Wrap, body1Wrap, shapepart0, shapepart1, &pairset[0].m_index1, pairset.size());
		}
#endif

		std::vector<id_pair> pairset;
		{
			double t0 = omp_get_wtime();
			const transf& trfA = ra->getTrf();
			const transf& trfB = rb->getTrf();
			const transf trfA2B = trfB.inverse() * trfA;

			int num = ra->getMesh()->getNbFaces();
			tri3f* tris = ra->getMesh()->getTris();
			vec3f* vtxs = ra->getMesh()->getVtxs();

//#pragma omp parallel for schedule(static)
			for (int i = 0; i < num; i++) {
				BOX bx;
				tri3f& t = tris[i];
				vec3f va = vtxs[t.id0()];
				vec3f vb = vtxs[t.id1()];
				vec3f vc = vtxs[t.id2()];

				bx += trfA2B.getVertex(va);
				bx += trfA2B.getVertex(vb);
				bx += trfA2B.getVertex(vc);

				float margin = 0.01;
				bx.enlarge(margin);

				std::vector<int> rets;
				//rb->getMesh()->query(bx, rets);
				rb->getMesh()->query2(bx, rets);

#if 0
				//if (rets.size()) 
				{
					std::vector<int> rets2;
					rb->getMesh()->query2(bx, rets2);
					if (rets.size() != rets2.size()) {
						printf("error!\n");
						abort();
					}
					//else
					//	printf("passed!\n");
				}
#endif

#if 1
				if (rets.size() > maxL) {
					mylock.lock();
					maxL = rets.size();
					mylock.unlock();
				}
#endif

				for (auto t : rets) {
					mylock.lock();
					pairset.push_back(id_pair(i, t, false));
					mylock.unlock();
				}
			}
			tr1.inc(omp_get_wtime() - t0);
		}

		if (pairset.size() == 0)
			return;

#if  1
		//printf("\t\t\t$$$triangle-pair-size = %zd\n", pairset.size());
		if (pairset.size() > maxP) {
			mylock.lock();
			maxP = pairset.size();
			mylock.unlock();
		}
#endif

		{
			double t0 = omp_get_wtime();

			for (auto p : pairset)
			{
				unsigned int id0, id1;
				p.get(id0, id1);
				collide_sat_triangles(ra, rb, id0, id1);
			}
			tr2.inc(omp_get_wtime() - t0);
		}
	}

	void RigidRigidCollisionFast(crigid* ra, crigid* rb, myTimer2& tr1, myTimer2& tr2, int& maxL, int& maxP)
	{
		std::vector<id_pair> pairset;
		{
			double t0 = omp_get_wtime();
			const transf& trfA = ra->getTrf();
			const transf& trfB = rb->getTrf();
			cbox2boxTrfCache trf2to1;

			trf2to1.calc_from_homogenic(trfA, trfB);
			ra->getMesh()->collide(rb->getMesh(), trf2to1, pairset);
			tr1.inc(omp_get_wtime() - t0);
		}

		if (pairset.size() == 0)
			return;

#if  1
		//printf("\t\t\t$$$triangle-pair-size = %zd\n", pairset.size());
		//if (pairset.size() > maxP)
		{
			mylock.lock();
			maxP += pairset.size();
			mylock.unlock();
		}
#endif

		{
			double t0 = omp_get_wtime();

			for (auto p : pairset)
			{
				unsigned int id0, id1;
				p.get(id0, id1);
				collide_sat_triangles(ra, rb, id0, id1);
			}
			tr2.inc(omp_get_wtime() - t0);
		}
	}
#endif

	void collide_sat_triangles(crigid* ra, crigid* rb, unsigned int ta, unsigned int tb)
	{
		cprimitiveTriangle ptri0;
		cprimitiveTriangle ptri1;
		ctriangleContact contact_data;

		get_primitive_triangle(ra, ta, ptri0);
		get_primitive_triangle(rb, tb, ptri1);
		ptri0.applyTransform(ra->getTrf());
		ptri1.applyTransform(rb->getTrf());
		//build planes
		ptri0.buildTriPlane();
		ptri1.buildTriPlane();
		// test conservative
		if (ptri0.overlap_test_conservative(ptri1))
		{
			if (ptri0.find_triangle_collision_clip_method(ptri1, contact_data))
			{

				int j = contact_data.m_point_count;
				while (j--)
				{
					addContactPoint(ra, nullptr, rb,
						contact_data.m_points[j],
						contact_data.m_separating_normal.xyz(),
						-contact_data.m_penetration_depth);
				}
			}
		}
	}

	void RigidPlaneCollision(crigid* r, cplane* pln)
	{
		float margin = 0.01;

		int num = r->getVertexCount();
		while (num--) {
#if 0
			if (num == 11715)
				printf("here!");
#endif

			vec3f vp = r->getVertex(num);
			float dist = pln->distance(vp) - margin;
			if (dist < 0) //add contact
			{
				addContactPoint(nullptr, pln, r, vp, -pln->n(), dist);
			}
		}
	}

	manifold* getManifold(crigid *r0, cplane* pln, crigid* r)
	{
		for (auto mf : _predictiveManifolds)
		{
			const crigid* mfRig0 = mf->getBody0();
			const cplane *mfPln = mf->getPlane();
			const crigid* mfRig = mf->getBody1();
			if (mfRig0 == r0 && pln == mfPln && r == mfRig)
				return mf;
		}

		float contactBreakingThreshold = 0.128469273;
		float contactProcessingThreshold = GLH_LARGE_FLOAT;
		manifold* mf = new manifold(r0, pln, r, 0, contactBreakingThreshold, contactProcessingThreshold);
		mf->m_index1a = _predictiveManifolds.size();
		mylock.lock();
		_predictiveManifolds.push_back(mf);
		mylock.unlock();
		return mf;
	}

	void addContactPoint(crigid *r0, cplane* pln, crigid* r, vec3f &ptW, vec3f &nrmW, float depth)
	{
		manifold* mf = getManifold(r0, pln, r);
		if (depth > mf->getContactBreakingThreshold())
			return;

		vec3f pointA = ptW + nrmW * depth;
		vec3f localB = r->getWorldTransform().getVertexInv(ptW);
		vec3f localA = pointA;
		if (r0)
			localA = r0->getWorldTransform().getVertexInv(localA);

		manifoldPoint newPt(localA, localB, nrmW, depth);
		newPt.m_positionWorldOnA = pointA;
		newPt.m_positionWorldOnB = ptW;

		int insertIndex = mf->getCacheEntry(newPt);

		newPt.m_combinedFriction = 0.25f; //calculateCombinedFriction(m_body0Wrap->getCollisionObject(), m_body1Wrap->getCollisionObject());
		newPt.m_combinedRestitution = 0.0f; //calculateCombinedRestitution(m_body0Wrap->getCollisionObject(), m_body1Wrap->getCollisionObject());
		newPt.m_combinedRollingFriction = 0.0f; //calculateCombinedRollingFriction(m_body0Wrap->getCollisionObject(), m_body1Wrap->getCollisionObject());
		PlaneSpace1(newPt.m_normalWorldOnB, newPt.m_lateralFrictionDir1, newPt.m_lateralFrictionDir2);



		//BP mod, store contact triangles.
		newPt.m_partId0 = -1;
		newPt.m_partId1 = -1;
		newPt.m_index0 = -1;
		newPt.m_index1 = -1;

		//printf("depth=%f\n",depth);
		///@todo, check this for any side effects
		if (insertIndex >= 0)
		{
			//const btManifoldPoint& oldPoint = m_manifoldPtr->getContactPoint(insertIndex);
			mylock.lock();
			mf->replaceContactPoint(newPt, insertIndex);
			mylock.unlock();
		}
		else
		{
			mylock.lock();
			insertIndex = mf->addManifoldPoint(newPt);
			mylock.unlock();
		}
	}

	//TM: clear Manifolds and deal with CCD(ignored now...)
	void	createPredictiveContacts(float timeStep)
	{
		for (auto mf : _predictiveManifolds)
		{
			delete mf;
		}
		_predictiveManifolds.clear();
	}

	void	integrateTransforms(float timeStep)
	{
		transf predictedTrans;
		for (auto body : _rigids)
		{
			body->predictIntegratedTransform(timeStep, predictedTrans);
			body->proceedToTransform(predictedTrans);
		}
	}


	void	predictUnconstraintMotion(float timeStep)
	{
		for (auto body : _rigids)
		{
			//don't integrate/update velocities here, it happens in the constraint solver

			//damping
			body->applyDamping(timeStep);

			body->predictIntegratedTransform(timeStep, body->getInterpolationWorldTransform());
		}
	}


	//user action related, ignore now...
	void	updateActions(float timeStep)
	{
#if 0
		for (int i = 0; i < m_actions.size(); i++)
		{
			m_actions[i]->updateAction(this, timeStep);
		}
#endif
	}

	void	updateActivationState(float timeStep)
	{
		for (auto body : _rigids)
		{
			body->updateDeactivation(timeStep);

			if (body->wantsSleeping())
			{
						body->setAngularVelocity(vec3f(0, 0, 0));
						body->setLinearVelocity(vec3f(0, 0, 0));
			}
			else
			{
				if (body->getActivationState() != DISABLE_DEACTIVATION)
					body->setActivationState(ACTIVE_TAG);
			}
		}
	}
} g_scene;

vec3f dPt0, dPt1, dPtw;

bool readobjfile(const char *path, 
				 unsigned int &numVtx, unsigned int &numTri, 
				 tri3f *&tris, vec3f *&vtxs, REAL scale, vec3f shift, bool swap_xyz, vec2f *&texs, tri3f *&ttris)
{
	vector<tri3f> triset;
	vector<vec3f> vtxset;
	vector<vec2f> texset;
	vector<tri3f> ttriset;

	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
				double x, y, z;
				sscanf(buf+2, "%lf%lf%lf", &x, &y, &z);

				if (swap_xyz)
					vtxset.push_back(vec3f(z, x, y)*scale+shift);
				else
					vtxset.push_back(vec3f(x, y, z)*scale+shift);
		} else

			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y;
				sscanf(buf + 3, "%lf%lf", &x, &y);

				texset.push_back(vec2f(x, y));
			}
			else
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2, id3=0;
				int tid0, tid1, tid2, tid3=0;
				bool quad = false;

				int count = sscanf(buf+2, "%d/%d", &id0, &tid0);
				char *nxt = strchr(buf+2, ' ');
				sscanf(nxt+1, "%d/%d", &id1, &tid1);
				nxt = strchr(nxt+1, ' ');
				sscanf(nxt+1, "%d/%d", &id2, &tid2);

				nxt = strchr(nxt+1, ' ');
				if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
					if (sscanf(nxt+1, "%d/%d", &id3, &tid3))
						quad = true;
				}

				id0--, id1--, id2--, id3--;
				tid0--, tid1--, tid2--, tid3--;

				triset.push_back(tri3f(id0, id1, id2));
				if (count == 2) {
					ttriset.push_back(tri3f(tid0, tid1, tid2));
				}

				if (quad) {
					triset.push_back(tri3f(id0, id2, id3));
					if (count == 2)
						ttriset.push_back(tri3f(tid0, tid2, tid3));
				}
			}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i=0; i<numVtx; i++)
		vtxs[i] = vtxset[i];

	int numTex = texset.size();
	if (numTex == 0)
		texs = NULL;
	else {
		texs = new vec2f[numTex];
		for (unsigned int i = 0; i < numTex; i++)
			texs[i] = texset[i];
	}

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i=0; i<numTri; i++)
		tris[i] = triset[i];

	int numTTri = ttriset.size();
	if (numTTri == 0)
		ttris = NULL;
	else {
		ttris = new tri3f[numTTri];
		for (unsigned int i = 0; i < numTTri; i++)
			ttris[i] = ttriset[i];
	}

	return true;
}

//http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
vec3f randDir()
{
	REAL phi = REAL(rand()) / RAND_MAX * M_PI * 2;
	REAL costheta = REAL(rand()) / RAND_MAX*2-1;
	REAL theta = acos(costheta);

	REAL x = sin(theta)*cos(phi);
	REAL y = sin(theta)*sin(phi);
	REAL z = cos(theta);

	vec3f ret(x, y, z);
	ret.normalize();
	return ret;
}

REAL randDegree()
{
	return  REAL(rand()) / RAND_MAX * 90;
}

kmesh* initCyl()
{
	const REAL width = 3.0f;
	const REAL radius = 0.5f;
	int splits = 64;

	vec3f* points = new vec3f[2 * splits];
	for (int i = 0; i < splits; i++)
	{
		const REAL cosTheta = cos(i * M_PI * 2.0f / REAL(splits));
		const REAL sinTheta = sin(i * M_PI * 2.0f / REAL(splits));
		const REAL y = radius * cosTheta;
		const REAL z = radius * sinTheta;
		points[2 * i + 0] = vec3f(-width / 2.0f, y, z);
		points[2 * i + 1] = vec3f(+width / 2.0f, y, z);
	}

	int numTri = splits * 2;
	tri3f* tris = new tri3f[numTri];

	int idx = 0;
	for (int i = 0; i < splits-1; i++)
	{
		tris[idx++] = tri3f(i * 2, i * 2 + 1, i * 2 + 2);
		tris[idx++] = tri3f(i * 2+2, i * 2 + 1, i * 2 + 3);
	}
	tris[idx++] = tri3f(splits * 2 - 2, splits * 2 - 1, 0);
	tris[idx++] = tri3f(0, splits * 2 - 1, 1);

	//reverse all
	for (int i = 0; i < splits * 2; i++) {
		tri3f& t = tris[i];
		t.reverse();
	}

	kmesh* cyl = new kmesh(splits*2, numTri, tris, points, true);
	g_scene.addMesh(cyl);
	return cyl;
}

kmesh* initBunny(const char* ofile)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;
	vec2f* texs = NULL;
	tri3f* ttris = NULL;

	REAL scale = BUNNY_SCALE;
	vec3f shift(0, 0, 0);

	if (false == readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris)) {
		printf("loading %s failed...\n", ofile);
		exit(-1);
	}
#if 0
	for (int i = 0; i < 10; i++) {
		vec3f& p = vtxs[i];
		printf("%lf, %lf, %lf\n", p.x, p.y, p.z);
	}
#endif

	kmesh* bunny = new kmesh(numVtx, numTri, tris, vtxs, false);
	g_scene.addMesh(bunny);
	return bunny;
}

inline float randAngle() {
	return REAL(rand()) / RAND_MAX * M_PI * 2;
}

crigid *shootMesh(kmesh *km, const vec3f &startPos, const vec3f &dstPos)
{
	float mass = 4.f;
	float m_ShootBoxInitialSpeed = 40.f;
	static int shot = 0;

	crigid* body = new crigid(km, startPos, mass);

	vec3f linVel = dstPos - startPos;
	linVel.normalize();
	linVel *= m_ShootBoxInitialSpeed * 0.25;

	body->getWorldTransform().setOrigin(startPos);

	//body->getWorldTransform().setRotation(quaternion(3.14159265 * shot * 0.1 * 0.5, 0, 3.14159265 * shot * 0.1 * 0.5));
	body->getWorldTransform().setRotation(quaternion(randAngle(), randAngle(), randAngle()));
	body->setLinearVelocity(linVel);
	body->setAngularVelocity(vec3f(0, 0, 0));

	shot++;
	if (shot == 10)
		shot = 0;

	return body;
}

void initRigids(kmesh* km)
{
	BOX bx = km->bound();
	float dx = bx.width() * 1.2;
	float dy = bx.height() * 1.2;
	float dz = bx.depth() * 1.2;
	vec3f off = vec3f(-dx * 4 * 0.5,  dy * 4, -dz * 4 * 0.5);

	//150 bunnys
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			for (int k = 0; k < 6; k++)
			{
				vec3f pos = vec3f(i*dx, k*dy,  j*dz)+off;
				vec3f dst = pos - vec3f(0, 10, 0);
				crigid* rig = shootMesh(km, pos, dst);
				g_box += rig->bound();
				g_scene.addRigid(rig);
			}
}

void initRigids1(kmesh * km) {
	int num = 2;// 150;
	for (int i = 0; i < num; i++) {
		crigid *rig = shootMesh(km, vec3f(0, 10 * (i + 1), 0), vec3f(0, 10 * i, 0));
		g_scene.addRigid(rig);

		if (i == 0)
			g_box += rig->bound();
	}
}

void initPlanes()
{
	if (gFlat)
		g_scene.addPlane(new cplane(vec3f(0, 1, 0), 10));
	else {
		g_scene.addPlane(new cplane(vec3f(0, 1, 1), 0));
		g_scene.addPlane(new cplane(vec3f(0, 1, -1), 0));
		g_scene.addPlane(new cplane(vec3f(1, 1, 0), 0));
		g_scene.addPlane(new cplane(vec3f(-1, 1, 0), 0));
	}

	g_box += vec3f();
	g_box += vec3f(10, 10, 10);
	g_box += vec3f(-10, -5, -10);

}

void initModel(const char *cfile)
{
	kmesh* kmB = initBunny(cfile);

	initRigids(kmB);
	initPlanes();

#ifdef GPU
	g_scene.push2GPU();
#endif
}


bool exportModel(const char* cfile)
{
	return g_scene.output(cfile);
}

bool importModel(const char* cfile)
{
	bool ret = g_scene.input(cfile);

#ifdef GPU
	if (ret)
		g_scene.update2GPU();
#endif

	return ret;
}

void quitModel()
{
	g_scene.clear();

#ifdef GPU
	clearGPU();
#endif
}

extern void beginDraw(BOX &);
extern void endDraw();

void drawOther();

void drawBVH(int level) {
	NULL;
}

void setMat(int i, int id);

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	if (!g_box.empty())
		beginDraw(g_box);

	drawOther();
	g_scene.draw(level);
#if 0
	for (int i=0; i<150; i++)
		if (lions[i]) {
			setMat(i, midBunny);

#if 1
			if (!pnt) {
				vec3f off = lions[i]->_off;
				vec3f axis = lions[i]->_axis;
				REAL theta = lions[i]->_theta;

				useBunnyDL(off.x, off.y, off.z, axis.x, axis.y, axis.z, theta);
			}
			else
#endif
				lions[i]->display(tri, false, false, level, true, i == 0 ? lion_set : dummy_set, dummy_vtx, i);
		}
#endif

	drawBVH(level);

	if (!g_box.empty())
		endDraw();
}

extern double totalQuery;

bool dynamicModel(char*, bool, bool)
{
	static int st = 0;
//	if (st == 29)
//		printf("here!\n");

	double tstart = omp_get_wtime();
	float dt = float(5.) / float(60.);
	g_scene.stepSimulation(dt);
	double tdelta = omp_get_wtime() - tstart;

	printf("stepSimulation %d: %2.5f s\n", st++, tdelta);
	totalQuery += tdelta;
	return true;
}

