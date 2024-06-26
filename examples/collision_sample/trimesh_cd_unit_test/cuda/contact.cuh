#pragma once

CU_FORCEINLINE __host__ __device__ float calcArea4Points(const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3)
{
	// It calculates possible 3 area constructed from random 4 points and returns the biggest one.

	REAL3 a[3], b[3];
	a[0] = p0 - p1;
	a[1] = p0 - p2;
	a[2] = p0 - p3;
	b[0] = p2 - p3;
	b[1] = p1 - p3;
	b[2] = p1 - p2;

	//todo: Following 3 cross production can be easily optimized by SIMD.
	REAL3 tmp0 = cross(a[0], b[0]);
	REAL3 tmp1 = cross(a[1], b[1]);
	REAL3 tmp2 = cross(a[2], b[2]);

	return fmax(fmax(norm2(tmp0), norm2(tmp1)), norm2(tmp2));
}

typedef struct __align__(16) _manifoldPoint {
	REAL3 m_localPointA;
	REAL3 m_localPointB;
	REAL3 m_positionWorldOnB;
	REAL3 m_positionWorldOnA;
	REAL3 m_normalWorldOnB;
	REAL3 m_lateralFrictionDir1;
	REAL3 m_lateralFrictionDir2;

	float	m_distance1;
	float	m_combinedFriction;
	float	m_combinedRollingFriction;
	float	m_combinedRestitution;

	float		m_appliedImpulse;
	float		m_appliedImpulseLateral1;
	float		m_appliedImpulseLateral2;
	float		m_contactMotion1;
	float		m_contactMotion2;
	float		m_contactCFM1;
	float		m_contactCFM2;

	CU_FORCEINLINE __host__ __device__ _manifoldPoint(const REAL3 & pointA, const REAL3 & pointB, const REAL3 & normal, float distance) :
		m_localPointA(pointA),
		m_localPointB(pointB),
		m_normalWorldOnB(normal),
		m_distance1(distance),
		m_combinedFriction(float(0.)),
		m_combinedRollingFriction(float(0.)),
		m_combinedRestitution(float(0.)),
		m_appliedImpulse(0.f),
		m_appliedImpulseLateral1(0.f),
		m_appliedImpulseLateral2(0.f),
		m_contactMotion1(0.f),
		m_contactMotion2(0.f),
		m_contactCFM1(0.f),
		m_contactCFM2(0.f)
	{

	}

	CU_FORCEINLINE __host__ __device__ float getDistance() const
	{
		return m_distance1;
	}

} g_manifoldPoint;

#define MANIFOLD_CACHE_SIZE 4

typedef struct __align__(16) _manifold {
	g_manifoldPoint m_pointCache[MANIFOLD_CACHE_SIZE];

	int m_plane;
	int m_body0;
	int m_body1;

	int	m_cachedPoints;
	float	m_contactBreakingThreshold;
	float	m_contactProcessingThreshold;

	CU_FORCEINLINE __host__ __device__ void set(const uint2 &rids) {
		m_plane = -1;
		m_body0 = rids.x;
		m_body1 = rids.y;
		m_cachedPoints = 0;
		m_contactBreakingThreshold = 0.128469273;
		m_contactProcessingThreshold = GLH_LARGE_FLOAT;
	}

	CU_FORCEINLINE __host__ __device__ int	getNumContacts() const { return m_cachedPoints; }

	CU_FORCEINLINE __host__ __device__ float	getContactBreakingThreshold() const
	{
		return m_contactBreakingThreshold;
	}

	/// sort cached points so most isolated points come first
	CU_FORCEINLINE __host__ __device__ int sortCachedPoints(const g_manifoldPoint& pt)
	{
		//calculate 4 possible cases areas, and take biggest area
		//also need to keep 'deepest'

		int maxPenetrationIndex = -1;
#define KEEP_DEEPEST_POINT 1
#ifdef KEEP_DEEPEST_POINT
		float maxPenetration = pt.getDistance();
		for (int i = 0; i < 4; i++)
		{
			if (m_pointCache[i].getDistance() < maxPenetration)
			{
				maxPenetrationIndex = i;
				maxPenetration = m_pointCache[i].getDistance();
			}
		}
#endif //KEEP_DEEPEST_POINT

		float res0(float(0.)), res1(float(0.)), res2(float(0.)), res3(float(0.));
		{
			if (maxPenetrationIndex != 0)
			{
				REAL3 a0 = pt.m_localPointA - m_pointCache[1].m_localPointA;
				REAL3 b0 = m_pointCache[3].m_localPointA - m_pointCache[2].m_localPointA;
				REAL3 t = cross(a0, b0);
				res0 = norm2(t);
			}
			if (maxPenetrationIndex != 1)
			{
				REAL3 a1 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				REAL3 b1 = m_pointCache[3].m_localPointA - m_pointCache[2].m_localPointA;
				REAL3 t = cross(a1, b1);
				res1 = norm2(t);
			}

			if (maxPenetrationIndex != 2)
			{
				REAL3 a2 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				REAL3 b2 = m_pointCache[3].m_localPointA - m_pointCache[1].m_localPointA;
				REAL3 t = cross(a2, b2);
				res2 = norm2(t);
			}

			if (maxPenetrationIndex != 3)
			{
				REAL3 a3 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				REAL3 b3 = m_pointCache[2].m_localPointA - m_pointCache[1].m_localPointA;
				REAL3 t = cross(a3, b3);
				res3 = norm2(t);
			}
		}

		REAL4 maxvec = make_REAL4(res0, res1, res2, res3);
		int biggestarea = closestAxis4(maxvec);
		return biggestarea;
	}

	CU_FORCEINLINE __host__ __device__ int getCacheEntry(const g_manifoldPoint& newPoint) const
	{
		float shortestDist = getContactBreakingThreshold() * getContactBreakingThreshold();
		int size = getNumContacts();
		int nearestPoint = -1;
		for (int i = 0; i < size; i++)
		{
			const g_manifoldPoint& mp = m_pointCache[i];

			REAL3 diffA = mp.m_localPointA - newPoint.m_localPointA;
			const float distToManiPoint = dot(diffA, diffA);
			if (distToManiPoint < shortestDist)
			{
				shortestDist = distToManiPoint;
				nearestPoint = i;
			}
		}
		return nearestPoint;
	}

	CU_FORCEINLINE __host__ __device__ int addManifoldPoint(const g_manifoldPoint& newPoint, bool isPredictive = false)
	{
		int insertIndex = getNumContacts();
		if (insertIndex == MANIFOLD_CACHE_SIZE)
		{
#if MANIFOLD_CACHE_SIZE >= 4
			//sort cache so best points come first, based on area
			insertIndex = sortCachedPoints(newPoint);
#else
			insertIndex = 0;
#endif
		}
		else
		{
			m_cachedPoints++;


		}
		if (insertIndex < 0)
			insertIndex = 0;

		m_pointCache[insertIndex] = newPoint;
		return insertIndex;
	}

	CU_FORCEINLINE __host__ __device__ void replaceContactPoint(const g_manifoldPoint& newPoint, int insertIndex)
	{
		float	appliedImpulse = m_pointCache[insertIndex].m_appliedImpulse;
		float	appliedLateralImpulse1 = m_pointCache[insertIndex].m_appliedImpulseLateral1;
		float	appliedLateralImpulse2 = m_pointCache[insertIndex].m_appliedImpulseLateral2;
		m_pointCache[insertIndex] = newPoint;
		m_pointCache[insertIndex].m_appliedImpulse = appliedImpulse;
		m_pointCache[insertIndex].m_appliedImpulseLateral1 = appliedLateralImpulse1;
		m_pointCache[insertIndex].m_appliedImpulseLateral2 = appliedLateralImpulse2;
		m_pointCache[insertIndex].m_appliedImpulse = appliedImpulse;
		m_pointCache[insertIndex].m_appliedImpulseLateral1 = appliedLateralImpulse1;
		m_pointCache[insertIndex].m_appliedImpulseLateral2 = appliedLateralImpulse2;
	}

} g_manifold;

//##################################################################

/// Calc a plane from a triangle edge an a normal. plane is a REAL4
CU_FORCEINLINE __host__ __device__ void get_edge_plane2(const REAL3& e1, const REAL3& e2, const REAL3& normal, REAL4& plane)
{
	REAL3 planenormal = cross(e2 - e1, normal);
	REAL3 t = normalize(planenormal);
	plane = make_REAL4(t.x, t.y, t.z, dot(e2, t));
}

CU_FORCEINLINE __host__ __device__ float distance_point_plane(const REAL4& plane, const REAL3& point)
{
	return dot(point, make_REAL3(plane.x, plane.y, plane.z)) - plane.w;
}

/*! Vector blending
Takes two vectors a, b, blends them together*/
CU_FORCEINLINE __host__ __device__ void get_vec_blend(REAL3& vr, const REAL3& va, const REAL3& vb, float blend_factor)
{
	vr = (1 - blend_factor) * va + blend_factor * vb;
}

//! This function calcs the distance from a 3D plane
CU_FORCEINLINE __host__ __device__ void get_plane_clip_polygon_collect(
	const REAL3& point0,
	const REAL3& point1,
	float dist0,
	float dist1,
	REAL3* clipped,
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


CU_FORCEINLINE __host__ __device__ int get_plane_clip_triangle(
	const REAL4& plane,
	const REAL3& point0,
	const REAL3& point1,
	const REAL3& point2,
	REAL3* clipped // an allocated array of 16 points at least
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
typedef struct _triangleContact
{
	float m_penetration_depth;
	int m_point_count;
	REAL4 m_separating_normal;
	REAL3 m_points[MAX_TRI_CLIPPING];

	CU_FORCEINLINE __host__ __device__ void copy_from(const _triangleContact& other)
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

	CU_FORCEINLINE __host__ __device__ _triangleContact()
	{
	}

	CU_FORCEINLINE __host__ __device__ _triangleContact(const _triangleContact& other)
	{
		copy_from(other);
	}

	//! classify points that are closer
	CU_FORCEINLINE __host__ __device__ void merge_points(const REAL4& plane, float margin, const REAL3* points, int point_count)
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
} gtriangleContact;

typedef struct _primitiveTriangle
{
	REAL3 m_vertices[3];
	REAL4 m_plane;
	float m_margin;

	CU_FORCEINLINE __host__ __device__ _primitiveTriangle() : m_margin(0.01f)
	{
	}

	CU_FORCEINLINE __host__ __device__ void buildTriPlane()
	{
		REAL3 normal = cross(m_vertices[1] - m_vertices[0], m_vertices[2] - m_vertices[0]);
		REAL3 n = normalize(normal);
		m_plane = make_REAL4(n.x, n.y, n.z, dot(m_vertices[0], n));
	}

	//! Test if triangles could collide
	CU_FORCEINLINE __host__ __device__ bool overlap_test_conservative(const _primitiveTriangle& other)
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
	CU_FORCEINLINE __host__ __device__ void get_edge_plane(int edge_index, REAL4& plane)  const
	{
		const REAL3& e0 = m_vertices[edge_index];
		const REAL3& e1 = m_vertices[(edge_index + 1) % 3];
		get_edge_plane2(e0, e1, make_REAL3(m_plane.x, m_plane.y, m_plane.z), plane);
	}

	CU_FORCEINLINE __host__ __device__ void applyTransform(const g_transf& t)
	{
		m_vertices[0] = t(m_vertices[0]);
		m_vertices[1] = t(m_vertices[1]);
		m_vertices[2] = t(m_vertices[2]);
	}


	//! Clips a polygon by a plane
	/*!
	*\return The count of the clipped counts
	*/
	CU_FORCEINLINE __host__ __device__ int get_plane_clip_polygon(
		const REAL4 & plane,
		const REAL3 * polygon_points,
		int polygon_point_count,
		REAL3 * clipped)
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

	//! Clips the triangle against this
	/*!
	\pre clipped_points must have MAX_TRI_CLIPPING size, and this triangle must have its plane calculated.
	\return the number of clipped points
	*/
	CU_FORCEINLINE __host__ __device__ int clip_triangle(_primitiveTriangle& other, REAL3* clipped_points)
	{
		// edge 0

		REAL3 temp_points[MAX_TRI_CLIPPING];


		REAL4 edgeplane;

		get_edge_plane(0, edgeplane);


		int clipped_count = get_plane_clip_triangle(
			edgeplane, other.m_vertices[0], other.m_vertices[1], other.m_vertices[2], temp_points);

		if (clipped_count == 0) return 0;

		REAL3 temp_points1[MAX_TRI_CLIPPING];


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
	CU_FORCEINLINE __host__ __device__ bool find_triangle_collision_clip_method(_primitiveTriangle& other, _triangleContact& contacts)
	{
		float margin = m_margin + other.m_margin;

		REAL3 clipped_points[MAX_TRI_CLIPPING];
		int clipped_count;
		//create planes
		// plane v vs U points

		_triangleContact contacts1;

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
		_triangleContact contacts2;
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

} gprimitiveTriangle;
