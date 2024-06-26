#pragma once

float gContactBreakingThreshold = float(0.02);
bool gContactCalcArea3Points = true;

inline float calcArea4Points(const vec3f& p0, const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	// It calculates possible 3 area constructed from random 4 points and returns the biggest one.

	vec3f a[3], b[3];
	a[0] = p0 - p1;
	a[1] = p0 - p2;
	a[2] = p0 - p3;
	b[0] = p2 - p3;
	b[1] = p1 - p3;
	b[2] = p1 - p2;

	//todo: Following 3 cross production can be easily optimized by SIMD.
	vec3f tmp0 = a[0].cross(b[0]);
	vec3f tmp1 = a[1].cross(b[1]);
	vec3f tmp2 = a[2].cross(b[2]);

	return fmax(fmax(tmp0.length2(), tmp1.length2()), tmp2.length2());
}

/// ManifoldContactPoint collects and maintains persistent contactpoints.
/// used to improve stability and performance of rigidbody dynamics response.
class alignas(16) manifoldPoint
{
public:
	manifoldPoint() :
		m_appliedImpulse(0.f),
		m_appliedImpulseLateral1(0.f),
		m_appliedImpulseLateral2(0.f),
		m_contactMotion1(0.f),
		m_contactMotion2(0.f),
		m_contactCFM1(0.f),
		m_contactCFM2(0.f)
	{
	}

	manifoldPoint(const vec3f& pointA, const vec3f& pointB,
		const vec3f& normal,
		float distance) :
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



	vec3f m_localPointA;
	vec3f m_localPointB;
	vec3f m_positionWorldOnB;
	///m_positionWorldOnA is redundant information, see getPositionWorldOnA(), but for clarity
	vec3f m_positionWorldOnA;
	vec3f m_normalWorldOnB;
	vec3f m_lateralFrictionDir1;
	vec3f m_lateralFrictionDir2;

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


	float getDistance() const
	{
		return m_distance1;
	}

	const vec3f& getPositionWorldOnA() const {
		return m_positionWorldOnA;
		//				return m_positionWorldOnB + m_normalWorldOnB * m_distance1;
	}

	const vec3f& getPositionWorldOnB() const
	{
		return m_positionWorldOnB;
	}

	void	setDistance(float dist)
	{
		m_distance1 = dist;
	}

	///this returns the most recent applied impulse, to satisfy contact constraints by the constraint solver
	float	getAppliedImpulse() const
	{
		return m_appliedImpulse;
	}
};


#define MANIFOLD_CACHE_SIZE 4

///manifold is a contact point cache, it stays persistent as long as objects are overlapping in the broadphase.
///Those contact points are created by the collision narrow phase.
///The cache can be empty, or hold 1,2,3 or 4 points. Some collision algorithms (GJK) might only add one point at a time.
///updates/refreshes old contact points, and throw them away if necessary (distance becomes too large)
///reduces the cache to 4 points, when more then 4 points are added, using following rules:
///the contact point with deepest penetration is always kept, and it tries to maximuze the area covered by the points
///note that some pairs of objects might have more then one contact manifold.


class alignas(16) manifold {

	manifoldPoint m_pointCache[MANIFOLD_CACHE_SIZE];

	/// this two body pointers can point to the physics rigidbody class.
	int m_plane;
	int m_body0;
	int m_body1;

	int	m_cachedPoints;

	float	m_contactBreakingThreshold;
	float	m_contactProcessingThreshold;


	/// sort cached points so most isolated points come first
	int	sortCachedPoints(const manifoldPoint& pt)
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

		if (gContactCalcArea3Points)
		{
			if (maxPenetrationIndex != 0)
			{
				vec3f a0 = pt.m_localPointA - m_pointCache[1].m_localPointA;
				vec3f b0 = m_pointCache[3].m_localPointA - m_pointCache[2].m_localPointA;
				vec3f cross = a0.cross(b0);
				res0 = cross.length2();
			}
			if (maxPenetrationIndex != 1)
			{
				vec3f a1 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				vec3f b1 = m_pointCache[3].m_localPointA - m_pointCache[2].m_localPointA;
				vec3f cross = a1.cross(b1);
				res1 = cross.length2();
			}

			if (maxPenetrationIndex != 2)
			{
				vec3f a2 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				vec3f b2 = m_pointCache[3].m_localPointA - m_pointCache[1].m_localPointA;
				vec3f cross = a2.cross(b2);
				res2 = cross.length2();
			}

			if (maxPenetrationIndex != 3)
			{
				vec3f a3 = pt.m_localPointA - m_pointCache[0].m_localPointA;
				vec3f b3 = m_pointCache[2].m_localPointA - m_pointCache[1].m_localPointA;
				vec3f cross = a3.cross(b3);
				res3 = cross.length2();
			}
		}
		else
		{
			if (maxPenetrationIndex != 0) {
				res0 = calcArea4Points(pt.m_localPointA, m_pointCache[1].m_localPointA, m_pointCache[2].m_localPointA, m_pointCache[3].m_localPointA);
			}

			if (maxPenetrationIndex != 1) {
				res1 = calcArea4Points(pt.m_localPointA, m_pointCache[0].m_localPointA, m_pointCache[2].m_localPointA, m_pointCache[3].m_localPointA);
			}

			if (maxPenetrationIndex != 2) {
				res2 = calcArea4Points(pt.m_localPointA, m_pointCache[0].m_localPointA, m_pointCache[1].m_localPointA, m_pointCache[3].m_localPointA);
			}

			if (maxPenetrationIndex != 3) {
				res3 = calcArea4Points(pt.m_localPointA, m_pointCache[0].m_localPointA, m_pointCache[1].m_localPointA, m_pointCache[2].m_localPointA);
			}
		}
		vec4f maxvec(res0, res1, res2, res3);
		int biggestarea = maxvec.closestAxis4();
		return biggestarea;
	}

	int		findContactPoint(const manifoldPoint* unUsed, int numUnused, const manifoldPoint& pt);

public:

	manifold(int body0, int pln, int body1, int, float contactBreakingThreshold, float contactProcessingThreshold)
		: m_body0(body0), m_plane(pln), m_body1(body1), m_cachedPoints(0),
		m_contactBreakingThreshold(contactBreakingThreshold),
		m_contactProcessingThreshold(contactProcessingThreshold)
	{
	}

	__forceinline int getBody0() const { return m_body0; }
	__forceinline int getPlane() const { return m_plane; }
	__forceinline int getBody1() const { return m_body1; }

	__forceinline int	getNumContacts() const { return m_cachedPoints; }
	/// the setNumContacts API is usually not used, except when you gather/fill all contacts manually
	void setNumContacts(int cachedPoints)
	{
		m_cachedPoints = cachedPoints;
	}


	__forceinline const manifoldPoint& getContactPoint(int index) const
	{
		assert(index < m_cachedPoints);
		return m_pointCache[index];
	}

	__forceinline manifoldPoint& getContactPoint(int index)
	{
		assert(index < m_cachedPoints);
		return m_pointCache[index];
	}

	///@todo: get this margin from the current physics / collision environment
	float	getContactBreakingThreshold() const
	{
		return m_contactBreakingThreshold;
	}

	float	getContactProcessingThreshold() const
	{
		return m_contactProcessingThreshold;
	}

	void setContactBreakingThreshold(float contactBreakingThreshold)
	{
		m_contactBreakingThreshold = contactBreakingThreshold;
	}

	void setContactProcessingThreshold(float	contactProcessingThreshold)
	{
		m_contactProcessingThreshold = contactProcessingThreshold;
	}

	int getCacheEntry(const manifoldPoint& newPoint) const
	{
		float shortestDist = getContactBreakingThreshold() * getContactBreakingThreshold();
		int size = getNumContacts();
		int nearestPoint = -1;
		for (int i = 0; i < size; i++)
		{
			const manifoldPoint& mp = m_pointCache[i];

			vec3f diffA = mp.m_localPointA - newPoint.m_localPointA;
			const float distToManiPoint = diffA.dot(diffA);
			if (distToManiPoint < shortestDist)
			{
				shortestDist = distToManiPoint;
				nearestPoint = i;
			}
		}
		return nearestPoint;
	}

	int addManifoldPoint(const manifoldPoint& newPoint, bool isPredictive = false)
	{
		if (!isPredictive)
		{
			assert(validContactDistance(newPoint));
		}

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

	void removeContactPoint(int index)
	{
		int lastUsedIndex = getNumContacts() - 1;
		//		m_pointCache[index] = m_pointCache[lastUsedIndex];
		if (index != lastUsedIndex)
		{
			m_pointCache[index] = m_pointCache[lastUsedIndex];
			//get rid of duplicated userPersistentData pointer
			m_pointCache[lastUsedIndex].m_appliedImpulse = 0.f;
			m_pointCache[lastUsedIndex].m_appliedImpulseLateral1 = 0.f;
			m_pointCache[lastUsedIndex].m_appliedImpulseLateral2 = 0.f;
		}

		m_cachedPoints--;
	}
	void replaceContactPoint(const manifoldPoint& newPoint, int insertIndex)
	{
		assert(validContactDistance(newPoint));

#define MAINTAIN_PERSISTENCY 1
#ifdef MAINTAIN_PERSISTENCY
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
#else
		m_pointCache[insertIndex] = newPoint;
#endif
	}


	bool validContactDistance(const manifoldPoint& pt) const
	{
		return pt.m_distance1 <= getContactBreakingThreshold();
	}

	/// calculated new worldspace coordinates and depth, and reject points that exceed the collision margin
	void	refreshContactPoints(const transf& trA, const transf& trB);


	__forceinline	void	clearManifold()
	{
		m_cachedPoints = 0;
	}
};
