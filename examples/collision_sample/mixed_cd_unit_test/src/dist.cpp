#include "cmesh.h"
#include <set>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "tmbvh.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@

inline
void
VcV(double Vr[3], const double V[3])
{
	Vr[0] = V[0];  Vr[1] = V[1];  Vr[2] = V[2];
}

inline
void
VmV(double Vr[3], const double V1[3], const double V2[3])
{
	Vr[0] = V1[0] - V2[0];
	Vr[1] = V1[1] - V2[1];
	Vr[2] = V1[2] - V2[2];
}


inline
double
VdistV2(const double V1[3], const double V2[3])
{
	return ((V1[0] - V2[0]) * (V1[0] - V2[0]) +
		(V1[1] - V2[1]) * (V1[1] - V2[1]) +
		(V1[2] - V2[2]) * (V1[2] - V2[2]));
}


inline
double
VdotV(const double V1[3], const double V2[3])
{
	return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]);
}


inline
void
VcrossV(double Vr[3], const double V1[3], const double V2[3])
{
	Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
	Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
	Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
}


inline
void
VpVxS(double Vr[3], const double V1[3], const double V2[3], double s)
{
	Vr[0] = V1[0] + V2[0] * s;
	Vr[1] = V1[1] + V2[1] * s;
	Vr[2] = V1[2] + V2[2] * s;
}

inline
void
VpV(double Vr[3], const double V1[3], const double V2[3])
{
	Vr[0] = V1[0] + V2[0];
	Vr[1] = V1[1] + V2[1];
	Vr[2] = V1[2] + V2[2];
}


inline
void
VxS(double Vr[3], const double V[3], double s)
{
	Vr[0] = V[0] * s;
	Vr[1] = V[1] * s;
	Vr[2] = V[2] * s;
}

//--------------------------------------------------------------------------
// SegPoints() 
//
// Returns closest points between an segment pair.
// Implemented from an algorithm described in
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.   
//--------------------------------------------------------------------------

void
SegPoints(double VEC[3],
	double X[3], double Y[3],             // closest points
	const double P[3], const double A[3], // seg 1 origin, vector
	const double Q[3], const double B[3]) // seg 2 origin, vector
{
	double T[3], A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
	double TMP[3];

	VmV(T, Q, P);
	A_dot_A = VdotV(A, A);
	B_dot_B = VdotV(B, B);
	A_dot_B = VdotV(A, B);
	A_dot_T = VdotV(A, T);
	B_dot_T = VdotV(B, T);

	// t parameterizes ray P,A 
	// u parameterizes ray Q,B 

	double t, u;

	// compute t for the closest point on ray P,A to
	// ray Q,B

	double denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

	t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

	// clamp result so t is on the segment P,A

	if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

	// find u for point on ray Q,B closest to point at t

	u = (t * A_dot_B - B_dot_T) / B_dot_B;

	// if u is on segment Q,B, t and u correspond to 
	// closest points, otherwise, clamp u, recompute and
	// clamp t 

	if ((u <= 0) || isnan(u)) {

		VcV(Y, Q);

		t = A_dot_T / A_dot_A;

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VmV(VEC, Q, P);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(VEC, Q, X);
		}
		else {
			VpVxS(X, P, A, t);
			VcrossV(TMP, T, A);
			VcrossV(VEC, A, TMP);
		}
	}
	else if (u >= 1) {

		VpV(Y, Q, B);

		t = (A_dot_B + A_dot_T) / A_dot_A;

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VmV(VEC, Y, P);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(VEC, Y, X);
		}
		else {
			VpVxS(X, P, A, t);
			VmV(T, Y, P);
			VcrossV(TMP, T, A);
			VcrossV(VEC, A, TMP);
		}
	}
	else {

		VpVxS(Y, Q, B, u);

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VcrossV(TMP, T, B);
			VcrossV(VEC, B, TMP);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(T, Q, X);
			VcrossV(TMP, T, B);
			VcrossV(VEC, B, TMP);
		}
		else {
			VpVxS(X, P, A, t);
			VcrossV(VEC, A, B);
			if (VdotV(VEC, T) < 0) {
				VxS(VEC, VEC, -1);
			}
		}
	}
}

//--------------------------------------------------------------------------
// TriDist() 
//
// Computes the closest points on two triangles, and returns the 
// distance between them.
// 
// S and T are the triangles, stored tri[point][dimension].
//
// If the triangles are disjoint, P and Q give the closest points of 
// S and T respectively. However, if the triangles overlap, P and Q 
// are basically a random pair of points from the triangles, not 
// coincident points on the intersection of the triangles, as might 
// be expected.
//--------------------------------------------------------------------------

double
TriDistPQP(double P[3], double Q[3], const double S[3][3], const double T[3][3])
{
	// Compute vectors along the 6 sides

	double Sv[3][3], Tv[3][3];
	double VEC[3];

	VmV(Sv[0], S[1], S[0]);
	VmV(Sv[1], S[2], S[1]);
	VmV(Sv[2], S[0], S[2]);

	VmV(Tv[0], T[1], T[0]);
	VmV(Tv[1], T[2], T[1]);
	VmV(Tv[2], T[0], T[2]);

	// For each edge pair, the vector connecting the closest points 
	// of the edges defines a slab (parallel planes at head and tail
	// enclose the slab). If we can show that the off-edge vertex of 
	// each triangle is outside of the slab, then the closest points
	// of the edges are the closest points for the triangles.
	// Even if these tests fail, it may be helpful to know the closest
	// points found, and whether the triangles were shown disjoint

	double V[3];
	double Z[3];
	double minP[3], minQ[3], mindd;
	int shown_disjoint = 0;

	mindd = VdistV2(S[0], T[0]) + 1;  // Set first minimum safely high

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			// Find closest points on edges i & j, plus the 
			// vector (and distance squared) between these points

			SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

			VmV(V, Q, P);
			double dd = VdotV(V, V);

			// Verify this closest point pair only if the distance 
			// squared is less than the minimum found thus far.

			if (dd <= mindd)
			{
				VcV(minP, P);
				VcV(minQ, Q);
				mindd = dd;

				VmV(Z, S[(i + 2) % 3], P);
				double a = VdotV(Z, VEC);
				VmV(Z, T[(j + 2) % 3], Q);
				double b = VdotV(Z, VEC);

				if ((a <= 0) && (b >= 0))
					//return sqrt(dd);
					return dd;

				double p = VdotV(V, VEC);

				if (a < 0) a = 0;
				if (b > 0) b = 0;
				if ((p - a + b) > 0) shown_disjoint = 1;
			}
		}
	}

	// No edge pairs contained the closest points.  
	// either:
	// 1. one of the closest points is a vertex, and the
	//    other point is interior to a face.
	// 2. the triangles are overlapping.
	// 3. an edge of one triangle is parallel to the other's face. If
	//    cases 1 and 2 are not true, then the closest points from the 9
	//    edge pairs checks above can be taken as closest points for the
	//    triangles.
	// 4. possibly, the triangles were degenerate.  When the 
	//    triangle points are nearly colinear or coincident, one 
	//    of above tests might fail even though the edges tested
	//    contain the closest points.

	// First check for case 1

	double Sn[3], Snl;
	VcrossV(Sn, Sv[0], Sv[1]); // Compute normal to S triangle
	Snl = VdotV(Sn, Sn);      // Compute square of length of normal

	// If cross product is long enough,

	if (Snl > 1e-15)
	{
		// Get projection lengths of T points

		double Tp[3];

		VmV(V, S[0], T[0]);
		Tp[0] = VdotV(V, Sn);

		VmV(V, S[0], T[1]);
		Tp[1] = VdotV(V, Sn);

		VmV(V, S[0], T[2]);
		Tp[2] = VdotV(V, Sn);

		// If Sn is a separating direction,
		// find point with smallest projection

		int point = -1;
		if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
		{
			if (Tp[0] < Tp[1]) point = 0; else point = 1;
			if (Tp[2] < Tp[point]) point = 2;
		}
		else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
		{
			if (Tp[0] > Tp[1]) point = 0; else point = 1;
			if (Tp[2] > Tp[point]) point = 2;
		}

		// If Sn is a separating direction, 

		if (point >= 0)
		{
			shown_disjoint = 1;

			// Test whether the point found, when projected onto the 
			// other triangle, lies within the face.

			VmV(V, T[point], S[0]);
			VcrossV(Z, Sn, Sv[0]);
			if (VdotV(V, Z) > 0)
			{
				VmV(V, T[point], S[1]);
				VcrossV(Z, Sn, Sv[1]);
				if (VdotV(V, Z) > 0)
				{
					VmV(V, T[point], S[2]);
					VcrossV(Z, Sn, Sv[2]);
					if (VdotV(V, Z) > 0)
					{
						// T[point] passed the test - it's a closest point for 
						// the T triangle; the other point is on the face of S

						VpVxS(P, T[point], Sn, Tp[point] / Snl);
						VcV(Q, T[point]);
						//return sqrt(VdistV2(P, Q));
						return VdistV2(P, Q);
					}
				}
			}
		}
	}

	double Tn[3], Tnl;
	VcrossV(Tn, Tv[0], Tv[1]);
	Tnl = VdotV(Tn, Tn);

	if (Tnl > 1e-15)
	{
		double Sp[3];

		VmV(V, T[0], S[0]);
		Sp[0] = VdotV(V, Tn);

		VmV(V, T[0], S[1]);
		Sp[1] = VdotV(V, Tn);

		VmV(V, T[0], S[2]);
		Sp[2] = VdotV(V, Tn);

		int point = -1;
		if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
		{
			if (Sp[0] < Sp[1]) point = 0; else point = 1;
			if (Sp[2] < Sp[point]) point = 2;
		}
		else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
		{
			if (Sp[0] > Sp[1]) point = 0; else point = 1;
			if (Sp[2] > Sp[point]) point = 2;
		}

		if (point >= 0)
		{
			shown_disjoint = 1;

			VmV(V, S[point], T[0]);
			VcrossV(Z, Tn, Tv[0]);
			if (VdotV(V, Z) > 0)
			{
				VmV(V, S[point], T[1]);
				VcrossV(Z, Tn, Tv[1]);
				if (VdotV(V, Z) > 0)
				{
					VmV(V, S[point], T[2]);
					VcrossV(Z, Tn, Tv[2]);
					if (VdotV(V, Z) > 0)
					{
						VcV(P, S[point]);
						VpVxS(Q, S[point], Tn, Sp[point] / Tnl);
						//return sqrt(VdistV2(P, Q));
						return VdistV2(P, Q);
					}
				}
			}
		}
	}

	// Case 1 can't be shown.
	// If one of these tests showed the triangles disjoint,
	// we assume case 3 or 4, otherwise we conclude case 2, 
	// that the triangles overlap.

	if (shown_disjoint)
	{
		VcV(P, minP);
		VcV(Q, minQ);
		//return sqrt(mindd);
		return mindd;
	}
	else return 0;
}

double
triDist(const vec3f& P1, const vec3f& P2, const vec3f& P3,
	const vec3f& Q1, const vec3f& Q2, const vec3f& Q3,
	vec3f& rP, vec3f& rQ)
{
	double S[3][3];
	double T[3][3];
	double P[3];
	double Q[3];

	S[0][0] = P1.x;
	S[0][1] = P1.y;
	S[0][2] = P1.z;
	S[1][0] = P2.x;
	S[1][1] = P2.y;
	S[1][2] = P2.z;
	S[2][0] = P3.x;
	S[2][1] = P3.y;
	S[2][2] = P3.z;

	T[0][0] = Q1.x;
	T[0][1] = Q1.y;
	T[0][2] = Q1.z;
	T[1][0] = Q2.x;
	T[1][1] = Q2.y;
	T[1][2] = Q2.z;
	T[2][0] = Q3.x;
	T[2][1] = Q3.y;
	T[2][2] = Q3.z;

	double d = TriDistPQP(P, Q, S, T);

	rP = vec3f(P[0], P[1], P[2]);
	rQ = vec3f(Q[0], Q[1], Q[2]);
	return d;
}


/////////////////////////////////////
inline double fmax(double a, double b, double c)
{
	double t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

inline double fmin(double a, double b, double c)
{
	double t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

inline int project3(const vec3f& ax,
	const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

inline int project6(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);
	double Q1 = ax.dot(q1);
	double Q2 = ax.dot(q2);
	double Q3 = ax.dot(q3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);
	double mx2 = fmax(Q1, Q2, Q3);
	double mn2 = fmin(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

bool
triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
	vec3f p1;
	vec3f p2 = P2 - P1;
	vec3f p3 = P3 - P1;
	vec3f q1 = Q1 - P1;
	vec3f q2 = Q2 - P1;
	vec3f q3 = Q3 - P1;

	vec3f e1 = p2 - p1;
	vec3f e2 = p3 - p2;
	vec3f e3 = p1 - p3;

	vec3f f1 = q2 - q1;
	vec3f f2 = q3 - q2;
	vec3f f3 = q1 - q3;

	vec3f n1 = e1.cross(e2);
	vec3f m1 = f1.cross(f2);

	vec3f g1 = e1.cross(n1);
	vec3f g2 = e2.cross(n1);
	vec3f g3 = e3.cross(n1);

	vec3f  h1 = f1.cross(m1);
	vec3f h2 = f2.cross(m1);
	vec3f h3 = f3.cross(m1);

	vec3f ef11 = e1.cross(f1);
	vec3f ef12 = e1.cross(f2);
	vec3f ef13 = e1.cross(f3);
	vec3f ef21 = e2.cross(f1);
	vec3f ef22 = e2.cross(f2);
	vec3f ef23 = e2.cross(f3);
	vec3f ef31 = e3.cross(f1);
	vec3f ef32 = e3.cross(f2);
	vec3f ef33 = e3.cross(f3);

	// now begin the series of tests
	if (!project3(n1, q1, q2, q3)) return false;
	if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

	return true;
}
