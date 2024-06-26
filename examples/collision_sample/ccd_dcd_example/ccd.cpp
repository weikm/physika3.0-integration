#include "ccd.h"

#define REAL_infinity 1.0e30

inline vec3f norm(const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	return (p2 - p1).cross(p3 - p1);
}

inline REAL stp(const vec3f& u, const vec3f& v, const vec3f& w)
{
	return u.dot(v.cross(w));
}

inline vec3f xvpos(const vec3f &x, const vec3f &v, REAL t)
{
	return x + v * t;
}


inline int sgn(REAL x) { return x < 0 ? -1 : 1; }
inline void fswap(REAL& a, REAL& b) {
	REAL t = b;
	b = a;
	a = t;
}

inline int solve_quadratic(REAL a, REAL b, REAL c, REAL x[2]) {
	// http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
	REAL d = b * b - 4 * a * c;
	if (d < 0) {
		x[0] = -b / (2 * a);
		return 0;
	}
	REAL q = -(b + sgn(b) * sqrt(d)) / 2;
	int i = 0;
	if (abs(a) > 1e-12 * abs(q))
		x[i++] = q / a;
	if (abs(q) > 1e-12 * abs(c))
		x[i++] = c / q;
	if (i == 2 && x[0] > x[1])
		fswap(x[0], x[1]);
	return i;
}

inline REAL newtons_method(REAL a, REAL b, REAL c, REAL d, REAL x0, int init_dir)
{
	if (init_dir != 0) {
		// quadratic approximation around x0, assuming y' = 0
		REAL y0 = d + x0 * (c + x0 * (b + x0 * a)),
			ddy0 = 2 * b + x0 * (6 * a);
		x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
	}
	for (int iter = 0; iter < 100; iter++) {
		REAL y = d + x0 * (c + x0 * (b + x0 * a));
		REAL dy = c + x0 * (2 * b + x0 * 3 * a);
		if (dy == 0)
			return x0;
		REAL x1 = x0 - y / dy;
		if (abs(x0 - x1) < 1e-6)
			return x0;
		x0 = x1;
	}
	return x0;
}

// solves a x^3 + b x^2 + c x + d == 0
inline int solve_cubic(REAL a, REAL b, REAL c, REAL d, REAL x[])
{
	REAL xc[2];
	int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
	if (ncrit == 0) {
		x[0] = newtons_method(a, b, c, d, xc[0], 0);
		return 1;
	}
	else if (ncrit == 1) {// cubic is actually quadratic
		return solve_quadratic(b, c, d, x);
	}
	else {
		REAL yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
			d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
		int i = 0;
		if (yc[0] * a >= 0)
			x[i++] = newtons_method(a, b, c, d, xc[0], -1);
		if (yc[0] * yc[1] <= 0) {
			int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
			x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
		}
		if (yc[1] * a <= 0)
			x[i++] = newtons_method(a, b, c, d, xc[1], 1);
		return i;
	}
}

inline bool
dnfFilter(
	const vec3f& a0, const vec3f& b0, const vec3f& c0, const vec3f& d0,
	const vec3f& a1, const vec3f& b1, const vec3f& c1, const vec3f& d1)
{
	vec3f n0 = norm(a0, b0, c0);
	vec3f n1 = norm(a1, b1, c1);
	vec3f delta = norm(a1 - a0, b1 - b0, c1 - c0);
	vec3f nX = (n0 + n1 - delta) * REAL(0.5);

	vec3f pa0 = d0 - a0;
	vec3f pa1 = d1 - a1;

	REAL A = n0.dot(pa0);
	REAL B = n1.dot(pa1);
	REAL C = nX.dot(pa0);
	REAL D = nX.dot(pa1);
	REAL E = n1.dot(pa0);
	REAL F = n0.dot(pa1);

	if (A > 0 && B > 0 && (REAL(2.0) * C + F) > 0 && (REAL(2.0) * D + E) > 0)
		return false;

	if (A < 0 && B < 0 && (REAL(2.0) * C + F) < 0 && (REAL(2.0) * D + E) < 0)
		return false;

	return true;
}


REAL signed_vf_distance(const vec3f& x, const vec3f& y0, const vec3f& y1, const vec3f& y2, vec3f &n, REAL w[])
{
	vec3f y10 = y1 - y0;
	vec3f y20 = y2 - y0;
	y10.normalize();
	y20.normalize();
	n = y10.cross(y20);

	if (n.dot(n) < 1e-6)
		return REAL_infinity;
	
	n.normalize();
	REAL h = (x - y0).dot(n);
	REAL b0 = stp(y1 - x, y2 - x, n),
		b1 = stp(y2 - x, y0 - x, n),
		b2 = stp(y0 - x, y1 - x, n);
	w[0] = 1;
	w[1] = -b0 / (b0 + b1 + b2);
	w[2] = -b1 / (b0 + b1 + b2);
	w[3] = -b2 / (b0 + b1 + b2);
	return h;
}

REAL signed_ee_distance(const vec3f& x0, const vec3f& x1, const vec3f& y0, const vec3f& y1, vec3f &n, REAL w[])
{
#if 0
	vec3f _n; if (!n) n = &_n;
	REAL _w[4]; if (!w) w = _w;
	*n = cross(normalize(x1 - x0), normalize(y1 - y0));
	if (norm2(*n) < 1e-6)
		return REAL_infinity;
	*n = normalize(*n);
	REAL h = dot(x0 - y0, *n);
	REAL a0 = stp(y1 - x1, y0 - x1, *n), a1 = stp(y0 - x0, y1 - x0, *n),
		b0 = stp(x0 - y1, x1 - y1, *n), b1 = stp(x1 - y0, x0 - y0, *n);
	w[0] = a0 / (a0 + a1);
	w[1] = a1 / (a0 + a1);
	w[2] = -b0 / (b0 + b1);
	w[3] = -b1 / (b0 + b1);
	return h;
#else
	return -1.0;
#endif
}

bool collisionTest(
	const vec3f& x0, const vec3f& x1, const vec3f& x2, const vec3f& x3,
	const vec3f& v0, const vec3f& v1, const vec3f& v2, const vec3f& v3,
	bool vf, REAL &ret)
{
	REAL a0 = stp(x1, x2, x3),
		a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
		a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
		a3 = stp(v1, v2, v3);

	REAL t[4];
	int nsol = solve_cubic(a3, a2, a1, a0, t);
	t[nsol] = 1; // also check at end of timestep
	for (int i = 0; i < nsol; i++) {
		if (t[i] < 0 || t[i] > 1)
			continue;

		ret = t[i];
		vec3f tx0 = xvpos(x0, v0, t[i]), tx1 = xvpos(x1 + x0, v1 + v0, t[i]),
			tx2 = xvpos(x2 + x0, v2 + v0, t[i]), tx3 = xvpos(x3 + x0, v3 + v0, t[i]);

		vec3f n;
		REAL w[4];
		REAL d;
		bool inside;
		if (vf) {
			d = signed_vf_distance(tx0, tx1, tx2, tx3, n, w);
			inside = (fmin(-w[1], fmin(-w[2], -w[3])) >= -1e-6);
		}
		else {// Impact::EE
			d = signed_ee_distance(tx0, tx1, tx2, tx3, n, w);
			inside = (fmin(fmin(w[0], w[1]), fmin(-w[2], -w[3])) >= -1e-6);
		}
		if (n.dot( w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
			n = -n;

		if (fabs(d) < 1e-6 && inside)
			return true;
	}
	return false;
}

bool testVF(
	const vec3f &x00, const vec3f &x10, const vec3f &x20, const vec3f &x30,
	const vec3f &x0, const vec3f &x1, const vec3f &x2, const vec3f &x3,
	REAL &rt)
{
	vec3f p0 = x00;
	vec3f p1 = x10 - x00;
	vec3f p2 = x20 - x00;
	vec3f p3 = x30 - x00;
	vec3f v0 = x0 - x00;
	vec3f v1 = x1 - x10 - v0;
	vec3f v2 = x2 - x20 - v0;
	vec3f v3 = x3 - x30 - v0;

	bool ret1 = dnfFilter(x00, x10, x20, x30, x0, x1, x2, x3);
	if (ret1 == false)
		return false;
	else
		return collisionTest(p0, p1, p2, p3, v0, v1, v2, v3, true, rt);
}


bool ccdVFtest(const vec3f& v0, const vec3f& v1, const vec3f& va, const vec3f& vb, const vec3f& vc, REAL& rt)
{
	return testVF(v0, va, vb, vc, v1, va, vb, vc, rt);
}
