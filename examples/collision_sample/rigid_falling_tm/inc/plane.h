#pragma once

class alignas(16) cplane {
private:
	vec3f _normal;
	REAL _d;

public:
	cplane(vec3f& n, REAL d)
	{
		_normal = n;
		_d = d;
		_normal.normalize();
	}

	const vec3f& n() const {
		return _normal;
	}

	REAL d() const {
		return _d;
	}

	REAL distance(vec3f& p) {
		return p.dot(_normal) - _d;
	}
};
