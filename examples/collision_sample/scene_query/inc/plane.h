#pragma once

class cplane {
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
};
