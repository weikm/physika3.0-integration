#pragma once

#include <float.h>
#include <stdlib.h>
#include "real.h"
#include "vec3f.h"
#include "box.h"

bool ccdVFtest(const vec3f& v0, const vec3f& v1, const vec3f& va, const vec3f& vb, const vec3f& vc, REAL& rt);