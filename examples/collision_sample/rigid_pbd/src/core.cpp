#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "core.h"

#define DYNAMIC_ARRAY_IMPLEMENT
#define GRAPHICS_MATH_IMPLEMENT
#define C_FEK_HASH_MAP_IMPLEMENT
#include <light_array.h>
#include <hash_map.h>
#include <gm.h>

#include "spot_storm.h"

void initModel(const char*, const char*)
{
	ex_spot_storm_init();
}

void quitModel()
{
	ex_spot_storm_destroy();
}
void drawModel(bool, bool, bool, bool, int)
{
	ex_spot_storm_render();
}

bool dynamicModel(char*, bool, bool)
{
	double	delta_time = 1.0 / 60.0;
	ex_spot_storm_update(delta_time);
	return true;
}
