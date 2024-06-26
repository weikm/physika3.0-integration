#pragma once
#define M_PI_2 1.57079632679489661923
// island management, _activationState1
#define ACTIVE_TAG 1
#define ISLAND_SLEEPING 2
#define WANTS_DEACTIVATION 3
#define DISABLE_DEACTIVATION 4
#define DISABLE_SIMULATION 5
#define ANGULAR_MOTION_THRESHOLD float(0.5) * M_PI_2