#pragma once
#include "solar_scene.h"
#include "rectgrid_raytracing.cuh"

void ray_tracing(const SunRay &sunray,				// sun
				Grid &grid,					// grid
				Receiver &receiver,					// receiver
				const float3 *d_helio_vertexs,		// 3 vertexs of heliostats
				const float3 *d_microhelio_normals,	// micro-heliostat's normal
				const float3 *d_microhelio_origs,	// micro-heliostat's origins
				const int *d_microhelio_groups,		// micro-heliostat's belonging group number
				const int &microhelio_num);		// micro-heliostat's belonging group number