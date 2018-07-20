#pragma once
#include "rectgrid_raytracing.cuh"
#include "solar_scene.h"

__global__ void map_tracing(const SunRay sunray,		// sun
	RectGrid grid,					// grid
	RectangleReceiver receiver,			// receiver
	const float3 *d_helio_vertexs,	// 3 vertexs of heliostats
	const float3 *d_microhelio_normals,	// micro-heliostat's normal
	const float3 *d_microhelio_center,	// micro-heliostat's origins
	const int *d_microhelio_groups,		// micro-heliostat's belonging group number
	const int microhelio_num);