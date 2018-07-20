#pragma once
#include "solar_scene.h"
#include "random_generator.h"

class SceneProcessor
{
public:
	//grid
	static void set_grid_content(vector<Grid *> &grids, const vector<Heliostat *> &heliostats);

	// receiver
	static void set_receiver_content(vector<Receiver *> &receivers);

	// helio
	static void set_helio_content(vector<Heliostat *> &heliostats, const float3 &focus_center, const float3 &sunray_dir);
	static void set_helio_content(vector<Heliostat *> &heliostats, const float3 *normals);
	// focus_centers is the head of array
	static bool set_helio_content(vector<Heliostat *> &heliostats, const float3 *focus_centers, const float3 &sunray_dir, const size_t &size);

	// sunray
	static void set_sunray_content(SunRay &sunray);

private:
	static void set_perturbation(SunRay &sunray);
	static void set_samplelights(SunRay &sunray);
};
