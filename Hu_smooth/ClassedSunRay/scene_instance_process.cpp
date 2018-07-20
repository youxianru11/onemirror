#include "scene_instance_process.h"

//grid
void SceneProcessor::set_grid_content(vector<Grid *> &grids, const vector<Heliostat *> &heliostats)
{
	for (int i = 0; i < grids.size(); ++i)
	{
		grids[i]->Cinit();
		grids[i]->CGridHelioMatch(heliostats);
	}
}

// receiver
void SceneProcessor::set_receiver_content(vector<Receiver *> &receivers)
{
	for (int i = 0; i < receivers.size(); ++i)
		receivers[i]->CInit(int(1.0f / solarenergy::receiver_pixel_length));
}

// helio
void SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const float3 &focus_center, const float3 &sunray_dir)
{
	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(focus_center, sunray_dir);
	}
}

void SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const float3 *normals)
{
	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(normals[i]);
	}
}

bool SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const float3 *focus_centers, const float3 &sunray_dir, const size_t &size)
{
	if (heliostats.size() != size)
		return false;

	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(focus_centers[i], sunray_dir);
	}
	return true;
}
