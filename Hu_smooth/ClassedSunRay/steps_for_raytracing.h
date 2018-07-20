#pragma once
#include "solar_scene.h"
#include "compact_sample.cuh"
#include "poisson_disk.cuh"

// float3 *d_microhelio_centers
// float3 *d_microhelio_normals
// microhelio_num
bool set_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers, float3 *&d_microhelio_normals, size_t &size);
int set_possion_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers,
									float3 *&d_microhelio_normals, size_t &size,
									int k=8);

// float3 *d_helio_vertexs
//	- start_pos:	start position of heliostats array
//	- end_pos:		the position after the end of heliostats array
bool set_helios_vertexes(vector<Heliostat *> heliostats, const int start_pos, const int end_pos,
							float3 *&d_helio_vertexs);

// int *d_microhelio_groups
bool set_microhelio_groups(int *&d_microhelio_groups, const int num_group, const size_t &size);

//#include "image_save.h"
//namespace tmp2
//{
//	inline void save_array(string filename, float2 *array, int size)
//	{
//		ofstream out(filename.c_str());
//		for (int i = 0; i < size; ++i)
//			out << array[i].x<<'\t'<<array[i].y << endl;
//		out.close();
//	}
//};