#include "scene_normals.h"

float3 *load_noraml()
{
	int N;
	fstream fin(solarenergy::normal_filepath);
	fin >> N;

	float3 *normals = new float3[N];
	for (int i = 0; i < N; ++i)
		fin >> normals[i].x >> normals[i].y >> normals[i].z;

	return normals;
}