#include "recthelio_tracing.h"
#include <chrono>

int recthelio_ray_tracing_init(const RectangleHelio &recthelio,			//	which heliostat will be traced
	const Grid &grid,						//	the grid heliostat belongs to
	const vector<Heliostat *> heliostats,	//	all heliostats
	const int &num_group,					//	number of sun-ray
	size_t &microhelio_num,
	float3 *&d_microhelio_centers,
	float3 *&d_microhelio_normals,
	float3 *&d_helio_vertexs,
	int *&d_microhelio_groups)
{
	int num_subcenters = 0;
	switch (recthelio.type)
	{
		case(SubCenterType::Grid):
		{	
			set_microhelio_centers(recthelio, d_microhelio_centers, d_microhelio_normals, microhelio_num);
			num_subcenters = microhelio_num;
			break;
		}			
		case(SubCenterType::Poisson):
		{
			num_subcenters = set_possion_microhelio_centers(recthelio, d_microhelio_centers, d_microhelio_normals, microhelio_num);
			break;
		}
		default:
			break;
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int start_pos = grid.start_helio_pos_;
	int end_pos = start_pos + grid.num_helios_;
	//float3 *d_helio_vertexs = nullptr;
	set_helios_vertexes(heliostats, start_pos, end_pos, d_helio_vertexs);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//int *d_microhelio_groups = nullptr;
	set_microhelio_groups(d_microhelio_groups, num_group, microhelio_num);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	return num_subcenters;
}

int recthelio_ray_tracing(const SunRay &sunray,
	Receiver &receiver,
	const RectangleHelio &recthelio,			//	which heliostat will be traced
	Grid &grid,									//	the grid heliostat belongs to
	const vector<Heliostat *> heliostats)		//	all heliostats
{
	size_t microhelio_num;
	float3 *d_microhelio_centers = nullptr;
	float3 *d_microhelio_normals = nullptr;
	float3 *d_helio_vertexs = nullptr;
	int *d_microhelio_groups = nullptr;
	
	//	Init
	int num_subcenters=recthelio_ray_tracing_init(recthelio, grid, heliostats, 
		sunray.num_sunshape_groups_*sunray.num_sunshape_lights_per_group_,
		microhelio_num, d_microhelio_centers, d_microhelio_normals, 
		d_helio_vertexs, d_microhelio_groups);

	//	ray_tracing
	ray_tracing(sunray, grid, receiver,
		d_helio_vertexs, d_microhelio_normals, d_microhelio_centers, d_microhelio_groups, microhelio_num);

	//	Clean all
	cudaFree(d_microhelio_centers);
	cudaFree(d_microhelio_normals);
	cudaFree(d_helio_vertexs);
	cudaFree(d_microhelio_groups);

	d_microhelio_centers = nullptr;
	d_microhelio_normals = nullptr;
	d_helio_vertexs = nullptr;
	d_microhelio_groups = nullptr;

	return num_subcenters;
}