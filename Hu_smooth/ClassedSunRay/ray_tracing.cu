#include "ray_tracing.h"

void ray_tracing(const SunRay &sunray,		// sun
	Grid &grid,								// grid
	Receiver &receiver,						// receiver
	const float3 *d_helio_vertexs,			// 3 vertexs of heliostats
	const float3 *d_microhelio_normals,		// micro-heliostat's normal
	const float3 *d_microhelio_origs,		// micro-heliostat's origins
	const int *d_microhelio_groups,			// micro-heliostat's belonging group number
	const int &microhelio_num)
{
	int nThreads = 256;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, microhelio_num*sunray.num_sunshape_lights_per_group_, true);

	//	tracing every single light
	switch (grid.type_) 
	{
	case 0: 
	{
		RectGrid *rectgrid = dynamic_cast<RectGrid *> (&grid);
		RectangleReceiver *rect_receiver = dynamic_cast<RectangleReceiver *> (&receiver);
		map_tracing << <nBlocks, nThreads >> >(sunray, *rectgrid, *rect_receiver,
			d_helio_vertexs, d_microhelio_normals, d_microhelio_origs, d_microhelio_groups,
			microhelio_num);
		auto a = cudaGetLastError();
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		break;
	}
	default:
		break;
	}
}