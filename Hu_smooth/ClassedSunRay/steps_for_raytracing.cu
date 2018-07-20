#include "steps_for_raytracing.h"

// Step 1: Generate local micro-heliostats' centers
__global__ void map_microhelio_centers(float3 *d_microhelio_centers, float3 helio_size,
	const int2 row_col, const int2 sub_row_col,
	const float2 gap,
	const float pixel_length, const float2 subhelio_rowlength_collength, const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	int row = myId / (row_col.y*sub_row_col.y);
	int col = myId % (row_col.y*sub_row_col.y);

	int block_row = row / sub_row_col.x;
	int block_col = col / sub_row_col.y;

	d_microhelio_centers[myId].x = col*pixel_length + block_col*gap.x + pixel_length / 2 - helio_size.x / 2;
	d_microhelio_centers[myId].y = helio_size.y / 2;
	d_microhelio_centers[myId].z = row*pixel_length + block_row*gap.y + pixel_length / 2 - helio_size.z / 2;
}

__global__ void map_microhelio_centers(float3 *d_microhelio_centers, float2 *d_sub_origin, 
	float3 helio_size,
	const float2 gap,const float2 subhelio_rowlength_collength, const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	int block_row = d_sub_origin[myId].y / subhelio_rowlength_collength.x;
	int block_col = d_sub_origin[myId].x / subhelio_rowlength_collength.y;

	d_microhelio_centers[myId].x = d_sub_origin[myId].x + block_col*gap.x - helio_size.x / 2;
	d_microhelio_centers[myId].y = helio_size.y / 2;
	d_microhelio_centers[myId].z = d_sub_origin[myId].y + block_row*gap.y - helio_size.z / 2;
}

// Step 2: Generate micro-heliostats' normals
__global__ void map_microhelio_normals(float3 *d_microhelio_normals, const float3 *d_microhelio_centers,
	float3 normal,
	const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	d_microhelio_normals[myId] = normal;
}

// Step 3: Transform local micro-helio center to world postion
__global__ void map_microhelio_center2world(float3 *d_microhelio_world_centers, float3 *d_microhelio_local_centers,
	const float3 normal, const float3 world_pos,
	const size_t size)
{
	unsigned long long int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	float3 local = d_microhelio_local_centers[myId];
	local = global_func::local2world(local, normal);		// Then Rotate
	local = global_func::transform(local, world_pos);		// Translation to the world system
	d_microhelio_world_centers[myId] = local;
}

bool set_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers, float3 *&d_microhelio_normals, size_t &size)
{
	int2 row_col = recthelio.row_col_;
	float3 helio_size = recthelio.size_;
	float2 gap = recthelio.gap_;
	float pixel_length = recthelio.pixel_length_;
	
	float2 subhelio_rowlength_collength;
	subhelio_rowlength_collength.x = (helio_size.z - gap.y*(row_col.x - 1)) / float(row_col.x);
	subhelio_rowlength_collength.y = (helio_size.x - gap.x*(row_col.y - 1)) / float(row_col.y);

	int2 sub_row_col;
	sub_row_col.x = subhelio_rowlength_collength.x / pixel_length;
	sub_row_col.y = subhelio_rowlength_collength.y / pixel_length;

	size = sub_row_col.x*sub_row_col.y*row_col.x*row_col.y;

	int nThreads;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, size);

	// 1. local center position
	if (d_microhelio_centers == nullptr)
		cudaMalloc((void **)&d_microhelio_centers, sizeof(float3)*size);
	map_microhelio_centers << <nBlocks, nThreads >> >
		(d_microhelio_centers, helio_size, row_col, sub_row_col, gap, pixel_length, subhelio_rowlength_collength, size);

	// 2. normal
	if (d_microhelio_normals == nullptr)
		cudaMalloc((void **)&d_microhelio_normals, sizeof(float3)*size);
	map_microhelio_normals <<<nBlocks, nThreads >>>(d_microhelio_normals, d_microhelio_centers, recthelio.normal_, size);

	// 3. world center position
	map_microhelio_center2world <<<nBlocks, nThreads >>>(d_microhelio_centers, d_microhelio_centers, recthelio.normal_, recthelio.pos_, size);

	return true;
}

int set_possion_microhelio_centers(const RectangleHelio &recthelio, float3 *&d_microhelio_centers, 
	float3 *&d_microhelio_normals, size_t &size,
	int k)
{
	int2 row_col = recthelio.row_col_;
	float3 helio_size = recthelio.size_;
	float2 gap = recthelio.gap_;
	float pixel_length = recthelio.pixel_length_;

	float2 subhelio_rowlength_collength;
	subhelio_rowlength_collength.x = (helio_size.z - gap.y*(row_col.x - 1)) / float(row_col.x);
	subhelio_rowlength_collength.y = (helio_size.x - gap.x*(row_col.y - 1)) / float(row_col.y);
	
	//	1. local center position
	//		1.1 generate width_int*height_int poisson samples(could be empty in some ceil)
	float width = subhelio_rowlength_collength.y*row_col.y;
	float height = subhelio_rowlength_collength.x*row_col.x;
	float r = pixel_length/sqrtf(1.5);
	int dimension = 2;
	int width_int, height_int;
	float2 *d_output = poisson_sample(width, height, r, k, dimension, width_int, height_int);

	//		1.2 compact it and save the result into sub_orign, return its length
	float2 *d_sub_origin = nullptr;
	size = generate_microhelio_origin(d_output, d_sub_origin,
		width, height, width_int, height_int);
	//float2 *h_result = new float2[size];
	//cudaMemcpy(h_result, d_sub_origin, size * sizeof(float2), cudaMemcpyDeviceToHost);
	//tmp2::save_array("../result/huge_gap_center.txt", h_result, size);

	//		1.3 map 2d -> 3d considering the gaps
	int nThreads;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, size);
	if (d_microhelio_centers == nullptr)
		cudaMalloc((void **)&d_microhelio_centers, sizeof(float3)*size); 
	map_microhelio_centers << <nBlocks, nThreads >> >(d_microhelio_centers, d_sub_origin,
		helio_size,
		 gap, subhelio_rowlength_collength, size);

	//	1.4 clear d_output and d_sub_origin
	cudaFree(d_output);
	cudaFree(d_sub_origin);

	// 2. normal
	if (d_microhelio_normals == nullptr)
		cudaMalloc((void **)&d_microhelio_normals, sizeof(float3)*size);
	map_microhelio_normals << <nBlocks, nThreads >> >(d_microhelio_normals, d_microhelio_centers, recthelio.normal_, size);

	// 3. world center position
	map_microhelio_center2world << <nBlocks, nThreads >> >(d_microhelio_centers, d_microhelio_centers, recthelio.normal_, recthelio.pos_, size);

	return size;
}

// const float3 *d_helio_vertexs
bool set_helios_vertexes(vector<Heliostat *> heliostats, const int start_pos, const int end_pos,
	float3 *&d_helio_vertexs)
{
	int size = end_pos-start_pos;
	float3 *h_helio_vertexes = new float3[size * 3];

	for (int i = start_pos; i < end_pos; ++i)
	{
		int j = i - start_pos;
		heliostats[i]->Cget_vertex(h_helio_vertexes[3 * j], h_helio_vertexes[3 * j + 1], h_helio_vertexes[3 * j + 2]);
	}
		
	
	global_func::cpu2gpu(d_helio_vertexs, h_helio_vertexes,  3 * size);

	delete[] h_helio_vertexes;
	return true;
}

// int *d_microhelio_groups
bool set_microhelio_groups(int *&d_microhelio_groups, const int num_group, const size_t &size)
{
	if (d_microhelio_groups == nullptr)
		checkCudaErrors(cudaMalloc((void **)&d_microhelio_groups, sizeof(int)*size));

	RandomGenerator::gpu_Uniform(d_microhelio_groups, 0, num_group, size);
	return true;
}