#include "poisson_disk.cuh"
#include "vector_arithmetic.cuh"

__global__ void poisson_point_1st_phase(float2* d_output,
	float *random_values,
	float pixel_length,
	int phase_length,
	int x_offset, int y_offset,
	int width, int height)
{
	// Calculate normalized texture coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int my_indx = y * width / phase_length + x;
	x = x * phase_length + x_offset;
	y = y * phase_length + y_offset;
	if (x > width - 1 || y > height - 1)// out of range
		return;

	d_output[y * width + x].x = (random_values[2 * my_indx] + x - 1)*pixel_length;		// -1 for transform to left a step
	d_output[y * width + x].y = (random_values[2 * my_indx + 1] + y -1)*pixel_length;	// -1 for transform to down a step
	return;
}

__device__ void cannot_place(cudaTextureObject_t &texObj, bool &cannot_place,
	const float2 &to_put_pos, const float radius,
	int x, int y, 
	int width, int height,
	float low_lmt)
{
	float2 value = tex2D<float2>(texObj, float(x) + 0.5, float(y) + 0.5);
	if (x<0 || x>width - 1 ||
		y<0 || y>height - 1 || // out of boundary
		value.x<low_lmt)// empty
		return;

	if (length(value - to_put_pos) < radius)
		cannot_place = true;
	return;
}

__global__ void poisson_point_other_phase(float2* d_output,
	cudaTextureObject_t texObj,
	float *random_values,
	float pixel_length, float radius,
	int phase_length,
	int x_offset, int y_offset,
	int k,
	int width, int height)
{
	// Calculate normalized texture coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int my_indx = y*width / phase_length + x;
	x = x*phase_length + x_offset;
	y = y*phase_length + y_offset;
	if (x > width - 1 || y > height - 1)// out of range
		return;

	float2 to_be_placed_loc;
	bool cannot_place_point;
	for (int i = 0; i < k; ++i)
	{
		cannot_place_point = false;
		to_be_placed_loc = make_float2((random_values[2 * (my_indx*k + i)] + x -1)*pixel_length,
			(random_values[2 * (my_indx*k + i) + 1] + y -1)*pixel_length);

		int2 col_lmt = make_int2((to_be_placed_loc.x - radius) / pixel_length + 1, (to_be_placed_loc.x + radius) / pixel_length + 1 );
		int2 row_lmt = make_int2((to_be_placed_loc.y - radius) / pixel_length + 1, (to_be_placed_loc.y + radius) / pixel_length + 1);
		
		for (int r = row_lmt.x; r<=row_lmt.y; ++r)
			for (int c = col_lmt.x; c <= col_lmt.y; ++c)
			{
				cannot_place(texObj,
					cannot_place_point,
					to_be_placed_loc, radius,
					c, r,
					width, height,
					-pixel_length);
			}
		if (!cannot_place_point)
		{
			d_output[y * width + x] = to_be_placed_loc;
			return;
		}
	}

	return;
}

// Step 1: 
//	generate width_int*height_int poisson samples(could be empty in some ceil)
float2 *poisson_sample(const float width, const float height, const float r, const int k,
	const int dimension,
	int &width_int, int &height_int)
{
	float pixel_length = r / sqrtf(dimension);
	width_int = ceil(width / pixel_length) + 2;	// +2 represent add padding around
	height_int = ceil(height / pixel_length) + 2; // ...
	int phase_length = ceil(sqrtf(dimension)) + 1;
	int phase_width_int = iDivUp(width_int, phase_length);
	int phase_height_int = iDivUp(height_int, phase_length);

	// 1.	Create textrue object
	//		1)	Allocate CUDA array in device memory
	//		2)	Specify texture
	//		3)	Specify texture object parameters
	//		4)	Create texture object
	//	1)	Allocate CUDA array in device memory
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float2>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &floatTex, width_int, height_int);

	//	2)	Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	//	3)	Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;

	//	4)	Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	// 2.	Allocate result of transformation in device memory	
	float2 *h_data = new float2[width_int*height_int];
	int size = width_int*height_int;
	for (int i = 0; i < size; ++i)
		h_data[i] = make_float2(-2 * pixel_length, -2 * pixel_length);

	float2* d_output = nullptr;
	cudaMalloc((void **)&d_output, size * sizeof(float2));
	cudaMemcpy(d_output, h_data, size * sizeof(float2), cudaMemcpyHostToDevice);

	// 7.	Generate random variable
	//		1)	Generate [(phase_length*phase_length-1)*k+1]*num_random float2 random variables
	//			for potientail poisson location
	//		2)	Generate shuffle permutation for phase iteration
	//	1)
	int num_random = phase_width_int*phase_height_int*((phase_length*phase_length - 1)*k + 1);
	float *d_x_y = nullptr;
	cudaMalloc((void**)&d_x_y, sizeof(float)*num_random * 2);
	RandomGenerator::gpu_Uniform(d_x_y, num_random * 2);
	//	2)
	int *tmp = new int[phase_length*phase_length];
	int *permuation = new int[phase_length*phase_length * 2];
	for (int i = 0; i < phase_length*phase_length; ++i)
		tmp[i] = i;
	//random_shuffle(tmp, tmp + phase_length*phase_length);
	for (int i = 0; i < phase_length*phase_length; ++i)
	{
		permuation[2 * i] = tmp[i] % phase_length;
		permuation[2 * i + 1] = tmp[i] / phase_length;
	}

	dim3 dimBlock(32, 8);
	dim3 dimGrid((phase_width_int + dimBlock.x - 1) / dimBlock.x,
		(phase_height_int + dimBlock.y - 1) / dimBlock.y);

	// 8. Generate poisson sample
	poisson_point_1st_phase << <dimGrid, dimBlock >> >(d_output,
		d_x_y,
		pixel_length,
		phase_length,
		permuation[0], permuation[1],
		width_int, height_int);

	float *d_random_values = d_x_y + 2 * phase_width_int*phase_height_int;
	for (int i = 1; i < phase_length*phase_length; ++i)
	{
		cudaMemcpyToArray(cuArray, 0, 0, d_output, size * sizeof(float2), cudaMemcpyDeviceToDevice);
		poisson_point_other_phase << <dimGrid, dimBlock >> >(d_output,
			texObj,
			d_random_values + 2 * (i - 1)*phase_width_int*phase_height_int*k,
			pixel_length, r,
			phase_length,
			permuation[2 * i], permuation[2 * i + 1],
			k,
			width_int, height_int);
	}

	// Clear
	delete[] h_data;
	h_data = nullptr;
	delete[] permuation;
	permuation = nullptr;
	delete[] tmp;
	tmp = nullptr;

	cudaFreeArray(cuArray);
	cudaFree(d_x_y);
	d_x_y = nullptr;
	return d_output;
}