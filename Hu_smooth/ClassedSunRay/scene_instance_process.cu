#include "scene_instance_process.h"
#include <iostream>
#include <fstream>

// sunray
void SceneProcessor::set_sunray_content(SunRay &sunray)
{
	sunray.sun_dir_ = normalize(sunray.sun_dir_);
	set_perturbation(sunray);
	set_samplelights(sunray);
}

// perturbation
__global__ void map_turbulance(float3 *d_turbulance, const float *d_guassian, const float *d_uniform, const size_t size)
{	
	unsigned int myId = global_func::getThreadId();
	if (myId >= size)
		return;

	float theta = d_guassian[myId], phi = d_uniform[myId] * 2 * MATH_PI;
	float3 dir = global_func::angle2xyz(make_float2(theta, phi));
	d_turbulance[myId] = dir;
}

void SceneProcessor::set_perturbation(SunRay &sunray)
{
	int size = sunray.num_sunshape_groups_* sunray.num_sunshape_lights_per_group_;
	//	Step 1:	Allocate memory for sunray.d_perturbation_ on GPU
	if (sunray.d_perturbation_ == nullptr)
		checkCudaErrors(cudaMalloc((void **)&sunray.d_perturbation_, sizeof(float3)*size));

	//	Step 2:	Allocate memory for theta and phi
	float *d_guassian_theta = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_guassian_theta, sizeof(float)*size));
	float *d_uniform_phi = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_uniform_phi, sizeof(float)*size));

	//	Step 3:	Generate theta and phi
	RandomGenerator::gpu_Gaussian(d_guassian_theta, 0.0f, solarenergy::disturb_std, size);
	RandomGenerator::gpu_Uniform(d_uniform_phi, size);

	//	Step 4:	(theta, phi) -> ( x, y, z)
	int nThreads;
	dim3 nBlocks;
	global_func::setThreadsBlocks(nBlocks, nThreads, size);
	map_turbulance << <nBlocks, nThreads >> > (sunray.d_perturbation_, d_guassian_theta, d_uniform_phi, size);

	//	Step 5: Cleanup
	checkCudaErrors(cudaFree(d_guassian_theta));
	checkCudaErrors(cudaFree(d_uniform_phi));
}

// sample lights
namespace samplelights
{
	float sunshape_intensity(float theta, const float &k, const float &gamma)
	{
		// theta must be in range [0,4.65)
		return cosf(0.326*theta) / cosf(0.308*theta);
	}

	float inter_larger_465_intensity(float theta, const float &k, const float &gamma)
	{
		// theta must be in range [4.65, 9.3]
		return expf(k) / (gamma + 1)*powf(theta, gamma + 1);
	}

	// Description:
	//	- h_k:(return value) slope (length_interval / subarea)
	//	- num_group: num of intervals
	//	- k, gamma: used in intensity calculation
	//	- upper_lm: length_interval = upper_lm / num_group
	float2* parameters_generate(float *h_k, int num_group, float k, float gamma, float upper_lm)
	{
		float length_interval = upper_lm / float(num_group);
		float x = 0.0f;
		float2 *h_cdf = new float2[num_group + 1];
		h_cdf[0].x = 0.0f;
		h_cdf[0].y = 0.0f;
		float hist_pre = sunshape_intensity(x, k, gamma);
		float hist_current;
		float subarea;
		for (int i = 1; i <= num_group; ++i)
		{
			x += length_interval;
			hist_current = sunshape_intensity(x, k, gamma);
			subarea = (hist_pre + hist_current)*length_interval / 2.0f;

			h_cdf[i].x = x;
			h_cdf[i].y = subarea + h_cdf[i - 1].y;
			h_k[i - 1] = length_interval / subarea;

			hist_pre = hist_current;
		}
		return h_cdf;
	}

	__host__ __device__ int max_less_index(float2 *d_cdf, float value, size_t n)// d_cdf[n+1]
	{
		int left = 0, right = n;
		int mid;
		while (left <= right)
		{
			mid = (left + right) >> 1;
			if (value > d_cdf[mid].y)
				left = mid + 1;
			else if (value < d_cdf[mid].y)
				right = mid - 1;
			else //value==d_cdf[mid].y
				return mid;
		}
		return right;
	}

	__global__ void linear_interpolate(float *d_0_1, float2 *d_cdf,
		float *d_k,
		float int_less_465,
		float gamma, float k,
		float A, float B, float C,
		size_t n,// d_cdf[n+1]
		size_t size)
	{
		const int myId = global_func::getThreadId();
		if (myId >= size)
			return;

		float u = d_0_1[myId] * A;

		if (u < int_less_465)
		{
			int id = max_less_index(d_cdf, u, n);
			d_0_1[myId] = (u - d_cdf[id].y)*d_k[id] + d_cdf[id].x;
		}
		else
			d_0_1[myId] = powf((u - int_less_465)*B + C, 1 / (gamma + 1));
		return;
	}

	__global__ void map_samplelights(float3 *d_samplelights, float *d_theta, float *d_phi_0_1, size_t size)
	{
		unsigned int myId = global_func::getThreadId();
		if (myId >= size)
			return;

		float theta = d_theta[myId]/1000, phi = d_phi_0_1[myId] * 2 * MATH_PI;
		float3 dir = global_func::angle2xyz(make_float2(theta, phi));
		d_samplelights[myId] = dir;
	}
}

// delete it when doesn't need
namespace tmp2
{
	void save_array(string filename, float *array, int size)
	{
		ofstream out(filename.c_str());
		for (int i = 0; i < size; ++i)
			out << array[i] << endl;
		out.close();
	}
};

void SceneProcessor::set_samplelights(SunRay &sunray)
{
	// input parameters
	int num_group = 1024;
	float csr = sunray.csr_;
	float upper_lm = 4.65f;

	// 
	float k = 0.9f*logf(13.5f*csr)*powf(csr, -0.3f);
	float gamma = 2.2f*logf(0.52f*csr)*powf(csr, 0.43f) - 0.1f;
	float value_465_930 = samplelights::inter_larger_465_intensity(9.3f, k, gamma) - samplelights::inter_larger_465_intensity(upper_lm, k, gamma);

	float *h_k = new float[num_group];
	float2 *h_cdf = samplelights::parameters_generate(h_k, num_group, k, gamma, upper_lm);
	float value_less_465 = h_cdf[num_group].y;

	float2 *d_cdf = nullptr;
	cudaMalloc((void **)&d_cdf, sizeof(float2)*(num_group + 1));
	cudaMemcpy(d_cdf, h_cdf, sizeof(float2)*(num_group + 1), cudaMemcpyHostToDevice);
	float *d_k = nullptr;
	cudaMalloc((void **)&d_k, sizeof(float)*num_group);
	cudaMemcpy(d_k, h_k, sizeof(float)*num_group, cudaMemcpyHostToDevice);

	// Generate uniform random theta and phi in range [0,1]
	float *d_theta = nullptr;
	int num_random = sunray.num_sunshape_groups_ * sunray.num_sunshape_lights_per_group_;
	cudaMalloc((void **)&d_theta, sizeof(float)*num_random);
	RandomGenerator::gpu_Uniform(d_theta, num_random);

	float *d_phi = nullptr;
	cudaMalloc((void **)&d_phi, sizeof(float)*num_random);
	RandomGenerator::gpu_Uniform(d_phi, num_random);
		
	int nThreads = 1024;
	int nBlock = (num_random + nThreads - 1) / nThreads;
	float A = value_less_465 + value_465_930;
	float B = (gamma + 1) / expf(k);
	float C = powf(4.65, gamma + 1);

	// Change to correct theta
	samplelights::linear_interpolate << <nBlock, nThreads >> >(d_theta, d_cdf, d_k, value_less_465, gamma, k,
		A, B, C,
		num_group, num_random);

	// 
	if (sunray.d_samplelights_ == nullptr)
		cudaMalloc((void **)&sunray.d_samplelights_, sizeof(float3)*num_random);
	samplelights::map_samplelights <<<nBlock, nThreads >>>(sunray.d_samplelights_,d_theta,d_phi,num_random);

	// clear
	delete[] h_k;
	delete[] h_cdf;
	h_k = nullptr;
	h_cdf = nullptr;

	cudaFree(d_theta);
	cudaFree(d_phi);
	cudaFree(d_cdf);
	cudaFree(d_k);
	d_theta = nullptr;
	d_phi = nullptr;
	d_cdf = nullptr;
	d_k = nullptr;
}

float sunshape_normalizedintensity(const float &theta, const float &k, const float &gamma)
{
	if (theta > 0 && theta <= 4.65)
		return cosf(0.326*theta) / cosf(0.308*theta);
	else if (theta > 4.65)
		return exp(k)*pow(theta, gamma);

	return 0.0;
}

//void SceneProcessor::set_samplelights(SunRay &sunray)
//{
//	int num_all_lights = sunray.num_sunshape_groups_ * sunray.num_sunshape_lights_per_group_;
//	float CSR = sunray.csr_;
//	float k = 0.9*logf(13.5*CSR)*powf(CSR, -0.3);
//	float gamma = 2.2*logf(0.52*CSR)*powf(CSR, 0.43) - 0.1;
//
//	//	Step 1:	Allocate memory temporarily used as h_samplelights, h_tmp_theta and h_tmp_phi on CPU
//	float3 *h_samplelights = new float3[num_all_lights];
//
//	//	Step 2:
//	float x, y;
//	float theta, phi;
//	float xmin = 0.0f, xmax = 9.3f;
//	int accept = 0, count = 0;
//	while (accept<num_all_lights)
//	{
//		x = (float)((float)rand() / (RAND_MAX))*(xmax - xmin) + xmin;
//		y = (float)((float)rand() / (RAND_MAX));
//
//		if (y <= sunshape_normalizedintensity(x, k, gamma))
//		{
//			theta = x / 1000;
//			phi = (float)((float)rand() / (RAND_MAX)) * 2 * MATH_PI;
//			h_samplelights[accept] = global_func::angle2xyz(make_float2(theta, phi));
//			++accept;
//		}
//		++count;
//	}
//
//	// Step 3 : Transfer h_samplelights to sunray.d_samplelights_
//	global_func::cpu2gpu(sunray.d_samplelights_, h_samplelights, num_all_lights);
//
//	//	Step 4 : Cleanup
//	delete[] h_samplelights;
//	h_samplelights = nullptr;
//}


//void SceneProcessor::set_samplelights(SunRay &sunray)
//{
//	int num_all_lights = sunray.num_sunshape_groups_ * sunray.num_sunshape_lights_per_group_;
//	float3 *h_angle = new float3[num_all_lights];
//	for (int i = 0; i < num_all_lights; ++i)
//	{
//		h_angle[i].x = 0;
//		h_angle[i].y = 1;
//		h_angle[i].z = 0;
//	}
//	global_func::cpu2gpu(sunray.d_samplelights_, h_angle, num_all_lights);
//
//	delete[] h_angle;
//	h_angle = nullptr;
//}