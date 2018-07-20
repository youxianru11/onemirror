#include "random_generator.h"

__global__ void map_float2int(int *d_iData, const float const *d_fData,
	const int low_threshold, const int high_threshold, const size_t size)
{
	unsigned int myId = global_func::getThreadId();
	if (myId >= size)
		return;
	
	d_iData[myId] = floorf(d_fData[myId] * (high_threshold - low_threshold) + low_threshold);
}

bool RandomGenerator::gpu_Uniform(int *d_min_max_array, const int &low_threshold, const int &high_threshold, const size_t &array_length)
{
	if (d_min_max_array == nullptr)
		return false;

	//curandGenerator_t gen;
	//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

	float *d_uniform = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_uniform, array_length * sizeof(float)));
	curandGenerateUniform(*gen, d_uniform, array_length);

	int nThreads;
	dim3 nBlocks;
	if (!global_func::setThreadsBlocks(nBlocks, nThreads, array_length))
		return false;
	
	map_float2int << <nBlocks, nThreads >> > (d_min_max_array, d_uniform, low_threshold,high_threshold, array_length);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	/* Cleanup */
	/*curandDestroyGenerator(gen);*/
	checkCudaErrors(cudaFree(d_uniform));
	return true;
}