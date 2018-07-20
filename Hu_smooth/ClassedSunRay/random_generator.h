#pragma once

#include<ctime>
#include<cstdlib>
#include <random>
#include "curand.h"
#include "utils.h"
#include "global_function.cuh"

class RandomGenerator
{	
public:
	//static unsigned int seed_;
	static curandGenerator_t *gen;
	//init the seed
	static void initSeed() {
		unsigned int mrand = (unsigned int)time(NULL);
		//seed_ = mrand;
		srand(mrand);
	};
	static void initSeed(unsigned int seed){
		//seed_ = seed;
		srand(seed);
	};
	
	// [0,1]
	static bool cpu_Uniform(float *h_0_1_array, const size_t &array_length);
	static bool gpu_Uniform(float *d_0_1_array, const size_t &array_length);

	static bool cpu_Gaussian(float *h_0_1_array, const float &mean, const float &stddev, const size_t &array_length);
	static bool gpu_Gaussian(float *d_0_1_array, const float &mean, const float &stddev, const size_t &array_length);

	// [low_threshold, high_threshold)
	static bool cpu_Uniform(int *h_min_max_array, const int &low_threshold, const int &high_threshold, const size_t &array_length);
	static bool gpu_Uniform(int *d_min_max_array, const int &low_threshold, const int &high_threshold, const size_t &array_length);
};