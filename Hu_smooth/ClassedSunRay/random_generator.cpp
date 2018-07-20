#include "random_generator.h"

bool RandomGenerator::cpu_Uniform(float *h_0_1_array, const size_t &array_length)
{
	if (h_0_1_array == nullptr)
		return false;
	//initSeed();

	for(int i=0;i<array_length;++i)
		h_0_1_array[i] = (float)((float)rand() / (RAND_MAX));
	return true;
}

bool RandomGenerator::gpu_Uniform(float *d_0_1_array, const size_t &array_length)
{
	if (d_0_1_array == nullptr)		// d_0_1_array==nullptr
		return false;

	/* Generate */
	//curandGenerator_t gen;
	//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
		
	/* Generate 0-1 array */
	curandGenerateUniform(*gen, d_0_1_array, array_length);

	/* Cleanup */
	//curandDestroyGenerator(gen);
	return true;
}

bool RandomGenerator::cpu_Gaussian(float *h_0_1_array, const float &mean, const float &stddev, const size_t &array_length)
{
	if (h_0_1_array == nullptr)
		return false;
	initSeed();
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stddev);

	for(int i=0;i<array_length;++i)
		h_0_1_array[i]= distribution(generator);

	return true;
}

bool RandomGenerator::gpu_Gaussian(float *d_0_1_array, const float &mean, const float &stddev, const size_t &array_length)
{
	if (d_0_1_array == nullptr)
		return false;
	/* Generate */
	//curandGenerator_t gen;
	//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

	/* Generate 0-1 array */
	curandGenerateNormal(*gen, d_0_1_array, array_length, mean, stddev);

	/* Cleanup */
	//curandDestroyGenerator(gen);
	return true;
}

bool RandomGenerator::cpu_Uniform(int *h_min_max_array, const int &low_threshold, const int &high_threshold, const size_t &array_length)
{
	if (h_min_max_array == nullptr)
		return false;

	int range = high_threshold - low_threshold;
	for (int i = 0; i < array_length; ++i)	
		h_min_max_array[i] = floor(float(rand()) / float(RAND_MAX)*range + low_threshold);
	
	return true;
}
