#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "heap.cuh"

__device__ __host__ bool insert2(float *element_entry, int pos, float elem);

// Simple transformation kernel
__global__ void trimmed_mean(float* d_output,
	cudaTextureObject_t texObj,
	int kernel_radius, float trimmed_ratio,
	int width, int height);

class ImageSmoother 
{
public:
	static void image_smooth(float *d_array, //input as well as output
		int kernel_radius, float trimmed_ratio,
		int width, int height);
};