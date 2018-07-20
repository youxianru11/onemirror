#pragma once

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "random_generator.h"

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ void cannot_place(cudaTextureObject_t &texObj, bool &cannot_place,
	const float2 &to_put_pos, const float r,
	int x, int y,
	int width, int height,
	float low_lmt);

__global__ void poisson_point_1st_phase(float2* output,
	float *random_values,
	float pixel_length,
	int phase_length,
	int x_offset, int y_offset,
	int width, int height);

__global__ void poisson_point_other_phase(float2* output,
	cudaTextureObject_t texObj,
	float *random_values,
	float pixel_length, float r,
	int phase_length,
	int x_offset, int y_offset,
	int k,
	int width, int height);

// input:
//	1. width and height
//  2. r: least distance between any two points in domain
//	3. k: how many points will be iterated during finding a point in a given cell
//	4. dimension: 1d, 2d, 3d, ...
float2 *poisson_sample(const float width, const float height, const float r, const int k,
	const int dimension,
	int &width_int, int &height_int);