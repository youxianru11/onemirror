#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "vector_arithmetic.cuh"
#include "global_constant.h"
#include "utils.h"

namespace global_func
{
	//	Transform the local coordinate to world coordinate
	__host__ __device__ inline float3 local2world(const float3 &d_local, const float3 &aligned_normal)
	{
		// u : X
		// n is normal: Y
		// v : Z	
		// Sample(world coordinate)= Sample(local) * Transform matrix
		// Transform matrix:
		// |u.x		u.y		u.z|	
		// |n.x		n.y		n.z|	
		// |v.x		v.y		v.z|	

		float3 u, n, v;// could be shared

		n = aligned_normal;

		if (abs(n.x)<Epsilon&&abs(n.z)<Epsilon)
			return d_local; //	parallel to (0,1,0), don't need to transform

		//if (abs(n.x) > abs(n.z))
		//{
		//	v = cross(n, make_float3(0.0f, 1.0f, 0.0f));
		//	v = normalize(v);
		//	u = cross(n, v);
		//	u = normalize(u);
		//}
		//else
		//{
			u = cross(make_float3(0.0f, 1.0f, 0.0f), n);
			u = normalize(u);
			v = cross(u, n);
			v = normalize(v);
		//}

		float3 d_world = make_float3(d_local.x*u.x + d_local.y*n.x + d_local.z*v.x,
			d_local.x*u.y + d_local.y*n.y + d_local.z*v.y,
			d_local.x*u.z + d_local.y*n.z + d_local.z*v.z);

		return d_world;
	}

	// Transform
	template <typename T>
	__host__ __device__ inline T transform(const T &d_in, const T transform_vector)
	{
		return d_in + transform_vector;
	}

	__host__ __device__ inline float3 rotateY(const float3 &origin, const float3 &old_dir, const float3 &new_dir)
	{		
		// Rotate matrix:
		// |cos		0		sin|	
		// |0		1		0  |	
		// |-sin	0		cos|	
		int dir = (cross(old_dir, new_dir).y > 0) ? 1:-1;// when parallel to (0,1,0), sin=0, doesn't have effect
		float cos = dot(old_dir, new_dir);
		float sin = dir*sqrtf(1 - cos*cos);
		float3 rotate_result = make_float3(cos*origin.x+sin*origin.z,
											origin.y,
											-sin*origin.x + cos*origin.z);
		return rotate_result;
	}

	//	Ray intersects with Parallelogram comfirmed by vertexs A, B and C, 
	//	which A is the angle subtend diagonal, e.g: In rectangle, A=Pi/2
	__host__ __device__ inline bool rayParallelogramIntersect(
		const float3 &orig, const float3 &dir,
		const float3 &A, const float3 &B, const float3 &C,
		float &t, float &u, float &v)
	{
		float3 E1 = B - A;
		float3 E2 = C - A;
		float3 pvec = cross(dir, E2);
		float det = dot(E1, pvec);

		// ray and triangle are parallel if det is close to 0
		if (fabsf(det) < Epsilon) return false;

		float invDet = 1 / det;

		float3 T = orig - A;
		u = dot(T, pvec)* invDet;
		if (u < 0 || u > 1) return false;

		float3 qvec = cross(T, E1);
		v = dot(dir, qvec)*invDet;
		if (v < 0 || v > 1) return false;

		t = dot(E2, qvec)*invDet;
		if (t < Epsilon) return false;

		return true;
	}

	template <typename T>
	inline void cpu2gpu(T *&d_out, T *&h_in, const size_t &size)
	{
		if(d_out==nullptr)
			checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(T)*size));
		checkCudaErrors(cudaMemcpy(d_out, h_in, sizeof(T)*size, cudaMemcpyHostToDevice));
	}

	template <typename T>
	inline void gpu2cpu(T *&h_out, T *&d_in, const size_t &size)
	{
		if(h_out==nullptr)
			h_out = new T[size];
		checkCudaErrors(cudaMemcpy(h_out, d_in, sizeof(T)*size, cudaMemcpyDeviceToHost));
	}

	__host__ __device__ inline bool setThreadsBlocks(dim3 &nBlocks, const int const &nThreads,
		const size_t &size, const bool const &threadFixed)
	{
		if (size > MAX_ALL_THREADS)
		{
			printf("There are too many threads to cope with, please use less threads.\n");
			return false;
		}

		int block_lastDim = (size + nThreads - 1) / nThreads;
		if (block_lastDim < MAX_BLOCK_SINGLE_DIM)
		{
			nBlocks.x = block_lastDim;
			nBlocks.y = nBlocks.z = 1;
			return true;
		}

		block_lastDim = (block_lastDim + MAX_BLOCK_SINGLE_DIM - 1) / MAX_BLOCK_SINGLE_DIM;
		if (block_lastDim < MAX_BLOCK_SINGLE_DIM)
		{
			nBlocks.x = MAX_BLOCK_SINGLE_DIM;
			nBlocks.y = block_lastDim;
			nBlocks.z = 1;
			return true;
		}
		else
		{
			nBlocks.x = nBlocks.y = MAX_BLOCK_SINGLE_DIM;
			nBlocks.z = (block_lastDim + MAX_BLOCK_SINGLE_DIM - 1) / MAX_BLOCK_SINGLE_DIM;
			return true;
		}
	}

	__host__ __device__ inline bool setThreadsBlocks(dim3 &nBlocks, int &nThreads, const size_t &size)
	{
		nThreads = (MAX_THREADS <= size) ? MAX_THREADS : size;
		return setThreadsBlocks(nBlocks, nThreads, size, true);
	}

	__host__ __device__ inline unsigned long long int getThreadId()
	{
		// unique block index inside a 3D block grid
		const unsigned long long int blockId = blockIdx.x //1D
			+ blockIdx.y * gridDim.x //2D
			+ gridDim.x * gridDim.y * blockIdx.z; //3D

		// global unique thread index, block dimension uses only x-coordinate
		const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;

		return threadId;
	}
	
	__host__ __device__ inline float3 angle2xyz(float2 d_angles)
	{
		return make_float3(sinf(d_angles.x)*cosf(d_angles.y),
			cosf(d_angles.x),
			sinf(d_angles.x)*sinf(d_angles.y));
	}

	//	Unroll and roll the index and address
	__host__ __device__ inline int unroll_index(int3 index, int3 matrix_size)
	{
		int address = index.x*matrix_size.y*matrix_size.z + index.y*matrix_size.z + index.z;
		return address;
	}

	__host__ __device__ inline int unroll_index(int2 index, int2 matrix_size)
	{
		int address = index.x*matrix_size.y + index.y;
		return address;
	}
}