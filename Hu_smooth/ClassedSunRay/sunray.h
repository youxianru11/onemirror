#pragma once
#include "cuda_runtime.h"

class SunRay
{
public:
	__device__ __host__ SunRay() :d_samplelights_(nullptr), d_perturbation_(nullptr) {}

	__device__ __host__ SunRay(float3 sun_dir,int num_sunshape_groups,int lights_per_group,
		float dni,float csr) : SunRay(){
		sun_dir_ = sun_dir;
		dni_ = dni;
		csr_ = csr;
		num_sunshape_groups_ = num_sunshape_groups;
		num_sunshape_lights_per_group_ = lights_per_group;
	}
	//__device__ __host__ SunRay(const SunRay &sunray)
	//{
	//	sun_dir_ = sunray.sun_dir_;
	//	dni_ = sunray.dni_;
	//	csr_= sunray.csr_;
	//	num_sunshape_groups_ = sunray.num_sunshape_groups_;
	//	num_sunshape_lights_per_group_= sunray.num_sunshape_lights_per_group_;
	//	d_samplelights_ = sunray.d_samplelights_;								
	//	d_perturbation_ = sunray.d_perturbation_;
	//}

	__device__ __host__ ~SunRay()
	{
		if(d_samplelights_)
			d_samplelights_ = nullptr;
		if(d_perturbation_)
			d_perturbation_ = nullptr;
	}

	__device__ __host__ void CClear()
	{
		if (d_samplelights_)
		{
			cudaFree(d_samplelights_);
			d_samplelights_ = nullptr;
		}
		
		if (d_perturbation_)
		{
			cudaFree(d_perturbation_);
			d_perturbation_ = nullptr;
		}
	}

	float3 sun_dir_;					// e.g. 0.306454	-0.790155	0.530793
	float dni_;							// e.g. 1000.0
	float csr_;							// e.g. 0.1
	int num_sunshape_groups_;			// e.g. 8
	int num_sunshape_lights_per_group_;	// e.g. 1024
	float3* d_samplelights_;			// e.g. point to sample lights memory on GPU
										//		memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
	float3* d_perturbation_;			// e.g. point to the memory on GPU
										//		which obeys Gaussian distribution 
										//		memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
};