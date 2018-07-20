#pragma once
#ifndef COMMON_VAR_H
#define COMMON_VAR_H

#include <cuda_runtime.h>
#include <string>

using namespace std;

namespace solarenergy {
	//sun ray related default value
	extern float3 sun_dir;
	extern float dni;
	extern float csr;
	extern float num_sunshape_groups;
	extern float num_sunshape_lights_per_group;

	extern float helio_pixel_length;
	extern float receiver_pixel_length;
	extern float reflected_rate;
	extern float disturb_std;



	//default scene file
	extern string scene_filepath;
	extern string normal_filepath;

	// Result save path
	extern string result_save_nonsmooth_path;
	extern string result_save_smooth_path;
};

#endif