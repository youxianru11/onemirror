#include "common_var.h"


//Sun ray related default value

//float3 solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);
//float3 solarenergy::sun_dir = make_float3(0.57677749827947256, -0.80768127066332496, -0.12238742785984998);
float3 solarenergy::sun_dir;
float solarenergy::csr = 0.1f;
float solarenergy::dni;
float solarenergy::num_sunshape_groups = 128;
float solarenergy::num_sunshape_lights_per_group = 2048;

// Receiver with default value
float solarenergy::receiver_pixel_length = 0.01;

// Heliostat with default value
float solarenergy::disturb_std = 0.001;
float solarenergy::reflected_rate = 0.88;
float solarenergy::helio_pixel_length = 0.01;

//default scene file
string solarenergy::scene_filepath = "../userData/face2face_shadow.scn";
string solarenergy::normal_filepath = "../userData/normal.txt";

// Save path
string solarenergy::result_save_nonsmooth_path = "../result/result_nonsmooth.txt";
string solarenergy::result_save_smooth_path = "../result/result_smooth.txt";