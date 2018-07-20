#include "common_var.h"
#include "solar_scene.h"
#include "scene_normals.h"
#include "Gtest.cuh"
#include "SPA.h"
#include <windows.h>
#include <iostream>
#define pi 3.1415926535897932384626433832795028841971
#define receiverH 13.0
curandGenerator_t *RandomGenerator::gen;

using namespace std;

int main(int argc,char*argv[]) {
	if (argc!=6){
		cout<<"error!";
		return 0;
	} 
	float X,Y,Z;
	float A,B;
	sscanf(argv[1],"%f",&X);
	sscanf(argv[2],"%f",&Y);
	sscanf(argv[3],"%f",&Z);
	sscanf(argv[4],"%f",&A);
	sscanf(argv[5],"%f",&B);
	ofstream clc(solarenergy::result_save_nonsmooth_path.c_str(), ios::trunc);
	clc.close();
	ofstream clc1(solarenergy::result_save_smooth_path.c_str(), ios::trunc);
	clc1.close();
	float a, b, c;
	float start, stop, durationTime;
	start = clock();
	spa_data spa;
	spa.year = 2010;
	spa.minute = 0;
	spa.second = 0;
	spa.timezone = 8;
	spa.delta_ut1 = 0;
	spa.delta_t = 67;
	spa.longitude = 120.2;
	spa.latitude = 30.3;
	spa.elevation = 40;
	spa.pressure = 1013;
	spa.temperature = 11;
	spa.slope = 0;
	spa.azm_rotation = -10;
	spa.atmos_refract = 0.5667;
	spa.function = SPA_ALL;
	// Step 0: Initialize gen seed in RandomGenerator 
	
	curandGenerator_t gen_test;
	curandCreateGenerator(&gen_test, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen_test, time(NULL));
	RandomGenerator::gen = &gen_test;
	//cout <<"filepath: "<< solarenergy::scene_filepath << endl;
	for (int month = 1; month <= 12; month++) {
		for (int date = 1; date < 29; date += 3) {
			for (int hour = 6; hour <= 18; hour += 2) {
				//double start, stop, durationTime;
			//	float masterNormal[3] = { 0.64424 ,0.436177 ,-0.62825};
			    float masterNormal[3]; 
				spa.month = month;
				spa.day = date;
				spa.hour = hour;
				int result = spa_calculate(&spa);
				if (result == 0)  //check for SPA errors
				{
					if (spa.zenith > 75) {
						continue;
					}
					a = -sin(spa.zenith / 180 * pi)*sin(spa.azimuth / 180 * pi);
					b = -cos(spa.zenith / 180 * pi);
					c = sin(spa.zenith / 180 * pi)*cos(spa.azimuth / 180 * pi);
					solarenergy::dni = getDNI(&spa);
				}
				else {
					printf("SPA Error Code: %d\n", result);
					return 0;

				}
               float distance;
	           float rox, roy, roz;
	           float NR;
	           distance = sqrt(X*X + Z*Z + (receiverH - Y)*(receiverH - Y));
	           rox = -X / distance;
	           roz = -Z / distance; 
	           roy = (receiverH - Y) / distance;
	           NR = sqrt((-a + rox)*(-a + rox) + (-b + roy)*(-b + roy) + (-c + roz)*(-c + roz));
	           masterNormal[0] = (-a + rox) / NR;
	           masterNormal[1] = (-b + roy) / NR;
   	           masterNormal[2] = (-c + roz) / NR; 

			   float3 * p = NULL;
			   solarenergy::sun_dir = make_float3(a, b, c);

				// Step 1: Load files
				//cout << solarenergy::disturb_std << endl;
				SolarScene *solar_scene;
				solar_scene = SolarScene::GetInstance(masterNormal,A,B,X,Y,Z);

				// Step 2: Load normals
				//		   Initialize the content in the scene
			//	float3 *normals = load_noraml();
				//solar_scene->InitContent(normals);
				
				// Step 3: Ray Tracing
				results_with_given_normal(*solar_scene,p);
				stop = clock();
				durationTime = ((double)(stop - start)) / CLK_TCK;
				cout << "³ÌÐòºÄÊ±£º" << durationTime << " s" << endl;
				//system("pause");
				// Finally, destroy solar_scene
				solar_scene->~SolarScene();
			}
		}
	}
	// destroy gen
	RandomGenerator::gen = nullptr;
	curandDestroyGenerator(gen_test);
	return 0;
}