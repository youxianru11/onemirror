#include "Gtest.cuh"
#include "scene_instance_process.h"
#include "recthelio_tracing.h"
#include "image_save.h"
#include "image_smooth.cuh"

#include <sstream>
#include <chrono>

void results_with_given_normal(SolarScene &solar_scene, float3 *normals)
{
	RectGrid *grid= dynamic_cast<RectGrid *>(solar_scene.grid0s[0]);
	Receiver *recv = dynamic_cast<RectangleReceiver *>(solar_scene.receivers[0]);
	float totalEnergy = 0.0f;
	// Init the heliostats
	//SceneProcessor::set_helio_content(solar_scene.heliostats, normals);
	// Clear result of receiver
	recv->Cclean_image_content();
	float Id = solar_scene.sunray_->dni_;
	float *h_image = nullptr;
	for (int i = 0; i < grid->num_helios_; ++i)
	{
		int id = i + grid->start_helio_pos_;
		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene.heliostats[id]);
		
		// Reset the content of sun
		SceneProcessor::set_sunray_content(*solar_scene.sunray_);
		
		totalEnergy += abs(dot(solar_scene.heliostats[id]->normal_, solar_scene.sunray_->sun_dir_)*Id*solar_scene.heliostats[id]->size_.x*solar_scene.heliostats[id]->size_.z);

		// Ray-tracing
		int num_subcenters = recthelio_ray_tracing(*solar_scene.sunray_,
			*recv, *recthelio, *grid,
			solar_scene.heliostats);
	}

	global_func::gpu2cpu(h_image, recv->d_image_, recv->resolution_.x*recv->resolution_.y);
	// Non Smooth
	
	float Ssub = solarenergy::helio_pixel_length*solarenergy::helio_pixel_length;
	float rou = solarenergy::reflected_rate;
	int Nc = solar_scene.sunray_->num_sunshape_lights_per_group_;
	float Srec = recv->pixel_length_*recv->pixel_length_;
	totalEnergy *= rou / Srec;
	for (int k = 0; k < recv->resolution_.x * recv->resolution_.y; ++k)	
		h_image[k] *= Id * Ssub * rou / Nc / Srec;
	ImageSaver::savetxt(solarenergy::result_save_nonsmooth_path, recv->resolution_.x, recv->resolution_.y, h_image,totalEnergy);

	// Smooth
	int kernel_radius = 5;
	float trimmed_ratio = 0.02;
	ImageSmoother::image_smooth(recv->d_image_,
								kernel_radius, trimmed_ratio,
								recv->resolution_.x, recv->resolution_.y);
	global_func::gpu2cpu(h_image, recv->d_image_, recv->resolution_.x*recv->resolution_.y);
	for (int k = 0; k < recv->resolution_.x * recv->resolution_.y; ++k)
		h_image[k] *= Id * Ssub * rou / Nc / Srec;

	ImageSaver::savetxt(solarenergy::result_save_smooth_path, recv->resolution_.x, recv->resolution_.y, h_image,totalEnergy);
	
	delete[] h_image; 
	h_image = nullptr;
}


//void test(SolarScene &solar_scene)
//{
//	string save_path("../result/QMCRT/13/normal_0.002//_.txt"); // e.g. - ../result/QMCRT/13/non_smooth/13_0.txt
//	int helio_id[] = { 13 };
//	int run_times_start = 1000, run_times_end = 1010;
//
//	// Smooth result
//	int kernel_radius = 5;
//	float trimmed_ratio = 0.02;
//	float *h_image = nullptr;
//	Receiver *recv = dynamic_cast<RectangleReceiver *>(solar_scene.receivers[0]);
//
//	// time
//	auto start = std::chrono::high_resolution_clock::now();			 // nano-seconds
//	auto elapsed = std::chrono::high_resolution_clock::now() - start;// nano-seconds
//	long long total_time = 0, ray_gen_time = 0;
//	long long time_tracing = 0, time_subcenter = 0, time_group_gen = 0;
//	long long time_smooth = 0;
//
//	//recthelio->type = SubCenterType::Poisson;
//	for (int i = 0; i < sizeof(helio_id) / sizeof(helio_id[0]); ++i)
//	{
//		// Initialize time
//		total_time = 0, ray_gen_time = 0;
//		time_tracing = 0, time_subcenter = 0, time_group_gen = 0;
//		time_smooth = 0;
//
//		string tmp_path = save_path;
//		//tmp_path.insert(tmp_path.size() - 7, to_string(helio_id[i]));
//
//		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene.heliostats[helio_id[i]]);
//		for (int j = run_times_start; j < run_times_end; ++j)
//		{
//			// Clear result of receiver
//			recv->Cclean_image_content();
//
//			// Reset the content of sun
//			start = std::chrono::high_resolution_clock::now();
//			SceneProcessor::set_sunray_content(*solar_scene.sunray_);
//			elapsed = std::chrono::high_resolution_clock::now() - start;
//			ray_gen_time+= std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//
//			// Ray-tracing
//			int num_subcenters = recthelio_ray_tracing(*solar_scene.sunray_,
//				*recv, *recthelio,
//				*solar_scene.grid0s[0],
//				solar_scene.heliostats,
//				time_tracing,
//				time_subcenter,
//				time_group_gen);
//
//			global_func::gpu2cpu(h_image, solar_scene.receivers[0]->d_image_, recv->resolution_.x*recv->resolution_.y);
//			
//			// Non Smooth
//			float Id = solar_scene.sunray_->dni_;
//			float Ssub = recthelio->pixel_length_*recthelio->pixel_length_;
//			float rou = solarenergy::reflected_rate;
//			int Nc = solar_scene.sunray_->num_sunshape_lights_per_group_;
//			float Srec = recv->pixel_length_*recv->pixel_length_;
//			float sum = 0.0;
//			for (int k = 0; k < recv->resolution_.x * recv->resolution_.y; ++k)
//			{
//				h_image[k] *= Id * Ssub * rou / Nc / Srec;
//				sum += h_image[k];
//			}
//
//			// Save image	
//			string tmp_save_path = tmp_path;
//			tmp_save_path.insert(tmp_save_path.size() - 6, "non_smooth");
//			tmp_save_path.insert(tmp_save_path.size() - 5, to_string(helio_id[i]));
//			tmp_save_path.insert(tmp_save_path.size() - 4, to_string(j));
//			ImageSaver::savetxt(tmp_save_path, recv->resolution_.x, recv->resolution_.y, h_image);
//
//			// Smooth
//			start = std::chrono::high_resolution_clock::now();
//			ImageSmoother::image_smooth(recv->d_image_,
//				kernel_radius, trimmed_ratio,
//				recv->resolution_.x, recv->resolution_.y);
//			elapsed = std::chrono::high_resolution_clock::now() - start;
//			time_smooth += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//
//			global_func::gpu2cpu(h_image, solar_scene.receivers[0]->d_image_, recv->resolution_.x*recv->resolution_.y);
//			for (int k = 0; k < recv->resolution_.x*recv->resolution_.y; ++k)
//				h_image[k] *= Id * Ssub * rou / Nc / Srec;
//			tmp_save_path = tmp_path;
//			tmp_save_path.insert(tmp_save_path.size() - 6, "smooth");
//			tmp_save_path.insert(tmp_save_path.size() - 5, to_string(helio_id[i]));
//			tmp_save_path.insert(tmp_save_path.size() - 4, to_string(j));
//			ImageSaver::savetxt(tmp_save_path, recv->resolution_.x, recv->resolution_.y, h_image);
//			cout << tmp_save_path << endl;
//		}
//		total_time = ray_gen_time + time_subcenter + time_group_gen + time_tracing + time_smooth;
//
//		std::cout << to_string(helio_id[i]) << endl;
//		std::cout << "Total Average Time:\t" + to_string(double(total_time / (run_times_end - run_times_start))) << endl;
//		std::cout << "Rays Generation Time:\t" + to_string(double(ray_gen_time / (run_times_end - run_times_start))) << endl;
//		std::cout << "Subcenter Generation Time:\t" + to_string(double(time_subcenter / (run_times_end - run_times_start))) << endl;
//		std::cout << "Groups Generation Time:\t" + to_string(double(time_group_gen / (run_times_end - run_times_start))) << endl;
//		std::cout << "Tracing Time:\t" + to_string(double(time_tracing / (run_times_end - run_times_start))) << endl;
//		std::cout << "Smooth Time:\t" + to_string(double(time_smooth / (run_times_end - run_times_start))) << endl;
//		std::cout << endl;
//
//		string info_path = tmp_path.erase(tmp_path.size() - 6);//../result/13/QMCRT/1group/
//		info_path.insert(tmp_path.size(), to_string(helio_id[i]) + "_information.txt");
//		ofstream fout(info_path);
//		fout << to_string(helio_id[i]) << endl;
//		fout << "Total Average Time:\t" + to_string(double(total_time / (run_times_end - run_times_start))) << endl;
//		fout << "Rays Generation Time:\t" + to_string(double(ray_gen_time / (run_times_end - run_times_start))) << endl;
//		fout << "Subcenter Generation Time:\t" + to_string(double(time_subcenter / (run_times_end - run_times_start))) << endl;
//		fout << "Groups Generation Time:\t" + to_string(double(time_group_gen / (run_times_end - run_times_start))) << endl;
//		fout << "Tracing Time:\t" + to_string(double(time_tracing / (run_times_end - run_times_start))) << endl;
//		fout << "Smooth Time:\t" + to_string(double(time_smooth / (run_times_end - run_times_start))) << endl;
//		fout.close();
//	}
//
//	delete[] h_image;
//	h_image = nullptr;
//	solar_scene.sunray_->CClear();
//	recv->Cclean_image_content();
//}