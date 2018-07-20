
#include "solar_scene.h"
#include "scene_file_proc.h"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance()
{
	InitInstance();
	return m_instance;
}

SolarScene* SolarScene::GetInstance(float *masterNormal,float a,float b,float X,float Y,float Z) {
	InitInstance(masterNormal,a,b,X,Y,Z);
	return m_instance;
}

void SolarScene::InitInstance()
{
	m_instance = new SolarScene();
}

void SolarScene::InitInstance(float *masternormal,float a,float b,float X,float Y,float Z) {
	m_instance = new SolarScene(masternormal,a,b,X,Y,Z);

}

SolarScene::SolarScene() {
	//init the random
	RandomGenerator::initSeed();
	//init the sunray
	sunray_ = new SunRay(solarenergy::sun_dir,solarenergy::num_sunshape_groups,solarenergy::num_sunshape_lights_per_group,
		solarenergy::dni,solarenergy::csr);
	InitSolarScece();
}

SolarScene::SolarScene(float *masternormal,float a,float b,float X,float Y,float Z) {
	//init the random
	RandomGenerator::initSeed();
	//init the sunray
	sunray_ = new SunRay(solarenergy::sun_dir, solarenergy::num_sunshape_groups, solarenergy::num_sunshape_lights_per_group,
		solarenergy::dni, solarenergy::csr);
	InitSolarScece(masternormal,a,b,X,Y,Z);

}
SolarScene::~SolarScene() {
	// 1. free memory on GPU
	free_scene::gpu_free(receivers);
	free_scene::gpu_free(grid0s);
	free_scene::gpu_free(sunray_);

	// 2. free memory on CPU
	free_scene::cpu_free(receivers);
	free_scene::cpu_free(grid0s);
	free_scene::cpu_free(heliostats);
	free_scene::cpu_free(sunray_);
}

bool SolarScene::InitSolarScece() {
	string filepath = solarenergy::scene_filepath;
	return LoadSceneFromFile(filepath);
}

bool SolarScene::InitSolarScene(string filepath) {
	return LoadSceneFromFile(filepath);
}

bool SolarScene::InitSolarScece(float *masterNormal,float a,float b,float X,float Y,float Z) {
	
	return LoadSceneFromFile( masterNormal,a,b,X,Y,Z);
}

bool SolarScene::LoadSceneFromFile(string filepath) {

	SceneFileProc proc;
	return proc.SceneFileRead(this,filepath);
}

bool SolarScene::LoadSceneFromFile(float *masterNormal,float a,float b,float X,float Y,float Z) {

	SceneFileProc proc;
	return proc.SceneFileRead(this, a, b, masterNormal,X,Y,Z);
}
bool SolarScene::InitContent()
{
	// 1. Sunray
	SceneProcessor::set_sunray_content(*this->sunray_);

	// 2. Grid
	SceneProcessor::set_grid_content(this->grid0s, this->heliostats);

	// 3. Receiver
	SceneProcessor::set_receiver_content(this->receivers);

	// 4. Heliostats
	float3 focus_center = this->receivers[0]->focus_center_;			// must after receiver init
	SceneProcessor::set_helio_content(this->heliostats, focus_center, this->sunray_->sun_dir_);

	return true;
}

bool SolarScene::InitContent(float3 *normals)
{
	// 1. Sunray
	SceneProcessor::set_sunray_content(*this->sunray_);

	// 2. Grid
	SceneProcessor::set_grid_content(this->grid0s, this->heliostats);

	// 3. Receiver
	SceneProcessor::set_receiver_content(this->receivers);

	// 4. Heliostats
	SceneProcessor::set_helio_content(this->heliostats, normals);

	return true;
}