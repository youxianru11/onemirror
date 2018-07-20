#pragma once

#include "solar_scene.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <string>

// Value-Defintions of the different String values
enum StringValue {
	pos,
	size,
	norm,
	face,
	end,
	gap,
	matrix,
	helio,
	inter,
	n,
	type,
	illegal
};

// Value-Defintions of the different String values
enum InputMode{
	none,
	ground,
	receiver,
	grid,
	heliostat
};


class SceneFileProc{
public:
	SceneFileProc();
	bool SceneFileRead(SolarScene *solarscene, std::string filepath);
	bool SceneFileRead(SolarScene *solarscene, float a, float b,float *mirrorNormal,float X,float Y,float Z);
	void matrixMulti(float A[][3], float *b, float *c);
private:
	SolarScene *solarScene_;  //eqaul the Scolar::GetInstance
	// Map to associate the strings with the enum values
	std::map<std::string, StringValue> string_value_read_map;
	StringValue Str2Value(string str);

};