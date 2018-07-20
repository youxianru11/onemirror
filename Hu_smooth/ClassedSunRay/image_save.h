#pragma once
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class ImageSaver 
{
public:	
	static void savetxt(const string filename, int w, int h, float *h_data, int precision=2);
	static void savetxt(const string filename, int w, int h, float *h_data, float t);
};