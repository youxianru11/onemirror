#include "compact_sample.cuh"
#include <iostream>

using namespace std;

int generate_microhelio_origin(float2 *d_input, float2 *&d_output,
	const float width, const float height,
	const int width_int, const int height_int)
{
	thrust::device_ptr<float2> dev_input_ptr(d_input);
	thrust::device_ptr<float2> dev_tmp_output_ptr = thrust::device_malloc<float2>(width_int*height_int);
			
	thrust::device_ptr<float2> dev_tmp_output_ptr_end =
		thrust::copy_if(dev_input_ptr, dev_input_ptr + width_int*height_int,
			dev_tmp_output_ptr, valid_element(width, height));
	int num = dev_tmp_output_ptr_end - dev_tmp_output_ptr;	
	d_output = thrust::raw_pointer_cast(dev_tmp_output_ptr);

	return num;
}