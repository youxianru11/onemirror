// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

struct valid_element
{
	float width;
	float height;
	valid_element(float width_input, float height_input):width(width_input),height(height_input){}
	__host__ __device__ bool operator()(const float2 element)
	{
		return (element.x > 0) && (element.y > 0) && // non-empty
			(element.x < width) && (element.y < height);// non out of space
	}
};

// Step 2:
//	compact it and save the result into sub_orign, return its length
int generate_microhelio_origin(float2 *d_input, float2 *&d_output,
	const float width, const float height,
	const int width_int, const int height_int);