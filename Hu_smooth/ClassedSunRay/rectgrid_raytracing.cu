#include "rectgrid_raytracing.cuh"

inline __device__ float calTMax(const float &dir, const float &interval, const int &current_index, const float &current_pos)
{
	if (dir >= 0)
		return float(current_index + 1)*interval - current_pos;
	else
		return current_pos - float(current_index)*interval;
}

template <typename T>
inline __host__ __device__ T absDivide(const T &denominator, const T &numerator)
{
	if (numerator <= Epsilon && numerator >= -Epsilon)
		return T(INT_MAX);
	return abs(denominator / numerator);
}

//	3DDDA
//	Intersect with the heliostat within this rectangle grid
__device__ bool Intersect(const float3 &orig, const float3 &dir,
	const int &grid_address, const float3 *d_heliostat_vertexs,
	const int const *d_grid_heliostat_match, const int const *d_grid_heliostat_index)
{
	for (unsigned int i = d_grid_heliostat_index[grid_address]; i < d_grid_heliostat_index[grid_address + 1]; ++i)
	{
		unsigned int heliostat_index = 3 * d_grid_heliostat_match[i];
		float u, v, t;
		bool intersect = global_func::rayParallelogramIntersect(orig, dir, d_heliostat_vertexs[heliostat_index],
			d_heliostat_vertexs[heliostat_index + 1],
			d_heliostat_vertexs[heliostat_index + 2], t, u, v);
		if (intersect)
			return true;
	}
	return false;
}

__device__ bool NotCollision(const float3 &d_orig, const float3 &d_dir,
	RectGrid &rectgrid, const float3 *d_helio_vertexs)
{
	// Step 1 - Initialization
	//	Step 1.1 Initial current position of origin in the scene
	int3 pos = make_int3((d_orig - rectgrid.pos_) / rectgrid.interval_);

	//	Step 1.2 StepX, StepY, StepZ
	int3 Step;
	Step.x = (d_dir.x >= 0) ? 1 : -1;
	Step.y = (d_dir.y >= 0) ? 1 : -1;
	Step.z = (d_dir.z >= 0) ? 1 : -1;

	//	Step 1.3 Initial tmaxX, tmaxY, tmaxZ
	float3 tMax, t;
	t.x = calTMax(d_dir.x, rectgrid.interval_.x, pos.x, d_orig.x - rectgrid.pos_.x);
	t.y = calTMax(d_dir.y, rectgrid.interval_.y, pos.y, d_orig.y - rectgrid.pos_.y);
	t.z = calTMax(d_dir.z, rectgrid.interval_.z, pos.z, d_orig.z - rectgrid.pos_.z);

	tMax.x = absDivide(t.x, d_dir.x);	// avoid divide 0
	tMax.y = absDivide(t.y, d_dir.y);
	tMax.z = absDivide(t.z, d_dir.z);

	//	Step 1.4 Initial tDeltaX, tDeltaY, tDeltaZ
	float3 tDelta;
	tDelta.x = absDivide(rectgrid.interval_.x, d_dir.x);// avoid divide 0
	tDelta.y = absDivide(rectgrid.interval_.y, d_dir.y);
	tDelta.z = absDivide(rectgrid.interval_.z, d_dir.z);

	// Step 2 - Intersection
	int3 grid_index = pos;
	int grid_address = global_func::unroll_index(grid_index, rectgrid.grid_num_);
	while (true)
	{
		if (tMax.x < tMax.y)
		{
			if (tMax.x < tMax.z)
			{
				grid_index.x += Step.x;
				if (grid_index.x >= rectgrid.grid_num_.x || grid_index.x<0)
					// Outside grid
					return true;
				tMax.x += tDelta.x;
			}
			else
			{
				grid_index.z += Step.z;
				if (grid_index.z >= rectgrid.grid_num_.z || grid_index.z<0)
					// Outside grid
					return true;
				tMax.z += tDelta.z;
			}
		}
		else
		{
			if (tMax.y < tMax.z)
			{
				grid_index.y += Step.y;
				if (grid_index.y >= rectgrid.grid_num_.y || grid_index.y<0)
					// Outside grid
					return true;
				tMax.y += tDelta.y;
			}
			else
			{
				grid_index.z += Step.z;
				if (grid_index.z >= rectgrid.grid_num_.z || grid_index.z<0)
					// Outside grid
					return true;
				tMax.z += tDelta.z;
			}
		}
		grid_address = global_func::unroll_index(grid_index, rectgrid.grid_num_);

		if (Intersect(d_orig, d_dir,
			grid_address, d_helio_vertexs, rectgrid.d_grid_helio_match_, rectgrid.d_grid_helio_index_))
			return false;
	}
	return false;
}

// Step 3: intersect with receiver
inline __device__ float eta_aAlpha(const float &d)
{
	/*if (d <= 1000.0f)
		return 0.99331f - 0.0001176f*d + 1.97f*(1e-8f) * d*d;
	else
		return expf(-0.0001106f*d);*/
	return 1;
}

inline __device__ float calEnergy(const float3 &sun_dir, const float3 &normal, const float &eta)
{
	float cosine = fabsf(dot(sun_dir, normal));
	return cosine*eta;
}

__device__ void receiver_drawing(RectangleReceiver &receiver, const SunRay &sunray,
	const float3 &orig, const float3 const &dir, const float3 const &normal)
{
	//	Step1: Intersect with receiver
	float t, u, v;
	bool intersect = receiver.GIntersect(orig, dir, t, u, v);
	if (!receiver.GIntersect(orig, dir, t, u, v))
		return;

	int2 row_col = make_int2(u* receiver.resolution_.y, v* receiver.resolution_.x); // Intersect location

																					//	Step2: Calculate the energy of the light
	float eta = eta_aAlpha(t);
	float energy = calEnergy(sunray.sun_dir_, normal, eta);

	//	Step3: Add the energy to the intersect position
	int address = row_col.x*receiver.resolution_.x + row_col.y;  //col_row.y + col_row.x*resolution.y;
	atomicAdd(&(receiver.d_image_[address]), energy);
}

__global__ void map_tracing(const SunRay sunray,		// sun
	RectGrid grid,					// grid
	RectangleReceiver receiver,			// receiver
	const float3 *d_helio_vertexs,	// 3 vertexs of heliostats
	const float3 *d_microhelio_normals,	// micro-heliostat's normal
	const float3 *d_microhelio_center,	// micro-heliostat's origins
	const int *d_microhelio_start,		// micro-heliostat's belonging group number
	const int microhelio_num)
{
	long long int myId = global_func::getThreadId();
	if (myId >= microhelio_num*sunray.num_sunshape_lights_per_group_)
		return;

	//	Step 1: whether the incident light is shadowed by other heliostats
	int nLights = sunray.num_sunshape_lights_per_group_;
	int nGroup = sunray.num_sunshape_groups_;
	int address = (d_microhelio_start[myId / nLights] + myId%nLights)%(nLights*nGroup);
	float3 dir = sunray.d_samplelights_[address];				// get the y-aligned direction
	dir = global_func::local2world(dir, sunray.sun_dir_);		// get the sun_direction-aligned direction	
	dir = -dir;													// Since the sun direction is incident direction, reverse it

	float3 orig = d_microhelio_center[myId / nLights];			// get the center of submirror

	if (!NotCollision(orig, dir, grid, d_helio_vertexs))
		return;

	//	Step 2: whether the reflect light is shadowed by other heliostats	
	float3 normal = d_microhelio_normals[myId / nLights];	
	int start_id = (myId / nLights - 1>0) ? (myId / nLights - 1) : 0;
	address = (d_microhelio_start[start_id] + myId%nLights) % (nLights*nGroup);
	float3 turbulence = sunray.d_perturbation_[address];
	normal = global_func::local2world(turbulence, normal); normal = normalize(normal);

	dir = -dir;
	dir = reflect(dir, normal);					// reflect light
	dir = normalize(dir);
	if (!NotCollision(orig, dir, grid, d_helio_vertexs))
		return;

	// Step 3: intersect with receiver
	receiver_drawing(receiver, sunray, orig, dir, normal);
}