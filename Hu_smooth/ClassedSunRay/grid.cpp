#include "grid.h"

void RectGrid::Cinit()
{
	grid_num_.x = ceil(size_.x / interval_.x);
	grid_num_.y = ceil(size_.y / interval_.y);
	grid_num_.z = ceil(size_.z / interval_.z);
}

void RectGrid::CClear()
{
	if (d_grid_helio_match_)
	{
		cudaFree(d_grid_helio_match_);
		d_grid_helio_match_ = nullptr;
	}

	if (d_grid_helio_index_)
	{
		cudaFree(d_grid_helio_index_);
		d_grid_helio_index_ = nullptr;
	}
}

int boxIntersect(const int &mirrowId, const float3 &min_pos, const float3 &max_pos, const RectGrid &grid,
	vector<vector<int>> &grid_mirrow_match_vector)
{
	int size = 0;
	int3 min_grid_pos = make_int3((min_pos - grid.pos_).x / grid.interval_.x,
		(min_pos - grid.pos_).y / grid.interval_.y,
		(min_pos - grid.pos_).z / grid.interval_.z);
	int3 max_grid_pos = make_int3((max_pos - grid.pos_).x / grid.interval_.x,
		(max_pos - grid.pos_).y / grid.interval_.y,
		(max_pos - grid.pos_).z / grid.interval_.z);

	for (int x = min_grid_pos.x; x <= max_grid_pos.x; ++x)
		for (int y = min_grid_pos.y; y <= max_grid_pos.y; ++y)
			for (int z = min_grid_pos.z; z <= max_grid_pos.z; ++z)
			{
				int pos = x * grid.grid_num_.y * grid.grid_num_.z + y * grid.grid_num_.z + z;
				grid_mirrow_match_vector[pos].push_back(mirrowId);
				++size;
			}
	return size;
}

void RectGrid::CGridHelioMatch(const vector<Heliostat *> &h_helios) // set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
{
	float3 minPos, maxPos;
	float  diagonal_length, radius;
	num_grid_helio_match_ = 0;
	vector<vector<int>> grid_mirrow_match_vector(grid_num_.x * grid_num_.y * grid_num_.z);
	for (int i = start_helio_pos_; i < start_helio_pos_  + num_helios_; ++i)
	{
		diagonal_length = length(h_helios[i]->size_);

		radius = sqrt(diagonal_length) / 2;
		minPos = h_helios[i]->pos_ - radius;
		maxPos = h_helios[i]->pos_ + radius;

		num_grid_helio_match_ += boxIntersect(i, minPos, maxPos, *this, grid_mirrow_match_vector);
	}

	int *h_grid_helio_index = new int[grid_num_.x * grid_num_.y * grid_num_.z + 1];
	h_grid_helio_index[0] = 0;
	int *h_grid_helio_match = new int[num_grid_helio_match_];

	int index = 0;
	for (int i = 0; i < grid_num_.x * grid_num_.y * grid_num_.z; ++i)
	{
		h_grid_helio_index[i + 1] = h_grid_helio_index[i] + grid_mirrow_match_vector[i].size();
		for (int j = 0; j < grid_mirrow_match_vector[i].size(); ++j, ++index)
			h_grid_helio_match[index] = grid_mirrow_match_vector[i][j];
	}

	global_func::cpu2gpu(d_grid_helio_match_, h_grid_helio_match, num_grid_helio_match_);
	global_func::cpu2gpu(d_grid_helio_index_, h_grid_helio_index, grid_num_.x * grid_num_.y * grid_num_.z + 1);

	delete[] h_grid_helio_index;
	delete[] h_grid_helio_match;
	h_grid_helio_index = nullptr;
	h_grid_helio_match = nullptr;
}