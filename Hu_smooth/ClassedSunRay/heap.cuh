#pragma once

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

class Heap {
private:
	int numElems;
	int max_size;
	bool(*cmp_func)(const float &, const float &);

	//public:
	float *elements;
	//Heap() :numElems(1),elements(nullptr) {}
	__device__ __host__ Heap(int size, float *element_entry) :numElems(1), max_size(size + 1), elements(element_entry)
	{
		/*elements = new float[max_size];*/
	}

	//__device__ __host__ ~Heap()
	//{
	//	delete[] elements;
	//	elements = nullptr;
	//}

	__device__ __host__ bool insert(float elem)
	{
		if (full())
			return false;

		int i = numElems++;
		for (; i != 1 && cmp_func(elem, elements[i / 2]); i /= 2)
		{
			elements[i] = elements[i / 2];
		}
		elements[i] = elem;
		return true;
	}

	__device__ __host__ float delete_t()
	{
		if (empty())
			return -100;

		float return_value = elements[1];
		float last_value = elements[--numElems];
		int i = 1, cmp_id;
		for (; 2 * i < numElems;)
		{
			if (2 * i + 1 >= numElems ||							// only has left child
				cmp_func(elements[2 * i], elements[2 * i + 1]))		// have 2 children		
				cmp_id = 2 * i;
			else
				cmp_id = 2 * i + 1;

			if (cmp_func(elements[cmp_id], last_value))
			{
				elements[i] = elements[cmp_id];
				i = cmp_id;
			}
			else
				break;
		}
		elements[i] = last_value;
		return return_value;
	}

	__device__ __host__ float delete_and_insert(float elem)
	{
		if (empty())
			return -100;

		float return_value = elements[1];
		int i = 1, cmp_id;
		for (; 2 * i < numElems;)
		{
			cmp_id = (2 * i + 1 >= numElems ||							// only has left child
				cmp_func(elements[2 * i], elements[2 * i + 1])) ? 2 * i : 2 * i + 1;

			//if (2 * i + 1 >= numElems ||							// only has left child
			//	cmp_func(elements[2 * i], elements[2 * i + 1]))		// have 2 children		
			//	cmp_id = 2 * i;
			//else
			//	cmp_id = 2 * i + 1;

			if (cmp_func(elements[cmp_id], elem))
			{
				elements[i] = elements[cmp_id];
				i = cmp_id;
			}
			else
				break;
		}
		elements[i] = elem;
		return return_value;
	}

	__device__ __host__ float top()
	{
		if (!empty())
			return elements[1];
	}

	__device__ __host__ int size() const { return numElems; }
	__device__ __host__ bool empty() const
	{
		return numElems == 1;
	}
	__device__ __host__ bool full() const
	{
		return numElems == max_size;
	}

	__device__ __host__ void set_cmp_func(bool(*compare)(const float &, const float &))
	{
		cmp_func = compare;
	}
	__device__ __host__ float sum() const
	{
		float sum = 0.0f;
		for (int i = 1; i < numElems; ++i)
			sum += elements[i];
		return sum;
	}

	friend class Max_heap;
	friend class Min_heap;
};

//__device__ __host__ bool larger(const float &t1, const float &t2);
inline __device__ __host__ bool larger(const float &t1, const float &t2)
{
	return t1 > t2;
}


class Max_heap {
private:
	Heap heap;
public:
	__device__ __host__ Max_heap(int size, float *element_entry) :heap(size, element_entry)
	{
		heap.set_cmp_func(larger);
	}
	//~Max_heap()
	//{
	//	heap.~Heap();
	//}
	__device__ __host__ bool insert(float elem)
	{
		return heap.insert(elem);
	}
	__device__ __host__ float delete_t()
	{
		return heap.delete_t();
	}
	__device__ __host__ float delete_and_insert(float elem)
	{
		return heap.delete_and_insert(elem);
	}
	__device__ __host__ float top()
	{
		return heap.top();
	}
	__device__ __host__ int size() const { return heap.size(); }
	__device__ __host__ bool empty() const { return heap.empty(); }
	__device__ __host__ bool full() const { return heap.full(); }
	__device__ __host__ float sum() const { return heap.sum(); }
};

//__device__ __host__ bool smaller(const float &t1, const float &t2);
inline __device__ __host__ bool smaller(const float &t1, const float &t2)
{
	return t1 < t2;
}

class Min_heap {
private:
	Heap heap;
public:
	__device__ __host__ Min_heap(int size, float *element_entry) :heap(size, element_entry)
	{
		heap.set_cmp_func(smaller);
	}
	//~Min_heap()
	//{
	//	heap.~Heap();
	//}
	__device__ __host__ bool insert(float elem)
	{
		return heap.insert(elem);
	}
	__device__ __host__ float delete_t()
	{
		return heap.delete_t();
	}
	__device__ __host__ float delete_and_insert(float elem)
	{
		return heap.delete_and_insert(elem);
	}
	__device__ __host__ float top()
	{
		return heap.top();
	}
	__device__ __host__ int size() const { return heap.size(); }
	__device__ __host__ bool empty() const { return heap.empty(); }
	__device__ __host__ bool full() const { return heap.full(); }
	__device__ __host__ float sum() const { return heap.sum(); }
};
