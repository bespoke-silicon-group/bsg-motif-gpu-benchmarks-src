#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a, b) std::chrono::duration_cast<std::chrono::microseconds>(a - b).count()

#define NBLOCKS 80


__global__ void chaseKernel(int *ptr_array, int num_elems, int iters) {
	int id = (threadIdx.x + blockIdx.x * blockDim.x) % num_elems;
	
	for(int i = 0; i < iters; ++i) {
		id = ptr_array[id];
	}
	
	// Needed to make sure prior loop doesn't get optimized out
	ptr_array[id] = (id + 1) % num_elems;
}

void make_array(int *ptr_array, int num_elems, int region_size) {
	
	int num_regions = (num_elems / region_size);
	
	for(int i = 0; i < num_elems; ++i) {
		// Pick random starting location
		int loc = (rand() % num_regions) * region_size;
		
		int j = i;
		for(; j < i + region_size; ++j) {
			ptr_array[j] = loc + j - i;
		}
		i = j - 1;
	}
}

int main(int argc, char *argv[]) 
{
	if(argc < 5) {
		printf("Format: ./exec array_size iterations array_region_size threads_per_block\n");
		return 1;
	}
	
	size_t data_size = atol(argv[1]);
	int iters = atoi(argv[2]);
	int region_size = atoi(argv[3]);
	int nthreads = atoi(argv[4]);
	
	printf("Dsize:\t%u\nNumIters:\t%d\nRegionSize:\t%d\nNthreads:\t%d\n", data_size, iters, region_size, nthreads * NBLOCKS);
	
	int *h_ptr_array;
	h_ptr_array = (int *)malloc(data_size);
	make_array(h_ptr_array, data_size / sizeof(int), region_size);
	
	int *d_ptr_array;
	cudaMalloc((void **)&d_ptr_array, data_size);
	cudaMemcpy(d_ptr_array, h_ptr_array, data_size, cudaMemcpyHostToDevice);
	
	auto start = TIME_NOW;
	int nblocks = max(NBLOCKS, NBLOCKS * (nthreads / 1024));
	chaseKernel<<<nblocks, nthreads>>> (d_ptr_array, data_size / sizeof(int), iters);
	cudaDeviceSynchronize();
	auto end = TIME_NOW;
	printf("Traversals:\t%ld\nTime:\t%ld\n", (long)nthreads * (long)NBLOCKS * (long)iters, time_diff(end, start));
	
	return 0;
}
