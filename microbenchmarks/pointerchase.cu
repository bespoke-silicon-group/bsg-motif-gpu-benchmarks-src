#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <bit>
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a, b) std::chrono::duration_cast<std::chrono::microseconds>(a - b).count()

#define NBLOCKS 80
#define ITERS 2000

class LFSR
{
    unsigned long long htaps[25] = {
        0x0,// Error                                                                                                                                                                                                                                                                                                                                                                                                             
        0x1,
        0x3,
        0x6,
        0xC,
        0x14,
        0x30,
        0x60,
        0xB4,
        0x110,
        0x240,
        0x500,
        0xE08,
        0x1C80,
        0x3802,
        0x6000,
        0xD008,
        0x12000,
        0x20400,
        0x72000,
        0x90000,
        0x140000,
        0x300000,
        0x420000,
        0xE10000};

    int BITS;

public:
	LFSR(int b) { 
		BITS = b; 
	}
    unsigned long long next(unsigned long long input){
    	unsigned long long taps = htaps[BITS];
        unsigned long long bit = __builtin_popcount(input & taps) & 1u;
        return ((input << 1) | (bit)) & ((1 << BITS)-1) ;
    }
};

__global__ void testKernel(uint64_t **ptr_array, uint64_t **ptr_start_array, int num_elems, uint64_t *output) {
	int id = (threadIdx.x + blockIdx.x * blockDim.x) % num_elems;
	volatile uint64_t *ptr;
	
	// Warmup
	#pragma unroll
	for(int i = threadIdx.x; i < num_elems; i += blockDim.x) {
		ptr = (uint64_t*)ptr_array[i];
	}
	__threadfence();
}

__global__ void chaseKernel1(uint64_t **ptr_array, uint64_t **ptr_start_array, int num_elems, uint64_t *output) {
	volatile uint64_t *ptr;
	// Warmup
	#pragma unroll
	for(int i = threadIdx.x; i < num_elems; i += blockDim.x) {
		ptr = (uint64_t*)ptr_array[i];
	}
	__threadfence();
	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	ptr = ptr_start_array[id];
	//uint64_t start = clock64();
	#pragma unroll 1000
	for(int i = 0; i < ITERS; ++i) {
		ptr = (uint64_t*)*ptr;
	}
	__threadfence();
}

__global__ void chaseKernel2(uint64_t **ptr_array, uint64_t **ptr_start_array, int num_elems, uint64_t *output) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	volatile uint64_t *ptr = ptr_start_array[id];
	//uint64_t start = clock64();
	#pragma unroll 1000
	for(int i = 0; i < ITERS; ++i) {
		ptr = (uint64_t*)*ptr;
	}
	__threadfence();
}

void make_array(uint64_t **h_ptr_array, uint64_t **d_ptr_array, uint64_t **h_ptr_start_array, int num_elems, int num_threads, int region_size) {
	
	int num_regions = (num_elems / region_size);
    LFSR rng(log2(num_regions));
	
	unsigned long long loc = 1;
	for(int i = 0; i < num_elems; ++i) {
		// Pick random starting location
		long choice = rng.next(loc);
		loc = (choice % num_regions) * region_size;
		
		int j = i;
		for(; j < i + region_size; ++j) {
			h_ptr_array[j] = (uint64_t *)&d_ptr_array[loc + j - i];
		}
		i = j - 1;
	}
	
	for(int i = 0; i < num_threads; ++i) {
		// Pick random starting location
		long choice = rand();
		loc = (choice % num_regions) * region_size;
		
		int j = i;
		for(; j < i + region_size; ++j) {
			h_ptr_start_array[j] = (uint64_t *)&d_ptr_array[loc + j - i];
		}
		i = j - 1;
	}
}

int main(int argc, char *argv[]) 
{
	if(argc < 4) {
		printf("Format: ./exec array_size array_region_size threads_per_block\n");
		return 1;
	}
	
	size_t data_size = atol(argv[1]);
	int region_size = atoi(argv[2]);
	int nthreads = atoi(argv[3]);
	int nblocks = max(NBLOCKS, NBLOCKS * (nthreads / 1024));
	nthreads = min(nthreads, 1024);
	
	printf("Dsize:\t%lu\nNumIters:\t%d\nRegionSize:\t%d\nNthreads:\t%d\n", data_size, ITERS, region_size, nthreads * nblocks);
	
	uint64_t **h_ptr_array, **h_ptr_start_array;
	h_ptr_array = (uint64_t **)malloc(data_size);
	h_ptr_start_array = (uint64_t **)malloc(nblocks * nthreads * sizeof(uint64_t*));
	
	uint64_t **d_ptr_array, **d_ptr_start_array;
	cudaMalloc((void **)&d_ptr_array, data_size);
	cudaMalloc((void **)&d_ptr_start_array, nblocks * nthreads * sizeof(uint64_t*));
	
	make_array(h_ptr_array, d_ptr_array, h_ptr_start_array, data_size / sizeof(uint64_t *), nblocks * nthreads, region_size);
	
	cudaMemcpy(d_ptr_array, h_ptr_array, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ptr_start_array, h_ptr_start_array, nblocks * nthreads * sizeof(uint64_t*), cudaMemcpyHostToDevice);
	
	uint64_t *d_output;
	cudaMalloc((void **)&d_output, sizeof(uint64_t) * nthreads * nblocks);
	
	auto start = TIME_NOW;
	// For small, need to warmup L1
	if(data_size <= 128 * 1024) {
		printf("Threekernel\n");
		testKernel<<<nblocks, nthreads>>> (d_ptr_array, d_ptr_start_array, data_size / sizeof(uint64_t), d_output);
		testKernel<<<nblocks, nthreads>>> (d_ptr_array, d_ptr_start_array, data_size / sizeof(uint64_t), d_output);
		chaseKernel1<<<nblocks, nthreads>>> (d_ptr_array, d_ptr_start_array, data_size / sizeof(uint64_t), d_output);
		cudaDeviceSynchronize();
	}
	else {
		chaseKernel2<<<nblocks, nthreads>>> (d_ptr_array, d_ptr_start_array, data_size / sizeof(uint64_t), d_output);
		cudaDeviceSynchronize();
	}
	auto end = TIME_NOW;
	printf("Traversals:\t%ld\nTime:\t%ld\n", (long)nthreads * (long)nblocks * (long)ITERS, time_diff(end, start));
	
	return 0;
}
