#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <bit>
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define time_diff(a, b) std::chrono::duration_cast<std::chrono::microseconds>(a - b).count()

#define NBLOCKS 80
#define ITERS 200

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


__global__ void chaseKernel(uint64_t **ptr_array, int num_elems) {
	int id = (threadIdx.x + blockIdx.x * blockDim.x) % num_elems;
	volatile uint64_t *ptr = ptr_array[id];
	
	#pragma unroll 200
	for(int i = 0; i < ITERS; ++i) {
		ptr = (uint64_t*)*ptr;
	}
	
	// Needed to make sure prior loop doesn't get optimized out
	//ptr_array[id] = (id + 1) % num_elems;
}

void make_array(uint64_t **h_ptr_array, uint64_t **d_ptr_array, int num_elems, int region_size) {
	
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
	
	printf("Dsize:\t%lu\nNumIters:\t%d\nRegionSize:\t%d\nNthreads:\t%d\n", data_size, ITERS, region_size, nthreads * NBLOCKS);
	
	uint64_t **h_ptr_array;
	h_ptr_array = (uint64_t **)malloc(data_size);
	
	uint64_t **d_ptr_array;
	cudaMalloc((void **)&d_ptr_array, data_size);
	
	
	make_array(h_ptr_array, d_ptr_array, data_size / sizeof(uint64_t *), region_size);
	cudaMemcpy(d_ptr_array, h_ptr_array, data_size, cudaMemcpyHostToDevice);
	
	auto start = TIME_NOW;
	int nblocks = max(NBLOCKS, NBLOCKS * (nthreads / 1024));
	chaseKernel<<<nblocks, nthreads>>> (d_ptr_array, data_size / sizeof(uint64_t));
	cudaDeviceSynchronize();
	auto end = TIME_NOW;
	printf("Traversals:\t%ld\nTime:\t%ld\n", (long)nthreads * (long)NBLOCKS * (long)ITERS, time_diff(end, start));
	
	return 0;
}