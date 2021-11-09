//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include "vkFFT.h"
#include "utils_VkFFT.h"

VkFFTResult run_FFT_benchmark(VkGPU* vkGPU, uint64_t num, int useLUT = 1)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
	cudaError_t res = cudaSuccess;
	//printf("0 - VkFFT FFT + iFFT C2C benchmark 1D batched in single precision. Size: %lu, using LUT? %d\n", num, useLUT);
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, 27));
	
	if (!buffer_input) 
		return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
		buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
	}
	
	for (uint64_t n = 0; n < 2; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = num;
			configuration.numberBatches = 1;
			
			configuration.device = &vkGPU->device;
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;
			cuFloatComplex* buffer = 0;
			res = cudaMalloc((void**)&buffer, bufferSize);
			if (res != cudaSuccess) 
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;

			configuration.bufferSize = &bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) 
				return VKFFT_ERROR_FAILED_TO_COPY;
			
			configuration.useLUT = useLUT;
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) 
				return resFFT;

			//Submit FFT+iFFT.
			uint64_t num_iter = 1;
			double totTime = 0;

			VkFFTLaunchParams launchParams = {};
			resFFT = performVulkanFFTiFFT(vkGPU, &app, &launchParams, num_iter, &totTime);
			if (resFFT != VKFFT_SUCCESS) 
				return resFFT;
			run_time[r] = totTime;
			if (n > 0) {
				if (r == num_runs - 1) {
					double std_error = 0;
					double avg_time = 0;
					for (uint64_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
					}
					avg_time /= num_runs;
					for (uint64_t t = 0; t < num_runs; t++) {
						std_error += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					}
					std_error = sqrt(std_error / num_runs);
					uint64_t num_tot_transfers = 0;
					for (uint64_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;

					printf("VkFFT - Size: %" PRIu64 ", Batches: %" PRIu64 ", avg_time_per_step: %0.3f ms, num_iter: %" PRIu64 ", Using LUT? %d\n", configuration.size[0], configuration.numberBatches, avg_time, num_iter, useLUT);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

			cudaFree(buffer);
			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	//benchmark_result /= 25;
	//printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
	return resFFT;
}

int main(int argc, char *argv[]) {
	uint64_t num = 65536;
	int useLUT = 1;
	if(argc >= 2)
		num = atoi(argv[1]);
	if(argc >= 3)
		useLUT = atoi(argv[2]);
	VkGPU vkGPU;
	cuCtxGetCurrent ( &vkGPU.context );
	cuCtxGetDevice ( &vkGPU.device );
	run_FFT_benchmark(&vkGPU, num, useLUT);
	return 0;
}
