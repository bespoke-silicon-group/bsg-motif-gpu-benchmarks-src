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
//#include <ofstream>
#include "utils_VkFFT.h"
int kernelCount = 0;

VkFFTResult run_FFT_benchmark(VkGPU* vkGPU, uint64_t num, std::ofstream &out, int num_execs = 1, int useLUT = 1)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
	cudaError_t res = cudaSuccess;
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float2* buffer_input = (float2*)malloc((uint64_t)sizeof(float2) * (uint64_t)pow(2, 27));
	float2* buffer_output = (float2*)malloc((uint64_t)sizeof(float2) * (uint64_t)pow(2, 27));
	
	if (!buffer_input) 
		return VKFFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < (uint64_t)pow(2, 27); i++) {
		//buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
		buffer_input[i].x = 1;
		buffer_input[i].y = 0;
	}
		
	const int num_runs = 1;
	double run_time[num_runs];
	for (uint64_t r = 0; r < num_runs; r++) {
		//Configuration + FFT application .
		VkFFTConfiguration *configuration = (VkFFTConfiguration *)malloc(num_execs * sizeof(VkFFTConfiguration));
		VkFFTApplication *app = (VkFFTApplication *)malloc(num_execs * sizeof(VkFFTApplication));
		for(int i = 0; i < num_execs; ++i) {
			configuration[i] = {};
			app[i] = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration[i].FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration[i].size[0] = num;
			
			configuration[i].device = &vkGPU->device;
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)sizeof(float2) * configuration[i].size[0];
			printf("%p\n", configuration[i].outputBuffer);
			configuration[i].isOutputFormatted = 1;

			configuration[i].bufferSize = &bufferSize;

			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			//res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
			//if (res != cudaSuccess) 
			//	return VKFFT_ERROR_FAILED_TO_COPY;
			
			configuration[i].num_streams = 1;
			cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
			cudaStreamCreate(stream);
			configuration[i].stream = stream;
			configuration[i].useLUT = useLUT;
			printf("Created stream\n");
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app[i], configuration[i]);
			fflush(stdout);
			if (resFFT != VKFFT_SUCCESS) 
				return resFFT;
		}
		printf("Setup %d configs\n", num_execs);

		//Submit FFT+iFFT.
		uint64_t num_iter = 1;
		double totTime = 0;

		VkFFTLaunchParams *launchParams = (VkFFTLaunchParams*)malloc(sizeof(VkFFTLaunchParams));
		//resFFT = performVulkanFFTiFFT(vkGPU, &app, &launchParams, num_iter, &totTime);
		cudaError_t res = cudaSuccess;
		std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
		for(int j = 0; j < num_execs; ++j) {
			*launchParams = {};
			
			uint64_t bufferSize = (uint64_t)sizeof(float2) * num;
			cuFloatComplex* buffer = 0, *buffer2;
			res = cudaMalloc((void**)&buffer, bufferSize);
			if (res != cudaSuccess) 
				return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			launchParams->buffer = (void**)&buffer;
			res = cudaMalloc((void**)&buffer2, bufferSize);
			launchParams->outputBuffer = (void**)&buffer2;
			res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) 
				return VKFFT_ERROR_FAILED_TO_COPY;
			
			
			resFFT = VkFFTAppend(&app[j], -1, &launchParams[j]);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			
			cudaMemcpy(buffer_output, buffer2, sizeof(float2) * num, cudaMemcpyDeviceToHost);
			for(int i = 0; i < num; ++i)
				out << "(" << buffer_output[i].x << "," << buffer_output[i].y << ") ";
			/*	
			out << " \n";
			resFFT = VkFFTAppend(&app[j], 1, &launchParams[j]);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			
			cudaMemcpy(buffer_output, buffer2, sizeof(float) * num, cudaMemcpyDeviceToHost);
			//for(int i = 0; i < num; ++i)
			//	out << buffer_output[i] << " ";
			*/
		}
		
		res = cudaDeviceSynchronize();
		if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
		std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
		totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
		totTime = totTime / num_iter;
		
		run_time[r] = totTime;
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
			/*uint64_t num_tot_transfers = 0;
			for (uint64_t i = 0; i < configuration.FFTdim; i++)
				num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
			num_tot_transfers *= 4;*/

			printf("VkFFT - Size: %" PRIu64 ", Batches: %" PRIu64 ", avg_time_per_step: %0.3f ms, num_iter: %" PRIu64 ", num streams: %d, Using LUT? %d\n", configuration[0].size[0], configuration[0].numberBatches, avg_time, num_iter, num_execs, useLUT);
			//benchmark_result += ((double)bufferSize / 1024) / avg_time;
		}
/*
		for(int j = 0; j < num_execs; ++j) {
			cudaFree(buffer[j]);
			deleteVkFFT(&app[j]);
		}
*/
	}
	free(buffer_input);
	//benchmark_result /= 25;
	//printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
	return resFFT;
}

int main(int argc, char *argv[]) {
	uint64_t num = 65536 / 4;
	int streams = 1;
	int useLUT = 1;
	std::ofstream out("test.out");
	if(argc >= 2)
		num = atoi(argv[1]);
	if(argc >= 3)
		streams = atoi(argv[2]);
	if(argc >= 4)
		useLUT = atoi(argv[3]);
	VkGPU vkGPU;
	cuCtxGetCurrent ( &vkGPU.context );
	cuCtxGetDevice ( &vkGPU.device );
	VkFFTResult res = run_FFT_benchmark(&vkGPU, num, out, streams, useLUT);
	printf("Result: %d\n", res);
	return 0;
}
