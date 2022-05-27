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
	float2* buffer_input = (float2*)malloc((uint64_t)sizeof(float2) * num * num_execs);
	float2* buffer_output = (float2*)malloc((uint64_t)sizeof(float2) * num * num_execs);
	
	if (!buffer_input) 
		return VKFFT_ERROR_MALLOC_FAILED;

	for(int i = 0; i < num_execs; ++i) {
		for (uint64_t j = 0; j < num; j++) {
			//buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
			buffer_input[j + i * num].x = j + 1;
			buffer_input[j + i * num].y = 0;
		}
	}
	
	int i = 0;
	//Configuration + FFT application .
	VkFFTConfiguration *configuration = (VkFFTConfiguration *)malloc(sizeof(VkFFTConfiguration));
	VkFFTApplication *app = (VkFFTApplication *)malloc(sizeof(VkFFTApplication));
	*configuration = {};
	*app = {};
	//FFT + iFFT sample code.
	//Setting up FFT configuration for forward and inverse FFT.
	configuration[i].FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration[i].size[0] = num;
	configuration[i].size[1] = num_execs;
	
	configuration[i].device = &vkGPU->device;
	//Allocate buffer for the input data.
	uint64_t bufferSize = (uint64_t)sizeof(float2) * configuration[i].size[0] * configuration[i].size[1];
	configuration[i].isOutputFormatted = 1;
	configuration[i].bufferSize = &bufferSize;
	configuration[i].omitDimension[0] = 1;
	
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

	//Submit FFT+iFFT.
	uint64_t num_iter = 1;
	double totTime = 0;

	VkFFTLaunchParams *launchParams = (VkFFTLaunchParams*)malloc(sizeof(VkFFTLaunchParams));
	res = cudaSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	*launchParams = {};

	cuFloatComplex *buffer = 0;
	res = cudaMalloc((void**)&buffer, bufferSize);
	if (res != cudaSuccess) 
		return VKFFT_ERROR_FAILED_TO_ALLOCATE;
	
	res = cudaMemcpy(buffer, buffer_input, bufferSize, cudaMemcpyHostToDevice);
	if (res != cudaSuccess) 
		return VKFFT_ERROR_FAILED_TO_COPY;
	launchParams->buffer = (void**)&buffer;
	
	cuFloatComplex *buffer2;
	res = cudaMalloc((void**)&buffer2, bufferSize);
	launchParams->outputBuffer = (void**)&buffer2;
	
	resFFT = VkFFTAppend(app, -1, launchParams);
	if (resFFT != VKFFT_SUCCESS) 
		return resFFT;
	
	if (cudaDeviceSynchronize() != cudaSuccess) 
		return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	
	cudaMemcpy(buffer_output, buffer2, sizeof(float2) * num * num_execs, cudaMemcpyDeviceToHost);
	for(int j = 0; j < num_execs; ++j) {
		for(int i = 0; i < num; ++i)
			out << "(" << buffer_output[i + j * num].x << "," << buffer_output[i + j * num].y << ") ";
		out << "\n";
	}
	
	printf("VkFFT - Size: %" PRIu64 ", Batches: %" PRIu64 ", num_iter: %" PRIu64 ", num streams: %d, Using LUT? %d\n", configuration[0].size[0], configuration[0].numberBatches, num_iter, num_execs, useLUT);
	return resFFT;
}

int main(int argc, char *argv[]) {
	uint64_t num = 64;
	int streams = 16384;
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
