/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <fstream>
#include <x86intrin.h>
////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int  NUM_ITERATIONS = 1;


#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

void flushc(void *addr, long size)
{
	long k = 0;
	for(auto i = (long)addr - ((long)addr) % 64; k < size; k += 64)
		_mm_clflush((void*)((char*)addr + k));
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	if(argc != 2) {
		printf("Format: ./BlackScholes input_file.in\n");
		return -1;
	}
    // Start logs
    //printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears,
    *h_RiskFree,
    *h_Volatility;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears,
    *d_RiskFree,
    *d_Volatility;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    //findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    //printf("Initializing data...\n");
    //std::ifstream input(argv[1]);
    unsigned OPT_N;
    //input >> OPT_N;    
    OPT_N=atoi(argv[1]);
    unsigned OPT_SZ = OPT_N * sizeof(float);
    
    
    //printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);
    h_RiskFree      = (float *)malloc(OPT_SZ);
    h_Volatility    = (float *)malloc(OPT_SZ);

    //printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_RiskFree,     OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_Volatility,   OPT_SZ));

    //printf("...generating input data in CPU mem.\n");
    //srand(5347);

    // Read options set
    for (i = 0; i < OPT_N; i++)
    {
        float ignore;
        char type;
        //input >> h_StockPrice[i] >> h_OptionStrike[i] >> h_RiskFree[i] >> 
        //	ignore >> h_Volatility[i] >> h_OptionYears[i] >> type;
        h_StockPrice[i] = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
        h_OptionYears[i] = RandFloat(0.25f, 10.0f);	
        h_RiskFree[i] = 0.02f;
	h_Volatility[i] = 0.30f;
	/*switch(type) {
        	case 'C':
        		input >> ignore >> h_CallResultCPU[i];
        		h_PutResultCPU[i]  = -1.0f;
        		break;
        	case 'P':
        		input >> ignore >> h_PutResultCPU[i];
        		h_CallResultCPU[i]  = -1.0f;
        		break;
        }*/
    }
 printf("%i\t", 2 * OPT_N);    double ttGPU = gpuTime;
    //printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    double total_time = 0;
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
	
    sdkResetTimer(&hTimer);
    //asm("wbinvd" ::: "memory");
    flushc(h_StockPrice, OPT_SZ);
    flushc(h_OptionStrike, OPT_SZ);
    flushc(h_OptionYears, OPT_SZ);
    flushc(h_RiskFree, OPT_SZ);
    flushc(h_Volatility, OPT_SZ);
    sdkStartTimer(&hTimer);
    //printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,   h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_RiskFree,     h_RiskFree,     OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Volatility,   h_Volatility,   OPT_SZ, cudaMemcpyHostToDevice));
    //printf("Data init done.\n\n");
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("%f\t", gpuTime);
	ttGPU = gpuTime;
	sdkResetTimer(&hTimer);
    	sdkStartTimer(&hTimer);
	if(i % 2 == 0)
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            (float2 *)d_RiskFree,
            (float2 *)d_Volatility,
            OPT_N
        );
	else
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            (float2 *)d_RiskFree,
            (float2 *)d_Volatility,
            OPT_N
        );
 
	checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        gpuTime = sdkGetTimerValue(&hTimer); /// NUM_ITERATIONS;
        printf("\t%f\t", gpuTime);
	total_time = gpuTime;
    	getLastCudaError("BlackScholesGPU() execution failed\n");
sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    //printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));

sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("\t%f\t%f\n\n", gpuTime, ttGPU + gpuTime + total_time);


    }
	//printf("Exec time:\t%f\n", total_time / NUM_ITERATIONS);
    //Both call and put is calculated
    //printf("Options count             : %i     \n", 2 * OPT_N);
    //printf("%f\n", gpuTime);
    //printf("Effective memory bandwidth: %f GB/s\n", ((double)(7 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    //printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    //printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
//           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);
    //printf("Checking the results...\n");
    //printf("...running CPU calculations.\n\n");

    //printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;
/*
    for (i = 0; i < OPT_N; i++) {
    	if(h_CallResultCPU[i] >= 0) {
		    ref   = h_CallResultCPU[i];
		    delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);
		    
		}
		else {
		    ref   = h_PutResultCPU[i];
		    delta = fabs(h_PutResultCPU[i] - h_PutResultGPU[i]);			
		}
		
		if (delta > max_delta) {
			max_delta = delta;
		}
        
        sum_delta += delta;
        sum_ref   += fabs(ref);
    }
*/
    L1norm = sum_delta / sum_ref;
    /*printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
    */
    return 0;
}
