FFT_SIZE?=16384
NUM?=64

NVCC = /usr/local/cuda-11/bin/nvcc

static: coder_0_1_${FFT_SIZE}_${NUM}.cu coder_1_1_${FFT_SIZE}_${NUM}.cu coder_1_0_${FFT_SIZE}_${NUM}.cu coder_0_0_${FFT_SIZE}_${NUM}.cu
	${NVCC} sample_0_benchmark_VkFFT_single.cu -DSTATIC_COMPILE -DVKFFT_BACKEND=1 utils_VkFFT.cu $^ -lnvrtc -lcuda -o fft_${FFT_SIZE}_${NUM} -cudart=shared -DFFT_SIZE=${FFT_SIZE} -DFFT_DIM=${NUM}
	
dynamic:
	${NVCC} sample_0_benchmark_VkFFT_single.cu  -DVKFFT_BACKEND=1 utils_VkFFT.cu -lnvrtc -lcuda -o fft_${FFT_SIZE}_${NUM} -cudart=shared -DFFT_SIZE=${FFT_SIZE} -DFFT_DIM=${NUM}
