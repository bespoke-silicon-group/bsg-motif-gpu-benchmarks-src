static: coder_0_1.cu coder_1_1.cu coder_1_0.cu coder_0_0.cu
	nvcc sample_0_benchmark_VkFFT_single.cu -DSTATIC_COMPILE -DVKFFT_BACKEND=1 utils_VkFFT.cu coder_0_1.cu coder_1_1.cu coder_1_0.cu coder_0_0.cu -lnvrtc -lcuda -o fft -cudart=shared
	
dynamic:
	nvcc sample_0_benchmark_VkFFT_single.cu  -DVKFFT_BACKEND=1 utils_VkFFT.cu -lnvrtc -lcuda -o fft -cudart=shared
