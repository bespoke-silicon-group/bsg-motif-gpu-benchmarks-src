all:
	nvcc sample_0_benchmark_VkFFT_single.cpp  -DVKFFT_BACKEND=1 utils_VkFFT.cpp -lnvrtc -lcuda -o fft -cudart=shared
