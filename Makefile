BIN       ?= ./bin
BINARIES  ?= ${BIN}/bfs ${BIN}/sssp ${BIN}/pr ${BIN}/bs ${BIN}/fft ${BIN}/spgemm ${BIN}/sgemm ${BIN}/aes
CUDA_PATH ?= /usr/local/cuda

.PHONY: all clean 

all: ${BINARIES} #Jacobi  HE BH DMR

./${BIN}:
	mkdir -p ${BIN}

${BIN}/bfs: | ./${BIN}
	cd gunrock; \
	mkdir -p build; \
	cd build; \
	cmake .. &&	$(MAKE) -j bfs; \
	cp bin/bfs ../../${BIN}/

${BIN}/sssp: | ./${BIN}
	cd gunrock; \
	mkdir -p build; \
	cd build; \
	cmake .. &&	$(MAKE) -j sssp;

${BIN}/pr: | ./${BIN}
	cd gunrock; \
	mkdir -p build; \
	cd build; \
	cmake .. &&	$(MAKE) -j pr;

${BIN}/bs: | ./${BIN}
	cd BlackScholes; \
	$(MAKE); \
	cp BlackScholes ../${BIN}/bs

${BIN}/fft: | ./${BIN}
	cd gpu-fft; \
	$(MAKE); \
	cp fft ../${BIN}/fft
	
${BIN}/spgemm: ${CUDA_PATH}/include/cusp | ./${BIN}
	cd SpGEMM_cuda;\
	$(MAKE); \
	cp spgemm ../${BIN}/spgemm
	
${BIN}/sgemm: | ./${BIN}
	cd gpu-app-collection/src/cuda/cutlass-bench; \
	export CUDACXX=${CUDA_PATH}/bin/nvcc; \
	mkdir -p build && cd build; \
	cmake .. -DUSE_GPGPUSIM=1 -DCUTLASS_NVCC_ARCHS=70 && make cutlass_perf_test; \
	cd tools/test/perf && ln -s ../../../../binary.sh .; ./binary.sh; \
	cp cutlass_perf_test ../../../../../${BIN}/sgemm;

${BIN}/aes: | ./${BIN}
	cd gpu-app-collection/src/cuda/ispass-2009/AES/; \
	make; \
	cp ../../bin/linux/release/ispass-2009-AES ../../../../../${BIN}/

${BIN}/sw: | ./${BIN}
	cd GPU-BSW/; \
	mkdir -p build; \ 
	cd build; \
	cmake CMAKE_BUILD_TYPE=Release ..; \
	$(MAKE); \
	cp ./program_gpu ../${BIN}/sw

${CUDA_PATH}/include/cusp: cusp/
	cp -r ./cusp ${CUDA_PATH}/include/

	
clean: 
	rm -r ${BIN}
