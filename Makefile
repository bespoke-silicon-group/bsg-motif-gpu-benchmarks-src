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
	cd cutlass; \
	nvcc cutlass.cu -I include/ -I tools/util/include/ -cudart=shared -lcudart; \
	cp a.out ../${BIN}/sgemm;

${BIN}/aes: | ./${BIN}
	cd gpu-app-collection/src/cuda/ispass-2009/AES/; \
	make; \
	cp ../../bin/linux/release/ispass-2009-AES ../../../../../${BIN}/aes

${BIN}/sw: | ./${BIN}
	cd GPU-BSW; \
	nvcc -x cu src/driver.cpp evaluation/main.cpp submodules/alignment_boilerplate/src/* -arch=sm_70 -I include/ -I ./submodules/alignment_boilerplate/include/ -cudart=shared -lcudart -lpthread -o program_gpu; \
	cp ./program_gpu ../${BIN}/sw
	
	#mkdir -p build; \
	#cd build; \
	#pwd; \
	#cmake CMAKE_BUILD_TYPE=Release ..; \
	#$(MAKE); \

${CUDA_PATH}/include/cusp: cusp/
	cp -r ./cusp ${CUDA_PATH}/include/

	
clean: 
	rm -r ${BIN}
