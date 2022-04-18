BIN       ?= ./bin
BINARIES  ?= graphit_apps ${BIN}/bs ${BIN}/fft ${BIN}/spgemm ${BIN}/sgemm ${BIN}/aes ${BIN}/bh ${BIN}/sw ${BIN}/dmr
CUDA_PATH ?= /usr/local/cuda-11/
NVCC      ?= ${CUDA_PATH}/bin/nvcc

.PHONY: all clean ${BINARIES}

all: ${BINARIES} #Jacobi  HE

./${BIN}:
	mkdir -p ${BIN}

graphit_apps: | ./${BIN}
	cd graphit; \
	mkdir build; \
	cd build; \
	cmake ..; \
	make -j$(nproc);
	
	cd ./graphit/graphit_eval/g2_cgo2021_eval; \
	python3 gen_table7.py small; \
	cp ./table7_outputs/pr ../../../${BIN}/; \
	cp ./table7_outputs/ds_road ../../../${BIN}/; \
	cp ./table7_outputs/ds_social ../../../${BIN}/; \
	cp ./table7_outputs/bfs_social ../../../${BIN}/; \
	cp ./table7_outputs/bfs_road ../../../${BIN}/;

${BIN}/bfs: | ./${BIN}
	cd gunrock; \
	mkdir -p build; \
	cd build; \
	cmake .. &&	$(MAKE) -j bfs; \
	cp bin/bfs ../../${BIN}/


${BIN}/bfs_chai: | ./${BIN}
	cd chai/CUDA-D/BFS/; \
	make; \
	cp bfs ../../../${BIN}/bfs_chai;

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
	${NVCC} cutlass.cu -I include/ -I tools/util/include/ -cudart=shared -lcudart; \
	cp a.out ../${BIN}/sgemm;

${BIN}/aes: | ./${BIN}
	cd gpu-app-collection/src/cuda/ispass-2009/AES/; \
	make; \
	cp ../../bin/linux/release/ispass-2009-AES ../../../../../${BIN}/aes

${BIN}/bh: | ./${BIN}
	cd Galois/lonestar/scientific/gpu/barneshut; \
	nvcc bh.cu -I./../../../../libgpu/include -I./../../../../external/moderngpu/src -I./../../../../external/cub ./../../../../libgpu/src/csr_graph.cu ./../../../../libgpu/src/ggc_rt.cu  -cudart shared -O3 -DNDEBUG  -arch=sm_70 --expt-extended-lambda -std=c++14 -DTHRUST_IGNORE_CUB_VERSION_CHECK -D_FORCE_INLINES  -lcudadevrt -lcudart -o bh; \
	cp bh ../../../../../${BIN}

${BIN}/dmr: | ./${BIN}
	cd Galois/lonestar/scientific/gpu/delaunayrefinement; \
	nvcc dmr.cu -I./../../../../libgpu/include -I./../../../../external/moderngpu/src -I./../../../../external/cub ./../../../../libgpu/src/csr_graph.cu ./../../../../libgpu/src/ggc_rt.cu  -cudart shared -O3 -DNDEBUG  -arch=sm_70 --expt-extended-lambda -std=c++14 -DTHRUST_IGNORE_CUB_VERSION_CHECK -D_FORCE_INLINES  -lcudadevrt -lcudart -o dmr; \
	cp dmr ../../../../../${BIN}


${BIN}/sw: | ./${BIN}
	cd GASAL2; \
	${NVCC} src/*.cpp src/*.cu test_prog/*.cpp submodules/alignment_boilerplate/src/* -I include/ -I submodules/alignment_boilerplate/include -I src/ -cudart=shared -lcudart -DN_CODE=0x4E -DMAX_QUERY_LEN=1024 -lpthread -Xcompiler -fopenmp -o program_gpu; \
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
