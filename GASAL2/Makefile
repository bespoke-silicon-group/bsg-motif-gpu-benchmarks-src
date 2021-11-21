all:
	nvcc src/*.cpp src/*.cu test_prog/*.cpp submodules/alignment_boilerplate/src/* -I include/ -I submodules/alignment_boilerplate/include -I src/ -cudart=shared -lcudart -DN_CODE=0x4E -DMAX_QUERY_LEN=1024 -lpthread -Xcompiler -fopenmp
