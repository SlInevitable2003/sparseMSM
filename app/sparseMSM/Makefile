lib := ec/alt_bn128.cpp ec/bigint.cpp
lib_cu := ec_cuda/alt_bn128.cu ec_cuda/bigint.cu
compile_options := -rdc=true

sparseMSM: sparseMSM.cu $(lib) $(lib_cu)
	nvcc -o sparseMSM sparseMSM.cu $(lib) $(lib_cu) $(compile_options)