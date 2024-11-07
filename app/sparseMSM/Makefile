lib := ec/alt_bn128.cpp ec/bigint.cpp
lib_cu := ec_cuda/alt_bn128.cu ec_cuda/bigint.cu
helper := instance_gen.cpp
compile_options := -rdc=true -Xcompiler -fopenmp

sparseMSM: sparseMSM.cu $(lib) $(lib_cu) $(helper)
	nvcc -o sparseMSM sparseMSM.cu $(lib) $(lib_cu) $(helper) $(compile_options)