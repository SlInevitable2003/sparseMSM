#include <iostream>
#include <cassert>
#include "ec/bigint.hpp"
#include "ec/alt_bn128.hpp"
#include "ec_cuda/bigint.cuh"
#include "ec_cuda/alt_bn128.cuh"
using namespace std;

#define whatis(x) cerr << #x << " = " << x << endl
#define whatare(pr) cerr << #pr << " = (" << pr.first << ", " << pr.second << ")" << endl

#define SEED 0

Bit randBit() { return Bit(rand() & 1); }
Int randInt() { Int ret = 0; for (int i = 0; i < 3; i++) ret = (ret << 31) | rand(); return ret; }
template<size_t t> Bigint<t> randBigint() { Bigint<t> ret; for (int i = 0; i < t; i++) ret.data[i] = randInt(); return ret; }
alt_bn128_Fp rand_alt_bn128_Fp() 
{
    Bigint<4> ret = randBigint<4>();
    while (larger_or_eq(ret.data, alt_bn128::pri.data, 4)) ret = randBigint<4>();
    return {ret};
}

const int n = 1 << 4;
alt_bn128_Fp A[n][n], B[n][n], C[n][n], D[n][n]; // total 3 * n * n * 4 * 8B = 24KB

void param_setup()
{
    cuda::Bigint<4> *pri_dev, *R_square_dev;
    Int *pri_inv_dev;
    cuda::alt_bn128_Fp* one_dev, *zero_dev;
    cudaMalloc((void **)&pri_dev, sizeof(cuda::Bigint<4>));
    cudaMalloc((void **)&R_square_dev, sizeof(cuda::Bigint<4>));
    cudaMalloc((void **)&pri_inv_dev, sizeof(Int));
    cudaMalloc((void **)&one_dev, sizeof(cuda::alt_bn128_Fp));
    cudaMalloc((void **)&zero_dev, sizeof(cuda::alt_bn128_Fp));

    cudaMemcpy((void *)pri_dev, &alt_bn128::pri, sizeof(Bigint<4>), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)R_square_dev, &alt_bn128::R_square, sizeof(Bigint<4>), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)pri_inv_dev, &alt_bn128::pri_inv, sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)one_dev, &alt_bn128::one, sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)zero_dev, &alt_bn128::zero, sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);

    cuda::alt_bn128::param_setup<<<1, 1>>>(pri_dev, R_square_dev, pri_inv_dev, one_dev, zero_dev);

    cudaFree(pri_dev);
    cudaFree(R_square_dev);
    cudaFree(pri_inv_dev);
    cudaFree(one_dev);
    cudaFree(zero_dev);
}

__global__ void matmul(cuda::alt_bn128_Fp *A, cuda::alt_bn128_Fp *B, cuda::alt_bn128_Fp *D, const int n)
{
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) D[i * n + j] = D[i * n + j] + (A[i * n + k] * B[k * n + j]);
        D[i * n + j].mont_unrepr();
    }
}

int main(int argc, char *argv[])
{
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        A[i][j] = rand_alt_bn128_Fp(), B[i][j] = rand_alt_bn128_Fp(), C[i][j] = alt_bn128::zero, D[i][j] = alt_bn128::zero;
        A[i][j].mont_repr(), B[i][j].mont_repr();
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) C[i][j] = C[i][j] + (A[i][k] * B[k][j]);
        C[i][j].mont_unrepr();
    }

    cuda::alt_bn128_Fp *dev_A, *dev_B, *dev_D;
    cudaMalloc((void **)&dev_A, n * n * sizeof(alt_bn128_Fp));
    cudaMalloc((void **)&dev_B, n * n * sizeof(alt_bn128_Fp));
    cudaMalloc((void **)&dev_D, n * n * sizeof(alt_bn128_Fp));

    cudaMemcpy((void *)dev_A, &A[0][0], n * n * sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dev_B, &B[0][0], n * n * sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dev_D, &D[0][0], n * n * sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);

    matmul<<<1, 1>>>(dev_A, dev_B, dev_D, n);

    cudaMemcpy(&D[0][0], (void *)dev_D, n * n * sizeof(alt_bn128_Fp), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) D[i][j].print_hex();

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_D);

    return 0;
}