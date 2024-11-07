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
alt_bn128_EC rand_alt_bn128_EC() { return alt_bn128::gentor.native_scale(rand_alt_bn128_Fp(), true); }

const int n = 1 << 4;
alt_bn128_Fp coeff[n];
alt_bn128_EC point[n];

int randomness[n];

void param_setup()
{
    cuda::Bigint<4> *pri_dev, *R_square_dev;
    Int *pri_inv_dev;
    cuda::alt_bn128_Fp* one_dev, *zero_dev;
    cuda::alt_bn128_EC* infty_dev, *gentor_dev;
    cudaMalloc((void **)&pri_dev, sizeof(cuda::Bigint<4>));
    cudaMalloc((void **)&R_square_dev, sizeof(cuda::Bigint<4>));
    cudaMalloc((void **)&pri_inv_dev, sizeof(Int));
    cudaMalloc((void **)&one_dev, sizeof(cuda::alt_bn128_Fp));
    cudaMalloc((void **)&zero_dev, sizeof(cuda::alt_bn128_Fp));
    cudaMalloc((void **)&infty_dev, sizeof(cuda::alt_bn128_EC));
    cudaMalloc((void **)&gentor_dev, sizeof(cuda::alt_bn128_EC));

    cudaMemcpy((void *)pri_dev, &alt_bn128::pri, sizeof(Bigint<4>), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)R_square_dev, &alt_bn128::R_square, sizeof(Bigint<4>), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)pri_inv_dev, &alt_bn128::pri_inv, sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)one_dev, &alt_bn128::one, sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)zero_dev, &alt_bn128::zero, sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)infty_dev, &alt_bn128::infty, sizeof(alt_bn128_EC), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)gentor_dev, &alt_bn128::gentor, sizeof(alt_bn128_EC), cudaMemcpyHostToDevice);

    cuda::alt_bn128::param_setup<<<1, 1>>>(pri_dev, R_square_dev, pri_inv_dev, one_dev, zero_dev, infty_dev, gentor_dev);

    cudaFree(pri_dev);
    cudaFree(R_square_dev);
    cudaFree(pri_inv_dev);
    cudaFree(one_dev);
    cudaFree(zero_dev);
    cudaFree(infty_dev);
    cudaFree(gentor_dev);
}

const int zero_bound = 5;

#define LOG2_BLK_SIZ 2
#define BLK_SIZ (1 << 2)

__global__ void sparse_msm(cuda::alt_bn128_Fp *coeff, cuda::alt_bn128_EC *point, const int log2n)
{
    int idx = blockIdx.x * BLK_SIZ + threadIdx.x;
    __shared__ cuda::alt_bn128_EC share[BLK_SIZ];
    share[threadIdx.x] = point[idx].native_scale(coeff[idx]);
    __syncthreads();

    for (int depth = 0; depth < LOG2_BLK_SIZ; depth++) {
        cuda::alt_bn128_EC t = share[threadIdx.x] + share[threadIdx.x ^ (1 << depth)];
        __syncthreads();
        share[threadIdx.x] = t;
        __syncthreads();
    }
    point[idx] = share[threadIdx.x];

    for (int depth = LOG2_BLK_SIZ; depth < log2n; depth++) {
        cuda::alt_bn128_EC t = point[idx] + point[idx ^ (1 << depth)];
        __syncthreads();
        point[idx] = t;
        __syncthreads();
    }
}

// __global__ void sparse_msm(cuda::alt_bn128_Fp *coeff, cuda::alt_bn128_EC *point, const int n)
// {
//     cuda::alt_bn128_EC buf[5][16];
//     for (int j = 0; j < 16; j++) buf[0][j] = point[j].native_scale(coeff[j]);
//     for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) buf[i + 1][j] = buf[i][j] + buf[i][j ^ (1 << i)];
//     point[0] = buf[4][0];
// }

int main(int argc, char *argv[])
{
    param_setup();

    cout << "randomness sampling..." << endl;

    int sparse_bound = int(0.9 * n);
    for (int i = 0; i < n; i++) {
        if (i < sparse_bound) {
            int r = rand() % 10;
            if (r < zero_bound) randomness[i] = 0;
            else randomness[i] = 1;
        } else randomness[i] = 2;
    }

    cout << "randomness sampling done." << endl;
    cout << "MSM instance generating..." << endl;

    for (int i = 0; i < n; i++) {
        switch (randomness[i]) {
            case 0: { coeff[i] = alt_bn128::zero; } break;
            case 1: { coeff[i] = alt_bn128::one; } break;
            case 2: { coeff[i] = rand_alt_bn128_Fp(); }
        }
        point[i] = rand_alt_bn128_EC();
    }

    cout << "MSM instance generating done." << endl;
    cout << "native MSM applying..." << endl;

    alt_bn128_EC sum = alt_bn128::infty; sum.mont_repr();
    for (int i = 0; i < n; i++) { 
        alt_bn128_EC inc = point[i]; inc.mont_repr();
        for (int j = 0; j < 4; j++) {
            Int ele = coeff[i].mont.data[j];
            for (int k = 0; k < WORD; k++) {
                if (ele & 1) sum = sum + inc;
                inc = inc.doubling();
                ele >>= 1;
            }
        }
    }

    cout << "native MSM applying done." << endl;

    sum.mont_unrepr();
    sum.print_hex();

    // alt_bn128_EC buf[5][16];
    // for (int j = 0; j < 16; j++) buf[0][j] = point[j].native_scale(coeff[j]);
    // for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) buf[i + 1][j] = buf[i][j] + buf[i][j ^ (1 << i)];
    // auto sum2 = buf[4][0];
    // sum2.mont_unrepr();
    // sum2.print_hex();

    cout << "native gpu-MSM applying ..." << endl;

    alt_bn128_EC gpu_sum;
    cuda::alt_bn128_Fp *coeff_dev;
    cuda::alt_bn128_EC *point_dev;
    cudaMalloc((void **)&coeff_dev, n * sizeof(cuda::alt_bn128_Fp));
    cudaMalloc((void **)&point_dev, n * sizeof(cuda::alt_bn128_EC));

    cudaMemcpy((void *)coeff_dev, &coeff[0], n * sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)point_dev, &point[0], n * sizeof(alt_bn128_EC), cudaMemcpyHostToDevice);

    sparse_msm<<<n / BLK_SIZ, BLK_SIZ>>>(coeff_dev, point_dev, 4);
    // sparse_msm<<<1, 1>>>(coeff_dev, point_dev, n);
    cudaDeviceSynchronize();

    cudaMemcpy((void *)&gpu_sum, (void *)point_dev, sizeof(cuda::alt_bn128_EC), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "native gpu-MSM applying done." << endl;

    // for (int i = 0; i < n - 1; i++) assert(point[i] == point[i + 1]);

    gpu_sum.mont_unrepr();
    gpu_sum.print_hex();

    cout << ((sum == gpu_sum) ? "sum == gpu_sum" : "sum != gpu_sum") << endl;

    return 0;
}