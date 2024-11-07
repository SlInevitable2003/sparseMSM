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

const int log2n = 10;
const int n = 1 << log2n;
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

#define LOG2_BLK_SIZ 5
#define BLK_SIZ (1 << 5)

__global__ void native_msm_step1(cuda::alt_bn128_Fp *coeff, cuda::alt_bn128_EC *point)
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
}
__global__ void native_msm_step2(cuda::alt_bn128_Fp *coeff, cuda::alt_bn128_EC *point, const int log2n)
{
    __shared__ cuda::alt_bn128_EC share[BLK_SIZ];
    share[threadIdx.x] = point[threadIdx.x * BLK_SIZ];
    __syncthreads();

    for (int depth = 0; depth < log2n - LOG2_BLK_SIZ; depth++) {
        cuda::alt_bn128_EC t = share[threadIdx.x] + share[threadIdx.x ^ (1 << depth)];
        __syncthreads();
        share[threadIdx.x] = t;
        __syncthreads();
    }
    point[0] = share[threadIdx.x];
}
void native_msm(cuda::alt_bn128_Fp *coeff_dev, cuda::alt_bn128_EC *point_dev, const int log2n)
{
    native_msm_step1<<<n / BLK_SIZ, BLK_SIZ>>>(coeff_dev, point_dev);
    cudaDeviceSynchronize();
    native_msm_step2<<<1, BLK_SIZ>>>(coeff_dev, point_dev, log2n);
}

extern void instance_generating(int *randomness, alt_bn128_Fp *coeff, alt_bn128_EC *point, int n);

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

    instance_generating(randomness, coeff, point, n);

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

    cout << "native gpu-MSM applying ..." << endl;

    alt_bn128_EC gpu_sum;
    cuda::alt_bn128_Fp *coeff_dev;
    cuda::alt_bn128_EC *point_dev;
    cudaMalloc((void **)&coeff_dev, n * sizeof(cuda::alt_bn128_Fp));
    cudaMalloc((void **)&point_dev, n * sizeof(cuda::alt_bn128_EC));

    cudaMemcpy((void *)coeff_dev, &coeff[0], n * sizeof(alt_bn128_Fp), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)point_dev, &point[0], n * sizeof(alt_bn128_EC), cudaMemcpyHostToDevice);

    native_msm(coeff_dev, point_dev, log2n);

    cudaMemcpy((void *)&gpu_sum, (void *)point_dev, sizeof(cuda::alt_bn128_EC), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "native gpu-MSM applying done." << endl;

    // for (int i = 0; i < n - 1; i++) assert(point[i] == point[i + 1]);

    gpu_sum.mont_unrepr();
    gpu_sum.print_hex();

    cout << ((sum == gpu_sum) ? "sum == gpu_sum" : "sum != gpu_sum") << endl;

    return 0;
}