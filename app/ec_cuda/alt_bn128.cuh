#pragma once
#include "bigint.cuh"

namespace cuda {
    namespace alt_bn128 {
        __device__ extern Bigint<4> pri, R_square;
        __device__ extern Int pri_inv;
    }

    class alt_bn128_Fp {
        Bigint<4> mont;
    public:
        alt_bn128_Fp() = default;
        __device__ alt_bn128_Fp(Bigint<4> data);

        __device__ void mont_repr();
        __device__ void mont_unrepr();

        // alt_bn128_Fp inverse() const;

        __device__ alt_bn128_Fp operator+(const alt_bn128_Fp& other) const;
        __device__ alt_bn128_Fp operator-(const alt_bn128_Fp& other) const;
        __device__ alt_bn128_Fp operator*(const alt_bn128_Fp& other) const;
    };

    namespace alt_bn128 {
        __device__ extern alt_bn128_Fp zero, one;

        __global__ void param_setup(Bigint<4> *pri_dev, Bigint<4> *R_square_dev, Int *pri_inv_dev, alt_bn128_Fp* one_dev, alt_bn128_Fp* zero_dev);
    }
}