#pragma once
#include "bigint.cuh"

namespace cuda {
    namespace alt_bn128 {
        __device__ extern Bigint<4> pri, R_square;
        __device__ extern Int pri_inv;
    }

    class alt_bn128_Fp {
    public:
        Bigint<4> mont;
        alt_bn128_Fp() = default;
        __device__ alt_bn128_Fp(Bigint<4> data);

        __device__ void mont_repr();
        __device__ void mont_unrepr();

        // alt_bn128_Fp inverse() const;

        __device__ alt_bn128_Fp operator+(const alt_bn128_Fp& other) const;
        __device__ alt_bn128_Fp operator-(const alt_bn128_Fp& other) const;
        __device__ alt_bn128_Fp operator*(const alt_bn128_Fp& other) const;

        __device__ alt_bn128_Fp square() const;

        __device__ bool operator==(const alt_bn128_Fp& other) const;
    };

    namespace alt_bn128 {
        __device__ extern alt_bn128_Fp zero, one;
    }

    class alt_bn128_EC { 
        // y^2 = x^3 + 3
        // refer to https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html for more details
    public:
        alt_bn128_Fp X, Y, Z;
        alt_bn128_EC() = default;
        __device__ alt_bn128_EC(alt_bn128_Fp X, alt_bn128_Fp Y, alt_bn128_Fp Z);

        __device__ void mont_repr() { X.mont_repr(), Y.mont_repr(), Z.mont_repr(); }
        __device__ void mont_unrepr() { X.mont_unrepr(), Y.mont_unrepr(), Z.mont_unrepr(); }

        __device__ alt_bn128_EC operator+(const alt_bn128_EC& other) const;
        __device__ alt_bn128_EC doubling() const;

        __device__ bool operator==(const alt_bn128_EC& other) const;

        __device__ alt_bn128_EC native_scale(const alt_bn128_Fp& scalar) const;
    };

    namespace alt_bn128 {
        __device__ extern alt_bn128_EC infty, gentor;
        __global__ void param_setup(Bigint<4> *pri_dev, 
                                    Bigint<4> *R_square_dev, 
                                    Int *pri_inv_dev, 
                                    alt_bn128_Fp* one_dev, 
                                    alt_bn128_Fp* zero_dev,
                                    alt_bn128_EC* infty_dev,
                                    alt_bn128_EC* gentor_dev);
    }
}