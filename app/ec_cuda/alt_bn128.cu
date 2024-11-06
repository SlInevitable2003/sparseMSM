#include "alt_bn128.cuh"

namespace cuda {
    namespace alt_bn128 {
        __device__ Bigint<4> pri, R_square;
        __device__ Int pri_inv;
        __device__ alt_bn128_Fp zero, one;

        __global__ void param_setup(Bigint<4> *pri_dev, Bigint<4> *R_square_dev, Int *pri_inv_dev, alt_bn128_Fp* one_dev, alt_bn128_Fp* zero_dev)
        {
            pri = *pri_dev;
            R_square = *R_square_dev;
            pri_inv = *pri_inv_dev;
            one = *one_dev;
            zero = *zero_dev;
        }
    }

    __device__ alt_bn128_Fp::alt_bn128_Fp(Bigint<4> data) { mont = data; }

    __device__ alt_bn128_Fp alt_bn128_Fp::operator+(const alt_bn128_Fp &other) const
    {
        alt_bn128_Fp res;
        big_modadd((Int*)res.mont.data, (Int*)mont.data, (Int*)other.mont.data, (Int*)alt_bn128::pri.data, 4);
        return res;
    }

    __device__ alt_bn128_Fp alt_bn128_Fp::operator-(const alt_bn128_Fp &other) const
    {
        alt_bn128_Fp res;
        big_modsub((Int*)res.mont.data, (Int*)mont.data, (Int*)other.mont.data, (Int*)alt_bn128::pri.data, 4);
        return res;
    }

    __device__ alt_bn128_Fp alt_bn128_Fp::operator*(const alt_bn128_Fp &other) const
    {
        Int res[5] = {0};
        for (int i = 0; i < 4; i++) {
            Int u = (res[0] + mont.data[i] * other.mont.data[0]) * alt_bn128::pri_inv;
            Int xiy[5], up[5], sum[5];
            big_mul(xiy, (Int*)mont.data + i, (Int*)other.mont.data, 1, 4);
            big_mul(up, &u, alt_bn128::pri.data, 1, 4);
            big_add(sum, xiy, up, 5);
            big_add(sum, sum, res, 5);
            for (int j = 0; j < 4; j++) res[j] = sum[j + 1]; res[4] = 0;
        }
        big_sub(res, res, alt_bn128::pri.data, larger_or_eq(res, alt_bn128::pri.data, 4) * 4);
        alt_bn128_Fp ret; for (int i = 0; i < 4; i++) ret.mont.data[i] = res[i];
        return ret;
    }

    __device__ void alt_bn128_Fp::mont_repr() { (*this) = (*this) * alt_bn128::R_square; }
    __device__ void alt_bn128_Fp::mont_unrepr() { (*this) = (*this) * alt_bn128::one; }
}