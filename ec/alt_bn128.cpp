#include <iostream>
#include <assert.h>
#include "alt_bn128.hpp"

namespace alt_bn128 {
    const std::string pri_r = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
    const std::string R_square_r = "0216d0b17f4e44a58c49833d53bb808553fe3ab1e35c59e31bb8e645ae216da7";
    const Int pri_inv_r = 0xc2e1f593efffffff;

    Bigint<4> pri{pri_r}, R_square{R_square_r};
    Int pri_inv{pri_inv_r};
}

alt_bn128_Fp::alt_bn128_Fp(Bigint<4> data)
{
    assert(!larger_or_eq(data.data, alt_bn128::pri.data, 4));
    mont = data;
}

namespace alt_bn128 {
    alt_bn128_Fp zero = {Bigint<4>{0}};
    alt_bn128_Fp one = {Bigint<4>{1}};
}

alt_bn128_Fp alt_bn128_Fp::operator+(const alt_bn128_Fp &other) const
{
    alt_bn128_Fp res;
    big_modadd((Int*)res.mont.data, (Int*)mont.data, (Int*)other.mont.data, (Int*)alt_bn128::pri.data, 4);
    return res;
}

alt_bn128_Fp alt_bn128_Fp::operator-(const alt_bn128_Fp &other) const
{
    alt_bn128_Fp res;
    big_modsub((Int*)res.mont.data, (Int*)mont.data, (Int*)other.mont.data, (Int*)alt_bn128::pri.data, 4);
    return res;
}

alt_bn128_Fp alt_bn128_Fp::operator*(const alt_bn128_Fp &other) const
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

void alt_bn128_Fp::print_hex() { mont.print_hex(); }

void alt_bn128_Fp::mont_repr() { (*this) = (*this) * alt_bn128::R_square; }
void alt_bn128_Fp::mont_unrepr() { (*this) = (*this) * alt_bn128::one; }