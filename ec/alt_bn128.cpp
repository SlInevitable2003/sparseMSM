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

alt_bn128_Fp alt_bn128_Fp::square() const { return (*this) * (*this); }

bool alt_bn128_Fp::operator==(const alt_bn128_Fp& other) const { return eq((Int*)mont.data, (Int*)other.mont.data, 4); }

void alt_bn128_Fp::print_hex() { mont.print_hex(); }

void alt_bn128_Fp::mont_repr() { (*this) = (*this) * alt_bn128::R_square; }
void alt_bn128_Fp::mont_unrepr() { (*this) = (*this) * alt_bn128::one; }

alt_bn128_EC::alt_bn128_EC(alt_bn128_Fp X, alt_bn128_Fp Y, alt_bn128_Fp Z) : X{X}, Y{Y}, Z{Z} {}

namespace alt_bn128 {
    alt_bn128_Fp two = {Bigint<4>{2}};
    alt_bn128_EC infty = {zero, one, zero}, gentor = {one, two, one};
}

alt_bn128_EC alt_bn128_EC::doubling() const
{
    alt_bn128_Fp three = {Bigint<4>{3}}, eight = {Bigint<4>{8}};

    if (Z == alt_bn128::zero) return (*this);
    alt_bn128_EC res;

    auto A = X.square();
    auto B = Y.square();
    auto C = B.square();
    auto D = alt_bn128::two * ((X + B) * (X + B) - A - C);
    auto E = three * A;
    auto F = E.square();
    res.X = F - (alt_bn128::two * D);
    res.Y = E * (D - res.X) - (eight * C);
    res.Z = alt_bn128::two * Y * Z;
    return res;
}

alt_bn128_EC alt_bn128_EC::operator+(const alt_bn128_EC& other) const
{
    if (Z == alt_bn128::zero) return other;
    else if (other.Z == alt_bn128::zero) return (*this);
    
    alt_bn128_EC res;
    auto Z1Z1 = Z.square();
    auto Z2Z2 = other.Z.square();
    auto U1 = X * Z2Z2;
    auto U2 = other.X * Z1Z1;
    auto S1 = Y * other.Z * Z2Z2;
    auto S2 = other.Y * Z * Z1Z1;

    if (U1 == U2 && S1 == S2) return doubling();

    auto H = U2 - U1;
    auto I = (alt_bn128::two * H).square();
    auto J = H * I;
    auto r = alt_bn128::two * (S2 - S1);
    auto V = U1 * I;
    res.X = r.square() - J - (alt_bn128::two * V);
    res.Y = r * (V - res.X) - (alt_bn128::two * S1 * J);
    res.Z = ((Z + other.Z).square() - Z1Z1 - Z2Z2) * H;

    return res;
}

alt_bn128_EC alt_bn128_EC::native_scale(const alt_bn128_Fp& scalar) const
{
    alt_bn128_EC sum = alt_bn128::infty, inc = (*this);
    sum.mont_repr(), inc.mont_repr();

    Int *arr = (Int*)scalar.mont.data;
    for (int i = 0; i < 4; i++) {
        Int ele = arr[i];
        for (int j = 0; j < WORD; j++) {
            if (ele & 1) sum = sum + inc;
            inc = inc.doubling();
            ele >>= 1;
        }
    }
    sum.mont_unrepr();
    return sum;
}

void alt_bn128_EC::print_hex()
{
    std::cout << "== Point Info ==" << std::endl;
    X.print_hex(), Y.print_hex(), Z.print_hex();
    std::cout << "== End Info ==" << std::endl;
}