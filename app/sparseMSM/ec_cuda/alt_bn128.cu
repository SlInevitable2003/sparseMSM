#include "alt_bn128.cuh"

namespace cuda {
    namespace alt_bn128 {
        __device__ Bigint<4> pri, R_square;
        __device__ Int pri_inv;
        __device__ alt_bn128_Fp zero, one;
        __device__ alt_bn128_EC infty, gentor;

        __global__ void param_setup(Bigint<4> *pri_dev, 
                                    Bigint<4> *R_square_dev, 
                                    Int *pri_inv_dev, 
                                    alt_bn128_Fp* one_dev, 
                                    alt_bn128_Fp* zero_dev,
                                    alt_bn128_EC* infty_dev,
                                    alt_bn128_EC* gentor_dev)
        {
            pri = *pri_dev;
            R_square = *R_square_dev;
            pri_inv = *pri_inv_dev;
            one = *one_dev;
            zero = *zero_dev;
            infty = *infty_dev;
            gentor = *gentor_dev;
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
    __device__ alt_bn128_Fp alt_bn128_Fp::square() const { return (*this) * (*this); }

    __device__ bool alt_bn128_Fp::operator==(const alt_bn128_Fp& other) const { return eq((Int*)mont.data, (Int*)other.mont.data, 4); }

    __device__ void alt_bn128_Fp::mont_repr() { (*this) = (*this) * alt_bn128::R_square; }
    __device__ void alt_bn128_Fp::mont_unrepr() { (*this) = (*this) * alt_bn128::one; }

    __device__ alt_bn128_EC::alt_bn128_EC(alt_bn128_Fp X, alt_bn128_Fp Y, alt_bn128_Fp Z) : X{X}, Y{Y}, Z{Z} {}

    __device__ alt_bn128_EC alt_bn128_EC::doubling() const
    {
        if (Z == alt_bn128::zero) return (*this);
        alt_bn128_EC res;

        auto A = X.square();
        auto B = Y.square();
        auto C = B.square();
        auto D = ((X + B).square() - A - C); D = D + D;
        auto E = A + A + A;
        auto F = E.square();
        res.X = F - (D + D);
        C = C + C; C = C + C; C = C + C;
        res.Y = E * (D - res.X) - C;
        res.Z = Y * Z; res.Z = res.Z + res.Z;
        return res;
    }

    __device__ alt_bn128_EC alt_bn128_EC::operator+(const alt_bn128_EC& other) const
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
        auto I = (H + H).square();
        auto J = H * I;
        auto r = (S2 - S1); r = r + r;
        auto V = U1 * I;
        res.X = r.square() - J - (V + V);
        S1 = S1 * J; S1 = S1 + S1;
        res.Y = r * (V - res.X) - S1;
        res.Z = ((Z + other.Z).square() - Z1Z1 - Z2Z2) * H;

        return res;
    }

    __device__ bool alt_bn128_EC::operator==(const alt_bn128_EC& other) const
    {
        auto P1 = (*this), P2 = other;
        P1.mont_repr(); P2.mont_repr();
        if (P1.Z == alt_bn128::zero && P2.Z == alt_bn128::zero) return true;
        if (P1.Z == alt_bn128::zero || P2.Z == alt_bn128::zero) return false;
        
        auto Z1Z1 = P1.Z.square();
        auto Z2Z2 = P2.Z.square();
        auto U1 = P1.X * Z2Z2;
        auto U2 = P2.X * Z1Z1;
        auto S1 = P1.Y * P2.Z * Z2Z2;
        auto S2 = P2.Y * P1.Z * Z1Z1;

        return (U1 == U2 && S1 == S2);
    }

    __device__ alt_bn128_EC alt_bn128_EC::native_scale(const alt_bn128_Fp& scalar) const
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
        return sum;
    }

}