#include <iostream>
#include <cassert>
#include <omp.h>
#include "ec/bigint.hpp"
#include "ec/alt_bn128.hpp"
using namespace std;

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

void instance_generating(int *randomness, alt_bn128_Fp *coeff, alt_bn128_EC *point, int n)
{
    omp_set_num_threads(12);
    #pragma omp parallel
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            switch (randomness[i]) {
                case 0: { coeff[i] = alt_bn128::zero; } break;
                case 1: { coeff[i] = alt_bn128::one; } break;
                case 2: { coeff[i] = rand_alt_bn128_Fp(); }
            }
            point[i] = rand_alt_bn128_EC();
        }
    }
}