#pragma once
#include <iostream>
#include "bigint.hpp"

namespace alt_bn128 {
    extern Bigint<4> pri, R_square;
    extern Int pri_inv;
}

class alt_bn128_Fp {
    Bigint<4> mont;
public:
    alt_bn128_Fp() = default;
    alt_bn128_Fp(Bigint<4> data);

    void mont_repr();
    void mont_unrepr();

    // alt_bn128_Fp inverse() const;

    alt_bn128_Fp operator+(const alt_bn128_Fp& other) const;
    alt_bn128_Fp operator-(const alt_bn128_Fp& other) const;
    alt_bn128_Fp operator*(const alt_bn128_Fp& other) const;

    void print_hex();
};

namespace alt_bn128 {
    extern alt_bn128_Fp zero, one;
}