#pragma once
#include <iostream>
#include "bigint.hpp"

namespace alt_bn128 {
    extern Bigint<4> pri, R_square;
    extern Int pri_inv;
}

class alt_bn128_Fp {    
public:
    Bigint<4> mont;
    
    alt_bn128_Fp() = default;
    alt_bn128_Fp(Bigint<4> data);

    void mont_repr();
    void mont_unrepr();

    // alt_bn128_Fp inverse() const;

    alt_bn128_Fp operator+(const alt_bn128_Fp& other) const;
    alt_bn128_Fp operator-(const alt_bn128_Fp& other) const;
    alt_bn128_Fp operator*(const alt_bn128_Fp& other) const;
    alt_bn128_Fp square() const;

    bool operator==(const alt_bn128_Fp& other) const;

    void print_hex();
};

namespace alt_bn128 {
    extern alt_bn128_Fp zero, one;
}

class alt_bn128_EC { 
    // y^2 = x^3 + 3
    // refer to https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html for more details
public:
    alt_bn128_Fp X, Y, Z;
    alt_bn128_EC() = default;
    alt_bn128_EC(alt_bn128_Fp X, alt_bn128_Fp Y, alt_bn128_Fp Z);

    void mont_repr() { X.mont_repr(), Y.mont_repr(), Z.mont_repr(); }
    void mont_unrepr() { X.mont_unrepr(), Y.mont_unrepr(), Z.mont_unrepr(); }

    alt_bn128_EC operator+(const alt_bn128_EC& other) const;
    alt_bn128_EC doubling() const;

    bool operator==(const alt_bn128_EC& other) const;

    alt_bn128_EC native_scale(const alt_bn128_Fp& scalar, bool unrepr = false) const;

    void print_hex();
};

namespace alt_bn128 {
    extern alt_bn128_EC infty, gentor;
}