#include <iostream>
#include "bigint.hpp"
#include "alt_bn128.hpp"
using namespace std;

#define whatis(x) cerr << #x << " = " << x << endl
#define whatare(pr) cerr << #pr << " = (" << pr.first << ", " << pr.second << ")" << endl

#define SEED 0

Bit randBit() { return Bit(rand() & 1); }
Int randInt() { Int ret = 0; for (int i = 0; i < 3; i++) ret = (ret << 31) | rand(); return ret; }
template<size_t t> Bigint<t> randBigint() { Bigint<t> ret; for (int i = 0; i < t; i++) ret.data[i] = randInt(); return ret; }

extern template class Bigint<4>;
extern template class Bigint<8>;

alt_bn128_Fp rand_alt_bn128_Fp() {
    Bigint<4> ret = randBigint<4>();
    while (larger_or_eq(ret.data, alt_bn128::pri.data, 4)) ret = randBigint<4>();
    return {ret};
}

int main(int argc, char *argv[])
{
    srand(SEED);
    Int a = randInt(), b = randInt();
    whatis(a), whatis(b);
    Bit carry = randBit();

    auto pr = add_with_carry(a, b, carry); whatare(pr);
    pr = sub_with_carry(a, b, carry); whatare(pr);
    auto pr2 = mul(a, b); whatare(pr2);

    Bigint<4> c = randBigint<4>(), p = randBigint<4>(), f, d = randBigint<4>();
    Bigint<8> e;
    c.print_hex();
    d.print_hex();
    p.print_hex();
    big_add(f.data, c.data, d.data, 4);
    f.print_hex();
    big_sub(f.data, c.data, d.data, 4);
    f.print_hex();
    big_mul(e.data, c.data, d.data, 4, 4);
    e.print_hex();
    big_square(e.data, c.data, 4);
    e.print_hex();
    big_modadd(f.data, c.data, d.data, p.data, 4);
    f.print_hex();
    big_modsub(f.data, c.data, d.data, p.data, 4);
    f.print_hex();

    cout << endl;

    {
        alt_bn128_Fp a = {rand_alt_bn128_Fp()}, b = {rand_alt_bn128_Fp()}, c;
        a.print_hex();
        b.print_hex();
        (a + b).print_hex();
        (a - b).print_hex();
        a.mont_repr(), b.mont_repr();
        c = a * b;
        c.mont_unrepr();
        c.print_hex();
    }


}