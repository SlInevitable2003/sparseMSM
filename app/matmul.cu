#include <iostream>
#include "../ec/bigint.hpp"
#include "../ec/alt_bn128.hpp"
using namespace std;

#define whatis(x) cerr << #x << " = " << x << endl
#define whatare(pr) cerr << #pr << " = (" << pr.first << ", " << pr.second << ")" << endl

#define SEED 0

Bit randBit() { return Bit(rand() & 1); }
Int randInt() { Int ret = 0; for (int i = 0; i < 3; i++) ret = (ret << 31) | rand(); return ret; }
template<size_t t> Bigint<t> randBigint() { Bigint<t> ret; for (int i = 0; i < t; i++) ret.data[i] = randInt(); return ret; }
alt_bn128_Fp rand_alt_bn128_Fp() 
{
    Bigint<4> ret = randBigint<4>();
    while (larger_or_eq(ret.data, alt_bn128::pri.data, 4)) ret = randBigint<4>();
    return {ret};
}

const int n = 1 << 4;
alt_bn128_Fp A[n][n], B[n][n], C[n][n]; // total 3 * n * n * 4 * 8B = 24KB

int main(int argc, char *argv[])
{
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        A[i][j] = rand_alt_bn128_Fp(), B[i][j] = rand_alt_bn128_Fp(), C[i][j] = alt_bn128::zero;
        A[i][j].print_hex(), B[i][j].print_hex();
        A[i][j].mont_repr(), B[i][j].mont_repr();
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) C[i][j] = C[i][j] + (A[i][k] * B[k][j]);
        C[i][j].mont_unrepr();
        C[i][j].print_hex();
    }

    return 0;
}