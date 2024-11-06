#pragma once

namespace cuda {
    #define WORD 64

    typedef unsigned long Int;
    typedef bool Bit;

    template<typename T1, typename T2>
    struct pair {
        T1 first; T2 second;
        pair() = default;
        __device__ pair(T1 a, T2 b) { first = a, second = b; }
    };

    __device__ pair<Int, Bit> add_with_carry(Int a, Int b, Bit carry);
    __device__ pair<Int, Bit> sub_with_carry(Int a, Int b, Bit carry);
    __device__ pair<Int, Int> mul(Int a, Int b);

    template <size_t t>
    class Bigint {
    public:
        Int data[t] = {0};

        Bigint() = default;
        __device__ Bigint(const Int sml);
    };

    __device__ bool larger_or_eq(Int* a, Int* b, size_t t);

    __device__ Bit big_add(Int* res, Int* a, Int* b, size_t t);
    __device__ Bit big_sub(Int* res, Int* a, Int* b, size_t t);

    __device__ void big_mul(Int* res, Int* a, Int* b, size_t ta, size_t tb);
    __device__ void big_square(Int* res, Int* a, size_t t);

    __device__ void big_modadd(Int* res, Int* a, Int* b, Int* p, size_t t);
    __device__ void big_modsub(Int* res, Int* a, Int* b, Int* p, size_t t);
}