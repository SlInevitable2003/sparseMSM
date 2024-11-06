#pragma once
#include <iostream>

#define WORD 64

typedef uint64_t Int;
typedef uint32_t Half;
typedef bool Bit;

std::pair<Int, Bit> add_with_carry(Int a, Int b, Bit carry);
std::pair<Int, Bit> sub_with_carry(Int a, Int b, Bit carry);
std::pair<Int, Int> mul(Int a, Int b);

template <size_t t>
class Bigint {
public:
    Int data[t] = {0};

    Bigint() = default;
    Bigint(const Int sml);
    Bigint(const std::string s);

    void print_hex() const;
};

bool larger_or_eq(Int* a, Int* b, size_t t);
bool eq(Int* a, Int *b, size_t t);

Bit big_add(Int* res, Int* a, Int* b, size_t t);
Bit big_sub(Int* res, Int* a, Int* b, size_t t);

void big_mul(Int* res, Int* a, Int* b, size_t ta, size_t tb);
void big_square(Int* res, Int* a, size_t t);

void big_modadd(Int* res, Int* a, Int* b, Int* p, size_t t);
void big_modsub(Int* res, Int* a, Int* b, Int* p, size_t t);
