#include <iostream>
#include <iomanip>
#include <assert.h>
#include "bigint.hpp"

std::pair<Int, Bit> add_with_carry(Int a, Int b, Bit carry)
{
    int half_word = WORD >> 1;
    Int mask = (1ull << half_word) - 1;
    Int a_low = a & mask, b_low = b & mask;
    Int a_high = a >> half_word, b_high = b >> half_word;

    Int low_sum = a_low + b_low + carry;
    Int c_low = low_sum & mask, c_high = (low_sum >> half_word) + (a_high + b_high);
    Int c = (c_high << half_word) | c_low;
    Bit carry_out = Bit(c_high >> half_word);
    return {c, carry_out};
}

std::pair<Int, Bit> sub_with_carry(Int a, Int b, Bit carry)
{
    int half_word = WORD >> 1;
    Int mask = (1ull << half_word) - 1;
    Int a_low = a & mask, b_low = b & mask;
    Int a_high = a >> half_word, b_high = b >> half_word;

    Int low_sum = a_low - b_low - carry;
    Int c_low = low_sum & mask, c_high = a_high - b_high - ((low_sum >> half_word) & 1ull);
    Int c = (c_high << half_word) | c_low;
    Bit carry_out = Bit(c_high >> half_word);
    return {c, carry_out};
}

std::pair<Int, Int> mul(Int a, Int b)
{
    int half_word = WORD >> 1;
    Int mask = (1ull << half_word) - 1;
    Int a_low = a & mask, b_low = b & mask;
    Int a_high = a >> half_word, b_high = b >> half_word;

    Int p0 = a_low * b_low, p1 = a_low * b_high, p2 = a_high * b_low, p3 = a_high * b_high;
    Int c = p3, d = p0;
    Bit carry = 0;
    std::pair<Int, Bit> pr;

    pr = add_with_carry(d, ((p1 & mask) << half_word), 0); d = pr.first, carry = pr.second;
    c += (p1 >> half_word) + carry;
    pr = add_with_carry(d, ((p2 & mask) << half_word), 0); d = pr.first, carry = pr.second;
    c += (p2 >> half_word) + carry;
    return {c, d};
}

template<size_t t> Bigint<t>::Bigint(const Int sml) { data[0] = sml; }
template<size_t t> Bigint<t>::Bigint(const std::string s) 
{
    int hex_count_per_word = 16;
    assert(s.length() == t * hex_count_per_word);
    for (int i = 0; i < s.length(); i++) {
        char c = s[i]; int j = t - (i / hex_count_per_word) - 1;
        if ('0' <= c && c <= '9') data[j] = (data[j] << 4) | Int(c - '0');
        else if ('a' <= c && c <= 'z') data[j] = (data[j] << 4) | Int(10 + c - 'a');
        else if ('A' <= c && c <= 'Z') data[j] = (data[j] << 4) | Int(10 + c - 'A');
    }
}

template<size_t t> void Bigint<t>::print_hex() const
{
    std::cout << "0x";
    for (int i = t - 1; i >= 0; i--) std::cout << std::hex << std::setw(16) << std::setfill('0') << data[i];
    std::cout << std::dec << std::endl;
}

bool larger_or_eq(Int* a, Int* b, size_t t) 
{
    int highest_less_bit = -1;
    int highest_larger_bit = -1;
    for (int i = 0; i < t; i++) {
        bool flag_less = (a[i] < b[i]), flag_larger = (a[i] > b[i]);
        highest_less_bit = flag_less * i + (1 - flag_less) * highest_less_bit;
        highest_larger_bit = flag_larger * i + (1 - flag_larger) * highest_larger_bit;
    }
    return (highest_larger_bit > highest_less_bit) || (highest_less_bit == -1);
}

bool eq(Int* a, Int *b, size_t t)
{
    int highest_less_bit = -1;
    int highest_larger_bit = -1;
    for (int i = 0; i < t; i++) {
        bool flag_less = (a[i] < b[i]), flag_larger = (a[i] > b[i]);
        highest_less_bit = flag_less * i + (1 - flag_less) * highest_less_bit;
        highest_larger_bit = flag_larger * i + (1 - flag_larger) * highest_larger_bit;
    }
    return (highest_larger_bit == -1) && (highest_less_bit == -1);
}

Bit big_add(Int* res, Int* a, Int* b, size_t t)
{
    Bit carry = 0;
    for (int i = 0; i < t; i++) { auto pr = add_with_carry(a[i], b[i], carry); res[i] = pr.first, carry = pr.second; }
    return carry;
}

Bit big_sub(Int* res, Int* a, Int* b, size_t t)
{
    Bit carry = 0;
    for (int i = 0; i < t; i++) { auto pr = sub_with_carry(a[i], b[i], carry); res[i] = pr.first, carry = pr.second; }
    return carry;
}

void big_mul(Int* res, Int* a, Int* b, size_t ta, size_t tb)
{
    Int r0 = 0, r1 = 0, r2 = 0;
    for (int i = 0; i < ta + tb - 1; i++) {
        int lower = (i >= (tb - 1)) * (i - (tb - 1));
        int upper = i - (i >= (ta - 1)) * (i - (ta - 1));
        for (int j = 0; j < ta; j++) {
            bool work = (j >= lower) && (j <= upper); 
            auto pr = mul(a[work * j], b[work * (i - j)]);
            Bit carry;

            auto pr2 = add_with_carry(r0, work * pr.second, 0); r0 = pr2.first, carry = pr2.second;
            pr2 = add_with_carry(r1, work * pr.first, carry); r1 = pr2.first, carry = pr2.second;
            r2 += carry;
        }
        res[i] = r0, r0 = r1, r1 = r2, r2 = 0;
    }
    res[ta + tb - 1] = r0;
}

void big_square(Int* res, Int* a, size_t t)
{
    // TODO: the count of single-precision multiplication can be reduced by half using bitwise left-shift
    big_mul(res, a, a, t, t);
}

void big_modadd(Int* res, Int* a, Int* b, Int* p, size_t t)
{
    Bit carry = big_add(res, a, b, t);
    bool flag = (carry || larger_or_eq(res, p, t));
    big_sub(res, res, p, flag * t);
}
void big_modsub(Int* res, Int* a, Int* b, Int* p, size_t t)
{
    Bit carry = big_sub(res, a, b, t);
    big_add(res, res, p, carry * t);
}

template class Bigint<4>;
template class Bigint<8>;