#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <vector>
#include <combo.h>
#include <rng.h>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> pile)
{
    for (auto c : pile) { os << c << " "; }
    os << "\n";
    return os;
}

template <typename T>
void shuffle(std::vector<T> &pile, u32 start, u32 end)
{
    if (end <= 1 || end <= start) { return; }
    i32 i, j;
    i32 len = end - start;
    for (i = 0; i < len - 1; ++i)
    {
        j = i + regi::randn(len - i);
        std::swap(pile[start + i], pile[start + j]);
    }
}

std::ostream &operator<<(std::ostream &os, const std::vector<regi::Combo> pile);
#endif
