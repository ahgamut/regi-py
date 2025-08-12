#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <vector>
#include <random>
#include <combo.h>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> pile)
{
    for (auto c : pile) { os << c << " "; }
    os << "\n";
    return os;
}

template <typename T>
void shuffle(std::vector<T> &pile, std::uint32_t start, std::uint32_t end)
{
    if (end <= 1 || end <= start) { return; }
    int i, j;
    int len = end - start;
    std::random_device dev;
    std::default_random_engine engine(dev());
    for (i = 0; i < len - 1; ++i)
    {
        j = i + (engine() % (len - i));
        std::swap(pile[start + i], pile[start + j]);
    }
}

std::ostream &operator<<(std::ostream &os, const std::vector<regi::Combo> pile);
#endif
