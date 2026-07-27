#ifndef RNG_H
#define RNG_H
#include <cstdint>
#include <random>

namespace regi
{
    /* One shared, reused pseudo-random engine per thread.  Previously every
     * random draw reconstructed a std::random_device + std::default_random_engine
     * (slow, and impossible to make reproducible); now a single thread_local
     * std::mt19937 backs every draw and can be seeded for deterministic runs. */

    // The shared thread-local engine (seeded from random_device until seed()).
    std::mt19937 &rng();

    // Seed this thread's engine for reproducible sequences.
    void seed(std::uint64_t s);

    // Uniform integer in [0, n); returns 0 when n == 0.
    std::uint32_t randn(std::uint32_t n);
}  // namespace regi

#endif
