#include <rng.h>

namespace regi
{
    std::mt19937 &rng()
    {
        // lazily constructed once per thread, seeded non-deterministically by
        // default so unseeded runs stay random
        static thread_local std::mt19937 engine{std::random_device{}()};
        return engine;
    }

    void seed(std::uint64_t s)
    {
        rng().seed(static_cast<std::mt19937::result_type>(s));
    }

    std::uint32_t randn(std::uint32_t n)
    {
        if (n == 0) { return 0; }
        return static_cast<std::uint32_t>(rng()() % n);
    }
}  // namespace regi
