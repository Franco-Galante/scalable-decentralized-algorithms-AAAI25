#include "Rand.hpp"

#include <random>
#include <vector>

namespace Rand {

    // rnd generator defined as static, constructed once at the beginning of the execution, then shared by all methods
    // it is used just by the methods in this file and provides abstraction to the calling functions
    std::mt19937& shared_engine(unsigned seed) {
        static std::mt19937 e{ seed };
        return e;
    }

    float gaussian_rv(float mu, float std, int seed) {
        std::normal_distribution<float> g_distrib(mu, std);
        return g_distrib(shared_engine(seed));
    }

    float uniform_rv(float a, float b, int seed) {
        std::uniform_real_distribution<float> uniform_distrib(a, b);
        return uniform_distrib(shared_engine(seed));
    }

    int int_uniform_rv(int a, int b, int seed) { // uniform samples in the close interval [a,b]
        std::uniform_int_distribution<int> uniform_distrib(a, b);
        return uniform_distrib(shared_engine(seed));
    }

    int discrete_distrib_rv(std::vector<float> p, int seed) { // samples in the interval [0,p.size() ) with probability p_i
        std::discrete_distribution<int> discrete_distrib(p.begin(), p.end());
        return discrete_distrib(shared_engine(seed));
    }

    bool bernoulli_rv(float p, int seed) {
        std::bernoulli_distribution bernoulli_distrib(p);
        return bernoulli_distrib(shared_engine(seed));
    }

}
