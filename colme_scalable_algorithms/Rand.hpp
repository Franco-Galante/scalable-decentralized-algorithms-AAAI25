#pragma once

#include <random>
#include <vector>

namespace Rand {

	// rnd generator defined as static, constructed once at the beginning of the execution, then shared by all methods
	// it is used just by the methods in this file and provides abstraction to the calling functions
	std::mt19937& shared_engine(unsigned seed);

	float gaussian_rv(float mu, float std, int seed);

	float uniform_rv(float a, float b, int seed);

	int int_uniform_rv(int a, int b, int seed);

	int discrete_distrib_rv(std::vector<float> p, int seed);

	bool bernoulli_rv(float p, int seed);
}
