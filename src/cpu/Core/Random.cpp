/* 
 * File:   Random.cpp
 * Author: nguyentran
 * 
 * Created on May 27, 2013, 10:46 AM
 */
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <thread>
#include <fmt/format.h>
#include "Random.h"
#include "../Helpers/NumberHelpers.h"

Random::Random(gsl_rng* g_rng) : seed_(0ul), G_RNG(g_rng) {}

Random::~Random() {
  release();
}

void Random::initialize(const unsigned long &seed) {
  const auto tt = gsl_rng_mt19937;
  G_RNG = gsl_rng_alloc(tt);

  auto now = std::chrono::high_resolution_clock::now();
  auto milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
  seed_ = seed == 0 ? static_cast<unsigned long>(milliseconds.count()) : seed;

  std::cout << fmt::format("Random initializing with seed: {}", seed_)<< std::endl;
  gsl_rng_set(G_RNG, seed_);

}

void Random::release() const {
  gsl_rng_free(G_RNG);
}

int Random::random_poisson(const double &poisson_mean) {
  return gsl_ran_poisson(G_RNG, poisson_mean);
}

unsigned long Random::random_uniform(unsigned long range) {
  return gsl_rng_uniform_int(G_RNG, range);
}

//return an integer in  [from, to) , not include to
unsigned long Random::random_uniform_int(const unsigned long &from, const unsigned long &to) {
  return from + gsl_rng_uniform_int(G_RNG, to - from);
}

double Random::random_uniform_double(const double &from, const double &to) {
  //    return from + gsl_rng_uniform_pos(G_RNG)*(to-from);
  return gsl_ran_flat(G_RNG, from, to);
}

double Random::random_uniform() {
  return gsl_rng_uniform(G_RNG);
}

double Random::random_normal(const double &mean, const double &sd) {
  return mean + gsl_ran_gaussian(G_RNG, sd);
}

double Random::random_normal_truncated(const double &mean, const double &sd) {
  double value = gsl_ran_gaussian(G_RNG, sd);
  while (value > 3 * sd || value < -3 * sd) {
    value = gsl_ran_gaussian(G_RNG, sd);
  }

  return mean + value;
}

int Random::random_normal(const int &mean, const int &sd) {
  return static_cast<int>(mean + round(gsl_ran_gaussian(G_RNG, sd)));
}

int Random::random_normal_truncated(const int &mean, const int &sd) {
  double value = gsl_ran_gaussian(G_RNG, sd);
  while (value > 3 * sd || value < -3 * sd) {
    value = gsl_ran_gaussian(G_RNG, sd);
  }

  return static_cast<int>(mean + round(value));
}

double Random::random_beta(const double &alpha, const double &beta) {
  //if beta =0, alpha = means
  if (NumberHelpers::is_equal(beta, 0.0))
    return alpha;
  return gsl_ran_beta(G_RNG, alpha, beta);
}

//
// the parameterization below is done so alpha*beta = mean
// and alpha*beta^2 = variance
//

double Random::random_gamma(const double &shape, const double &scale) {
  //if beta =0, alpha = means
  if (NumberHelpers::is_equal(scale, 0.0))
    return shape;
  return gsl_ran_gamma(G_RNG, shape, scale);
}

double Random::cdf_gamma_distribution(const double &x, const double &alpha, const double &beta) {
  //if beta =0, alpha = means
  if (NumberHelpers::is_equal(beta, 0.0))
    return 1.0;
  return gsl_cdf_gamma_P(x, alpha, beta);
}

double Random::cdf_gamma_distribution_inverse(const double &p, const double &alpha, const double &beta) {
  return gsl_cdf_gamma_Pinv(p, alpha, beta);
}

double Random::random_flat(const double &from, const double &to) {
  return gsl_ran_flat(G_RNG, from, to);
}

void Random::random_multinomial(const size_t &K, const unsigned &N, double p[], unsigned n[]) {
  gsl_ran_multinomial(G_RNG, K, N, p, n);
}

void Random::random_shuffle(void* base, size_t base_length, size_t size_of_type) {
  gsl_ran_shuffle(G_RNG, base, base_length, size_of_type);
}

double Random::cdf_standard_normal_distribution(const double &p) {
  return gsl_cdf_ugaussian_P(p);
}

int Random::random_binomial(const double &p, const unsigned int &n) {
  return gsl_ran_binomial(G_RNG, p, n);
}

void Random::shuffle(void* base, const size_t &n, const size_t &size) {
  gsl_ran_shuffle(G_RNG, base, n, size);
}
