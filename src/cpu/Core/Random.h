/* 
 * File:   Random.h
 * Author: nguyentran
 *
 * Created on May 27, 2013, 10:46 AM
 */

#ifndef RANDOM_H
#define    RANDOM_H

#include <gsl/gsl_rng.h>
#include "PropertyMacro.h"

class Model;

class Random {
 DISALLOW_COPY_AND_ASSIGN(Random)

 DISALLOW_MOVE(Random)

 VIRTUAL_PROPERTY(unsigned long, seed)

 public:
  gsl_rng *G_RNG;

  explicit Random(gsl_rng *g_rng = nullptr);

  virtual ~Random();

  void initialize(const unsigned long &seed = 0);

  void release() const;

  virtual int random_poisson(const double &poisson_mean);

  virtual unsigned long random_uniform(unsigned long range);

  virtual unsigned long random_uniform_int(const unsigned long &from, const unsigned long &to);

  virtual double random_uniform_double(const double &from, const double &to);

  /*
   * This function will return a random number in [0,1)
   */
  virtual double random_uniform();

  virtual double random_normal(const double &mean, const double &sd);

  virtual double random_normal_truncated(const double &mean, const double &sd);

  virtual int random_normal(const int &mean, const int &sd);

  virtual int random_normal_truncated(const int &mean, const int &sd);

  virtual double random_beta(const double &alpha, const double &beta);

  virtual double random_gamma(const double &shape, const double &scale);

  virtual double cdf_gamma_distribution(const double &x, const double &alpha, const double &beta);

  virtual double cdf_gamma_distribution_inverse(const double &p, const double &alpha, const double &beta);

  virtual double random_flat(const double &from, const double &to);

  virtual void random_multinomial(const size_t &K, const unsigned &N, double p[], unsigned n[]);

  virtual void random_shuffle(void *base, size_t base_length, size_t size_of_type);

  virtual double cdf_standard_normal_distribution(const double &p);

  virtual int random_binomial(const double &p, const unsigned int &n);

  void shuffle(void *base, const size_t &n, const size_t &size);
};

#endif    /* RANDOM_H */
