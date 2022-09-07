#ifndef SOLVERS_H
#define SOLVERS_H

#include <armadillo>

using namespace arma;

vec solve_tridiagonal_system(vec& a, vec b, vec& c, vec& g);
vec solve_poisson1d_s(vec f(vec&), int const M);
vec solve_poisson1d_g(vec f(vec&), int const M);

#endif // SOLVERS_H
