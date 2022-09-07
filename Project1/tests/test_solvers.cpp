#include "solvers.h"

#include <cmath> // For `exp`.
#include <cassert> // For `assert`.
#include <armadillo>

#include <iostream>

using namespace arma;

vec f(vec& x) {
    return 100.0*arma::exp(-10.0*x);
}

vec exact_solution(vec& x) {
    return 1.0 - (1.0 - exp(-10.0))*x - arma::exp(-10.0*x);
}

int main() {
    int const M = 10000;
    int N = M-2;
    double h = 1.0 / (double)(N+1);
    vec x = linspace(h, 1.0-h, N);
    vec a = vec(N-1).fill(-1.0);
    vec b = vec(N).fill(2.0);
    vec c = vec(N-1).fill(-1.0);

    vec v_exact = exact_solution(x);
    vec v_approx_s = solve_poisson1d_s(&f, M);
    vec v_approx_g = solve_poisson1d_g(&f, M);

    double atol = 1e-3;

    assert(norm(v_exact - v_approx_s, "inf") < atol);
    assert(norm(v_exact - v_approx_g, "inf") < atol);

    return 0;
}
