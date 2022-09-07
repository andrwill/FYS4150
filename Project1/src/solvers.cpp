#include "solvers.h"

#include <armadillo>

using namespace arma;

vec solve_tridiagonal_system(vec& a, vec b, vec& c, vec& g) {
    int const N = arma::size(b)[0]; // Writing `arma::size` explicitly to avoid confusion with `std::size` in C++17.

    vec v = g;

    double coeff;
    for (int i = 0; i < N-1; i++) { // Forward
        coeff = a(i)/b(i);
        b(i+1) -= coeff*c(i);
        v(i+1) -= coeff*v(i);
    }
    v(N-1) /= b(N-1);
    for (int i = N-1; i > 1; i--) { // Backward
        v(i-1) -= v(i)*c(i-1)/b(i);
    }

    return v;
}

vec solve_poisson1d_s(vec f(vec&), int const M=100) {
    int N = M-2;
    double h = 1.0/((double)(N+1));
    vec x = linspace(h, 1.0-h, N);
    vec v = (h*h)*f(x);

    double coeff;
    for (int i = 0; i < N-1; i++) { // Forward
        coeff = (double)(i+1) / (double)(i+2);
        v(i+1) += coeff*v(i);
    }
    
    v(N-1) *= (double)(N)/(double)(N+1);
    
    for (int i = N-1; i > 1; i--) { // Backward
        coeff = (double)(i) / (double)(i+1);
        v(i-1) += v(i);
        v(i-1) *= coeff;
    }

    return v;
}

vec solve_poisson1d_g(vec f(vec&), int const M=100) 
{
    int N = M-2;
    double h = 1.0/(double)(N+1);
    vec a = vec(N-1).fill(-1.0);
    vec b = vec(N).fill(2.0);
    vec c = vec(N-1).fill(-1.0);

    vec x = linspace(h, 1.0-h, N);
    vec g = (h*h)*f(x);

    vec v = solve_tridiagonal_system(a, b, c, g);

    return v;
}
