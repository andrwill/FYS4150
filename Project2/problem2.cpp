#include "problems.h"

#include <cmath>
#include <cassert>
#include <armadillo>

double pi = M_PI;

arma::mat construct_A(int N) {
    double h = 1.0/(double)(N+1);
    double a = -1.0/(h*h);
    double d = 2.0/(h*h);

    arma::mat A = d*arma::eye(N, N);

    for (int i = 0; i < N-1; i++) {
        A(i+1,i) = a;
        A(i,i+1) = a;
    }

    return A;
}

arma::vec get_exact_eigvals(int N, double a, double d) {
    arma::vec theta = arma::linspace(0.0, pi, N+2).rows(1,N);

    return 2.0*a*arma::cos(theta) + d;
}

arma::mat get_exact_eigvecs(int N, double a, double d) {
    arma::vec k = arma::linspace(1.0, N, N);

    arma::mat K = k*k.t(); // K = (i*j)_{i,j=1,...,N}.

    return arma::normalise(sin((pi/(double)(N+1))*K));
}

void problem2() {
    int N = 6;

    double h = 1.0/(double)(N+1);
    double a = -1.0/(h*h);
    double d = 2.0/(h*h);

    arma::vec exact_eigvals = get_exact_eigvals(N, a, d);
    arma::mat exact_eigvecs = get_exact_eigvecs(N, a, d);

    arma::mat A = d*arma::eye(N, N);

    for (int i = 0; i < N-1; i++) {
        A(i+1,i) = a;
        A(i,i+1) = a;
    }

    arma::vec eigval;
    arma::mat eigvec;

    arma::eig_sym(eigval, eigvec, A);

    // Correct any sign differences
    double sgn1, sgn2;
    for (int j = 0; j < N; j++) {
        sgn1 = copysign(1.0, exact_eigvecs(0,j));
        sgn2 = copysign(1.0, eigvec(0,j));
        eigvec.col(j) *= sgn1*sgn2;
    }

    assert(arma::approx_equal(exact_eigvals, eigval, "absdiff", 1e-8));
    assert(arma::approx_equal(exact_eigvecs, eigvec, "absdiff", 1e-8));
}
