#include "problems.h"

#include <cmath>
#include <string>
#include <cassert>
#include <stdexcept>
#include <armadillo>

void jacobi_rotate(arma::mat& A, arma::mat& R, int k, int l) {
    
    double s, c; // Sine and cosine.
    if ( A(k,l) == 0.0 ) { // cos(0) = 1, sin(0) = 0.
        c = 1.0;
        s = 0.0;
    } else {
        double tau, t;
        tau = 0.5*(A(l,l) - A(k,k))/A(k,l);
        if ( tau > 0 ) {
            t = 1.0/(tau + sqrt(1.0 + tau*tau)); // Tangent
        } else {
            t = -1.0/( -tau + sqrt(1.0 + tau*tau));
        }
        c = 1.0/sqrt(1.0+t*t);
        s = c*t;
    }

    double a_kk, a_kl, a_ll;
    a_kk = A(k,k);
    a_kl = A(k,l);
    a_ll = A(l,l);

    // Conjugate A by Jacobi rotation (A <-- J^T * A * J )
    A(k,k) = c*c*a_kk + s*s*a_ll - 2.0*c*s*a_kl;
    A(l,l) = s*s*a_kk + c*c*a_ll + 2.0*c*s*a_kl;
    A(k,l) = 0.0;
    A(l,k) = 0.0;

    double a_ik, a_il, r_ik, r_il;
    for (int i = 0; i < A.n_rows; i++) {
        if (i != k and i != l ) {
            a_ik = A(i,k);
            a_il = A(i,l);

            A(i,k) = c*a_ik - s*a_il;
            A(i,l) = s*a_ik + c*a_il;
            A(k,i) = A(i,k);
            A(l,i) = A(i,l);
        }

        // Apply Jacobi rotation to R.
        r_ik = R(i,k);
        r_il = R(i,l);
        R(i,k) = c*r_ik - s*r_il;
        R(i,l) = s*r_ik + c*r_il;
    }
}

void jacobi_eigensolver(arma::mat& A, arma::vec& eigvals, arma::mat& eigvecs, double eps, int maxiter) {
    if (not A.is_symmetric()) {
        throw std::invalid_argument("`A` must be symmetric.");
    }

    int N = A.n_cols;
    arma::mat R = arma::eye(N, N);
   
    int k, l;
    int iter = 0;
    double max_entry = max_offdiag_symmetric(A, k, l);
    while (abs(max_entry) > eps and iter < maxiter) {
        jacobi_rotate(A, R, k, l);
        max_entry = max_offdiag_symmetric(A, k, l);
        iter++;
    }

    if (abs(max_entry) > eps) {
        throw std::runtime_error(std::to_string(abs(max_entry)));//"Failed to reach desired accuracy. Current accuracy: " + abs(max_entry) + ".");
    }

    eigvals = arma::diagvec(A); // Extract the main diagonal of A.
    eigvecs = arma::normalise(R);
}
int num_rots_jacobi_eigensolver(arma::mat& A, arma::vec& eigvals, arma::mat& eigvecs, double eps, int maxiter) {
    if (not A.is_symmetric()) {
        throw std::invalid_argument("`A` must be symmetric.");
    }

    int N = A.n_cols;
    arma::mat R = arma::eye(N, N);
   
    int k, l;
    int iter = 0;
    double max_entry = max_offdiag_symmetric(A, k, l);
    while (abs(max_entry) > eps and iter < maxiter) {
        jacobi_rotate(A, R, k, l);
        max_entry = max_offdiag_symmetric(A, k, l);
        iter++;
    }

    if (abs(max_entry) > eps) {
        throw std::runtime_error(std::to_string(abs(max_entry)));//"Failed to reach desired accuracy. Current accuracy: " + abs(max_entry) + ".");
    }

    eigvals = arma::diagvec(A); // Extract the main diagonal of A.
    eigvecs = arma::normalise(R);

    return iter;
}

void problem4() {
    int N = 6;
    double h = 1.0/(double)(N+1);
    double a = -1.0/(h*h);
    double b = 2.0/(h*h);
    arma::vec exact_eigvals = get_exact_eigvals(N, a, b);
    arma::mat exact_eigvecs = get_exact_eigvecs(N, a, b);
    
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::mat A = construct_A(N);

    jacobi_eigensolver(A, eigvals, eigvecs, 1.0e-12, 10000);

    std::cout << arma::sort(eigvals).t() << std::endl;
    std::cout << arma::sort(exact_eigvals).t() << std::endl;
    std::cout << arma::sort(eigvecs) << std::endl;
    std::cout << arma::sort(exact_eigvecs) << std::endl;

    //double atol = 1.0e-2;
    //assert(arma::approx_equal(arma::sort(eigvals), arma::sort(exact_eigvals), "absdiff", atol));
    //assert(arma::approx_equal(arma::sort(eigvecs), arma::sort(exact_eigvecs), "absdiff", atol));
}
