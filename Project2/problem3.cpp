
#include "problems.h"

#include <ctgmath>
#include <cassert>
#include <armadillo>

double max_offdiag_symmetric(arma::mat& A, int& k, int& l) {
    arma::vec diag = A.diag(); // Store the diagonal of A.
    A.diag().zeros(); // Set the diagonal entries of A to zero.

    double max_offdiag = A.max();
    double min_offdiag = A.min();

    double max_abs;
    arma::uvec sub;
    if (abs(max_offdiag) > abs(min_offdiag)) {
        max_abs = max_offdiag;
        sub = arma::ind2sub(arma::size(A), A.index_max()); // Convert arma::uword to arma::uvec.
    } else {
        max_abs = min_offdiag;
        sub = arma::ind2sub(arma::size(A), A.index_min()); // Convert arma::uword to arma::uvec.
    }

    k = (int)(fmax(sub(0), sub(1)));
    l = (int)(fmin(sub(0), sub(1)));

    A.diag() = diag; // Restore the diagonal of A.

    return max_abs;
}

void problem3() {
    int k, l;
    arma::mat A = arma::mat({
        {1.0, 0.0, 0.0, 0.5},
        {0.0, 1.0, -0.7, 0.0},
        {0.0, -0.7, 1.0, 0.0},
        {0.5, 0.0, 0.0, 1.0}
    });

    double max_offdiag = max_offdiag_symmetric(A, k, l);

    assert(max_offdiag == -0.7);
    assert(k == 2 and l == 1);
}
