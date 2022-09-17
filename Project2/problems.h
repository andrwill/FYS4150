#ifndef PROBLEMS_H_
#define PROBLEMS_H_

#include <armadillo>

arma::mat construct_A(int N);
arma::vec get_exact_eigvals(int N, double a, double d);
arma::mat get_exact_eigvecs(int N, double a, double d);

double max_offdiag_symmetric(arma::mat& A, int& k, int& l);

void problem2();
void problem3();
void problem4();

#endif  // PROBLEMS_H_
