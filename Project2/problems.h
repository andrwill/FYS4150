#ifndef PROBLEMS_H_
#define PROBLEMS_H_

#include <armadillo>

arma::mat construct_A(int N);
arma::vec get_exact_eigvals(int N, double a, double d);
arma::mat get_exact_eigvecs(int N, double a, double d);

double max_offdiag_symmetric(arma::mat& A, int& k, int& l);
void jacobi_eigensolver(arma::mat& A, arma::vec& eigvals, arma::mat& eigvecs, double eps, int maxiter);
int num_rots_jacobi_eigensolver(arma::mat& A, arma::vec& eigvals, arma::mat& eigvecs, double eps, int maxiter);

void problem2(void);
void problem3(void);
void problem4(void);
void problem5(void);
void problem6(void);

#endif  // PROBLEMS_H_
