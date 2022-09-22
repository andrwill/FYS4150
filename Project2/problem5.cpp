#include "problems.h"

#include <string>
#include <ctgmath>
#include <armadillo>



void problem5() {
    int n_runs = 6;
    arma::vec N = arma::vec(n_runs);
    arma::vec num_rots = arma::vec(n_runs);

    arma::mat A;
    arma::vec eigvals;
    arma::mat eigvecs;

    for (int i = 0; i < n_runs; i++) {
        std::cout << i << std::endl;
        N(i) = (int)(pow(2.0, i+1));
        A = construct_A(N(i));
        num_rots(i) = num_rots_jacobi_eigensolver(A, eigvals, eigvecs, 1.0e-12, 10000);
    }

    // Make a table of N and the corresponding number of rotations.
    arma::mat table(n_runs, 2, arma::fill::zeros);
    table.col(0) = N;
    table.col(1) = num_rots;

    arma::field<std::string> header(2);
    header(0) = "N";
    header(1) = "Rotations";
    
    table.save(arma::csv_name("./tables/number_of_rotations.csv", header));
}
