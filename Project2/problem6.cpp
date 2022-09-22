#include "problems.h"

#include <string>
#include <iostream>
#include <armadillo>

void save_n_last_eigvecs(int num_eigvecs, int N, std::string filepath) {
    arma::mat A = construct_A(N);
    arma::vec eigvals;
    arma::mat eigvecs;

    jacobi_eigensolver(A, eigvals, eigvecs, 1.0e-8, 10000000);
    
    arma::mat table(N, num_eigvecs, arma::fill::zeros);
    arma::field<std::string> header(num_eigvecs);

    arma::uvec j = arma::sort_index(eigvals, "ascend");

    for (int i = 0; i < num_eigvecs; i++) {
        table.col(i) = eigvecs.col(j(i));
        header(i) = "eigenvector" + std::to_string(i);
    }
    
    std::cout << table;

    table.save(arma::csv_name(filepath, header));
}

void problem6() {
    int num_eigvecs = 3;
    save_n_last_eigvecs(num_eigvecs, 10, "./tables/three_last_eigvecs_10_by_10.csv");
    save_n_last_eigvecs(num_eigvecs, 100, "./tables/three_last_eigvecs_100_by_100.csv");
}
