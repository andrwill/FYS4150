#include "solvers.h"

#include <ctime>
#include <ctgmath> // For `pow`.
#include <armadillo>

arma::vec f(arma::vec& x) {
    return 100.0*arma::exp(-10.0*x);
}

void export_approximation_data() {
    int const I = 7;

    int const M = (int)(pow(10, I)); // 10^I.

    arma::field<std::string> header(I);
    arma::sp_mat data(M, I);

    int m = 10;
    for (int i = 0; i < I; i++) {
        header(i) = "v" + std::to_string(m);
        data.submat(1, i, m-2, i) = solve_poisson1d_g(&f, m);

        m *= 10;
    }
    data.save(arma::csv_name("./data/approximations.csv", header));
}

double time_general_solver(int m, int const reps = 10) {
    clock_t prev;
    clock_t curr;
    double total_elapsed_time = 0.0;
    for (int i = 0; i < reps; i++) {
        prev = clock();
        solve_poisson1d_g(&f, m);
        curr = clock();
        total_elapsed_time += ((double)(curr - prev)) / CLOCKS_PER_SEC;
    }

    double average_duration = total_elapsed_time / (double)(reps);

    return average_duration; // (in seconds)
}

double time_special_solver(int m, int const reps = 10) {
    clock_t prev;
    clock_t curr;
    double total_elapsed_time = 0.0;
    for (int i = 0; i < reps; i++) {
        prev = clock();
        solve_poisson1d_s(&f, m);
        curr = clock();
        total_elapsed_time += (double)(curr - prev) / (double)(CLOCKS_PER_SEC);
    }
    
    double average_duration = total_elapsed_time / (double)(reps);

    return average_duration; // (in seconds)
}

void export_runtime_data() {
    int num_solvers = 2;
    int max_exponent = 6;

    arma::field<std::string> runtimes_header(num_solvers+1);
    runtimes_header(0) = "Number of steps";
    runtimes_header(1) = "General solver";
    runtimes_header(2) = "Specialized solver";
    arma::mat runtimes(max_exponent, num_solvers+1);

    int n_steps;
    for (int i = 0; i < max_exponent; i++) {
        n_steps = (int)(pow(10, i+1));
        runtimes(i,0) = n_steps;
        runtimes(i,1) = time_general_solver(n_steps);
        runtimes(i,2) = time_special_solver(n_steps);
    }
    runtimes.save(arma::csv_name("./data/runtimes.csv", runtimes_header));
}

int main() {
    export_approximation_data();

    export_runtime_data();

    return 0;
}
