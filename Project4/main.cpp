#include "omp.h"
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <armadillo>

using namespace arma;

arma::mat random_sign_matrix(unsigned int m, unsigned int n) {
    return 2.0*round(randu(m, n)) - 1.0;
}

double E(mat& L) {
    return -accu(L%(shift(L,1,0) + shift(L,1,1)));
}
double abs_M(mat& L) {
    return abs(accu(L));
}

mat advance_2x2(mat& L, unsigned int num_steps, double T=1.0) {
    int m = L.n_rows;
    int n = L.n_cols;
    
    assert(m == 2 and n == 2);
    
    mat Es_and_abs_Ms = zeros(num_steps+1, 2);
    Es_and_abs_Ms(0, 0) = E(L);
    Es_and_abs_Ms(0, 1) = abs_M(L);
    double sum_of_neighbors = 0.0;
    double dE = 0.0;
    double dabs_M = 0.0;
    double beta = 1.0/T;
    int i = 0;
    int j = 0;
    for (int t = 0; t < num_steps; t++) {
        i = randi(distr_param(0, m-1));
        j = randi(distr_param(0, n-1));

        sum_of_neighbors = L((i+3)%2, j) + L(i, (j+3)%2);

        dE = 2.0*L(i,j)*sum_of_neighbors;
        dabs_M = -2.0*L(i,j);
        if (dE <= 0.0 or exp(-beta*dE) >= randu()) {
            L(i,j) *= -1.0;
        }
        else {
            dE = 0.0;
            dabs_M = 0.0;
        }

        Es_and_abs_Ms(t+1, 0) = Es_and_abs_Ms(t, 0) + dE;
        Es_and_abs_Ms(t+1, 1) = abs(Es_and_abs_Ms(t, 1) + dabs_M);
    }

    return Es_and_abs_Ms;
}

mat advance_mxn(mat& L, unsigned int num_steps, double T=1.0, unsigned int seed=0) {
    int m = L.n_rows;
    int n = L.n_cols;
    
    assert(m >= 3 and n >= 3);
    
    mat Es_and_abs_Ms = zeros(num_steps+1, 2);
    Es_and_abs_Ms(0, 0) = E(L);
    Es_and_abs_Ms(0, 1) = abs_M(L);
    double sum_of_neighbors = 0.0;
    double dE = 0.0;
    double dabs_M = 0.0;
    double beta = 1.0/T;

    int i = 0;
    int j = 0;

    std::mt19937 i_prng;
    std::mt19937 j_prng;

    std::uniform_int_distribution<int> random_i(0, m-1);
    std::uniform_int_distribution<int> random_j(0, n-1);

    i_prng.seed(seed);
    j_prng.seed(seed+1);
    for (int t = 0; t < num_steps; t++) {
        i = random_i(i_prng);//randi(distr_param(0, m-1));
        j = random_j(j_prng);//randi(distr_param(0, n-1));

        sum_of_neighbors = (
            L((i+1+m)%m, j) +
            L((i-1+m)%m, j) + 
            L(i, (j+1+n)%n) +
            L(i, (j-1+n)%n)
        );
        dE = 2.0*L(i,j)*sum_of_neighbors;
        dabs_M = -2.0*L(i,j);
        if (dE <= 0.0 or exp(-beta*dE) >= randu()) {
            L(i,j) *= -1.0;
        }
        else {
            dE = 0.0;
            dabs_M = 0.0;
        }

        Es_and_abs_Ms(t+1, 0) = Es_and_abs_Ms(t, 0) + dE;
        Es_and_abs_Ms(t+1, 1) = abs(Es_and_abs_Ms(t, 1) + dabs_M);
    }

    return Es_and_abs_Ms;
}

mat advance(mat& L, unsigned int num_steps, double T=1.0, unsigned int seed=0) {
    assert(T > 0.0);

    int m = L.n_rows;
    int n = L.n_cols;

    assert((m == 2 and n == 2) or (m >= 3 and n >= 3));

    if (m == 2 and n == 2) {
        return advance_2x2(L, num_steps, T);
    }
    else {
        return advance_mxn(L, num_steps, T, seed);
    }
}

double mean(vec& v) {
    return accu(v)/v.n_elem;
}

void problem4() {
    arma::arma_rng::set_seed(2022);

    double Z = 2.0*exp(8.0) + 2.0*exp(-8.0) + 12.0; // = 4.0*cosh(8.0) + 12.0;
    double exact_E = 16.0*(exp(-8.0) - exp(8.0))/Z; // = -32.0*sinh(8.0)/Z;
    double exact_E2 = 128.0*(exp(8.0)+exp(-8.0))/Z; // = 256.0*cosh(8.0)/Z;
    double exact_M2 = (32.0*exp(8.0) + 32.0)/Z;
    double exact_abs_M = (8.0*exp(8.0) + 16.0)/Z;
    double exact_C_V = (exact_E2 - exact_E*exact_E)/4.0;
    double exact_chi = (exact_M2 - exact_abs_M*exact_abs_M)/4.0;

    mat L = random_sign_matrix(2, 2);
    mat Es_and_abs_Ms = advance_2x2(L, 100, 1.0); // 100 cycles gives good agreement with analytical results.
    vec Es = Es_and_abs_Ms.col(0);
    vec abs_Ms = Es_and_abs_Ms.col(1);
    double est_E = mean(Es);
    double est_E2 = mean(Es%Es);
    double est_abs_M = mean(abs_Ms);
    double est_M2 = mean(abs_Ms%abs_Ms);
    double est_C_V = (est_E2 - est_E*est_E)/Z;
    double est_chi = (est_M2 - est_abs_M*est_abs_M)/Z;

    double atol = 0.05;
    assert(fabs(est_E - exact_E) < atol);
    assert(fabs(est_abs_M - exact_abs_M) < atol);
    assert(fabs(est_C_V - exact_C_V) < atol);
    assert(fabs(est_chi - exact_chi) < atol);
}

void problem5() {
    unsigned int num_cycles = 100000;
    mat ordered_lattice = ones(20, 20);
    mat ordered_T_1point0 = advance(ordered_lattice, num_cycles, 1.0);
    ordered_lattice = ones(20, 20);
    mat ordered_T_2point4 = advance(ordered_lattice, num_cycles, 2.4);

    mat disordered_lattice = random_sign_matrix(20, 20);
    mat disordered_T_1point0 = advance(disordered_lattice, num_cycles, 1.0);
    disordered_lattice = random_sign_matrix(20, 20);
    mat disordered_T_2point4 = advance(disordered_lattice, num_cycles, 2.4);

    ordered_T_1point0 /= 400.0;
    ordered_T_2point4 /= 400.0;
    disordered_T_1point0 /= 400.0;
    disordered_T_2point4 /= 400.0;

    field<std::string> header(2);
    header(0) = "eps";
    header(1) = "abs_m";
    ordered_T_1point0.save(csv_name(
        "./data/prob5_ordered_T10.csv",
        header
    ));
    ordered_T_1point0.save(csv_name(
        "./data/prob5_ordered_T24.csv",
        header
    ));
    disordered_T_1point0.save(csv_name(
        "./data/prob5_disordered_T10.csv",
        header
    ));
    disordered_T_2point4.save(csv_name(
        "./data/prob5_disordered_T24.csv",
        header
    ));
}


void problem8() {
    //ivec m = ivec({40, 60, 80, 100});
    //vec T = vec({2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4});

    std::vector<double> T = {2.1, 2.2, 2.25, 2.3, 2.4};

    unsigned int base_seed = 2022;

    #pragma omp parallel for
    for (int i = 0; i < T.size(); i++) {//T.n_elem; i++) {
        int thread_id = omp_get_thread_num();
        std::vector<int> m = {40, 60, 80, 100};
        for (int j = 0; j < m.size(); j++) {//.n_elem; j++) {
            mat L = random_sign_matrix(m[j], m[j]);//m(j), m(j));
            mat data = advance(L, 100000, T[i], base_seed+2*(m.size()*thread_id + j));
            
            field<std::string> header(2);
            header(0) = "eps";
            header(1) = "abs_m";

            std::stringstream s;
            s << std::fixed << std::setprecision(2) << T[i];
            std::string filename = "prob8_T"+s.str()+"_L"+std::to_string(m[j]);
            data.save(csv_name("./data/"+filename+".csv", header));
        }
    }
}

int main() {
    arma::arma_rng::set_seed(2022);

    // problem4();
    //problem5();
    problem8();

    return 0;
}
