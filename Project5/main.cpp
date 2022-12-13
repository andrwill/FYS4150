#include <complex>
#include <iostream>

#include <armadillo>

using namespace arma;
using namespace std::complex_literals;

void fill_A_and_B(sp_cx_mat& A, sp_cx_mat& B,  mat& V, unsigned int M, double h, double dt) {
    int N = M-2;
    int K = N*N;
    cx_double r = 1.0i*dt/(h*h);

    vec v = vectorise(V.t());

    A.diag(N).fill(-r);
    A.diag(1).fill(-r);
    A.diag(0) = (1.0 + 4.0*r)*ones(K) + 0.5i*dt*v;
    A.diag(-1).fill(-r);
    A.diag(-N).fill(-r);
    
    B.diag(N).fill(r);
    B.diag(1).fill(r);
    B.diag(0) = (1.0 - 4.0*r)*ones(K) - 0.5i*dt*v;
    B.diag(-1).fill(r);
    B.diag(-N).fill(r);
}

void update(cx_vec& u, sp_cx_mat& A, sp_cx_mat& B, cx_vec b) {
    b = B*u; // `*` means matrix multiplication in Armadillo.
    u = spsolve(A, b, "superlu");
}

// Function for specifying the initial state
cx_vec initial_state(
    vec& x, vec& y, 
    double x_c, double y_c, 
    double p_x, double p_y, 
    double s_x, double s_y
) {
    // Auxiliary variables for readability
    vec a = -((x-x_c)%(x-x_c))/(2.0*s_x*s_x) - ((y-y_c)%(y-y_c))/(2.0*s_y*s_y);
    vec b = p_x*(x-x_c) + p_y*(y-y_c);
    
    cx_vec u0 = exp(a + 1.0i*b);
    return u0/norm(u0); // Normalize u0
}

void init_single_slit_potential(
    mat& V,
    double v0=1.0e8, 
    double wall_thickness=0.02, 
    double wall_x_coordinate=0.5,
    double wall_piece_length=0.05,
    double aperture=0.05
) {
    // Translate coordinates to indices
    int x = int(wall_x_coordinate*((double)(V.n_cols)));
    int y = int(0.5*((double)(V.n_rows)));

    int dx = int(0.5*wall_thickness*((double)(V.n_cols)));
    int dy = int(0.5*aperture*((double)(V.n_rows)));
    int wall_length = int(wall_piece_length*((double)(V.n_rows)));

    V.submat(0, x-dx, y-dy, x+dx).fill(v0); // Upper wall
    V.submat(y+dy, x-dx, V.n_rows-1, x+dx).fill(v0); // Lower wall
}

void init_double_slit_potential(
    mat& V,
    double v0=1.0e8, 
    double wall_thickness=0.02, 
    double wall_x_coordinate=0.5,
    double wall_piece_length=0.05,
    double aperture=0.05
) {
    int x = int(wall_x_coordinate*((double)(V.n_cols)));
    int y = int(0.5*((double)(V.n_rows)));

    int dx = int(0.5*wall_thickness*((double)(V.n_cols)));
    int dy = int(0.5*aperture*((double)(V.n_rows)));
    int wall_length = int(wall_piece_length*((double)(V.n_rows)));

    V.submat(0, x-dx, y-dy-wall_length, x+dx).fill(v0); // Upper wall
    V.submat(y-dy, x-dx, y+dy, x+dx).fill(v0); // Separating wall piece
    V.submat(y+dy+wall_length, x-dx, V.n_rows-1, x+dx).fill(v0); // Lower wall
}

void init_triple_slit_potential(
    mat& V,
    double v0=1.0e8, 
    double wall_thickness=0.02, 
    double wall_x_coordinate=0.5,
    double wall_piece_length=0.05,
    double aperture=0.05
) {
    int x = int(wall_x_coordinate*((double)(V.n_cols)));
    int y = int(0.5*((double)(V.n_rows)));

    int dx = int(0.5*wall_thickness*((double)(V.n_cols)));
    int dy = int(0.5*aperture*((double)(V.n_rows)));
    int wall_length = int(wall_piece_length*((double)(V.n_rows)));

    V.submat(0, x-dx, y-3.0*dy-wall_length, x+dx).fill(v0); // Upper wall
    V.submat(y-dy-wall_length, x-dx, y-dy, x+dx).fill(v0); // Upper separating wall piece
    V.submat(y+dy, x-dx, y+dy+wall_length, x+dx).fill(v0); // Lower separating wall piece
    V.submat(y+3.0*dy+wall_length, x-dx, V.n_rows-1, x+dx).fill(v0); // Lower wall
}

// Struct for storing the simulation parameters
typedef struct {
    std::string outfile;
    std::string potential;
    double v0;
    double h;
    double dt; 
    double T;
    double x_c;
    double y_c;
    double p_x;
    double p_y;
    double s_x;
    double s_y;
} sim_params;

// Function for running the simulation with specific parameters
void run_simulation(sim_params& params) {
    std::string outfile = params.outfile;
    std::string potential = params.potential;
    double v0 = params.v0;
    double h = params.h;
    double dt = params.dt;
    double T = params.T;
    double x_c = params.x_c;
    double y_c = params.y_c;
    double p_x = params.p_x;
    double p_y = params.p_y;
    double s_x = params.s_x;
    double s_y = params.s_y;

    unsigned int M = (int)(1.0/h);
    unsigned int N = M-2;
    unsigned int K = N*N;
    mat V = zeros(N, N);

    // Initialize the boundary of the unit square
    V.submat(0, 0, 1, N-1).fill(1.0e10);
    V.submat(0, 0, N-1, 1).fill(1.0e10);
    V.submat(0, N-2, N-1, N-1).fill(1.0e10);
    V.submat(N-2, 0, N-1, N-1).fill(1.0e10);

    if (potential == "single-slit") {
        init_single_slit_potential(V, v0);
    }
    else if (potential == "double-slit") {
        init_double_slit_potential(V, v0);
    }
    else if (potential == "triple-slit") {
        init_triple_slit_potential(V, v0);
    }

    // Make a meshgrid for x and y
    vec unit_interval = linspace(0.0, 1.0, N+2).subvec(1,N);
    mat xy = zeros(N,N);
    for (int i = 0; i < N; i++) {
        xy.col(i) = unit_interval;
    }
    vec x = vectorise(xy);
    vec y = vectorise(xy.t());

    // Initialize u = u(x, y, 0)
    cx_vec u = initial_state(x, y, x_c, y_c, p_x, p_y, s_x, s_y);

    // Create the matrices A and B needed by the Crank-Nicholson scheme
    sp_cx_mat A(K, K);
    sp_cx_mat B(K, K);
    fill_A_and_B(A, B, V, M, h, dt);

    // Run the simulation and store each timestep in U
    int total_num_iters = (int)(T/dt);
    cx_mat U(K, total_num_iters);
    for (int i = 0; i < total_num_iters; i++) {
        u = spsolve(A, B*u, "superlu");
        U.col(i) = u;
        std::cout << i << std::endl;
    }

    U.save(outfile, csv_ascii);
}

int main() {
    sim_params zero_potential_sim;
    zero_potential_sim.outfile = "./data/zero_potential_simulation.csv";
    zero_potential_sim.potential = "double-slit";
    zero_potential_sim.v0 = 0.0;
    zero_potential_sim.h = 0.005;
    zero_potential_sim.dt = 2.5e-5; 
    zero_potential_sim.T = 0.008;
    zero_potential_sim.x_c = 0.25;
    zero_potential_sim.y_c = 0.5;
    zero_potential_sim.p_x = 200.0;
    zero_potential_sim.p_y = 0.0;
    zero_potential_sim.s_x = 0.05;
    zero_potential_sim.s_y = 0.05;

    sim_params double_slit_sim;
    double_slit_sim.outfile = "./data/double_slit_simulation.csv";
    double_slit_sim.potential = "double-slit";
    double_slit_sim.v0 = 1.0e10;
    double_slit_sim.h = 0.005;
    double_slit_sim.dt = 2.5e-5; 
    double_slit_sim.T = 0.008;
    double_slit_sim.x_c = 0.25;
    double_slit_sim.y_c = 0.5;
    double_slit_sim.p_x = 200.0;
    double_slit_sim.p_y = 0.0;
    double_slit_sim.s_x = 0.05;
    double_slit_sim.s_y = 0.1;

    sim_params problem8_sim;
    problem8_sim.outfile = "./data/problem8_simulation.csv";
    problem8_sim.potential="double-slit";
    problem8_sim.v0 = 1.0e10;
    problem8_sim.h = 0.005;
    problem8_sim.dt = 2.5e-5; 
    problem8_sim.T = 0.008;
    problem8_sim.x_c = 0.25;
    problem8_sim.y_c = 0.5;
    problem8_sim.p_x = 200.0;
    problem8_sim.p_y = 0.0;
    problem8_sim.s_x = 0.05;
    problem8_sim.s_y = 0.20;

    sim_params single_slit_sim;
    single_slit_sim.outfile = "./data/single_slit_simulation.csv";
    single_slit_sim.potential="single-slit";
    single_slit_sim.v0 = 1.0e10;
    single_slit_sim.h = 0.005;
    single_slit_sim.dt = 2.5e-5; 
    single_slit_sim.T = 0.008;
    single_slit_sim.x_c = 0.25;
    single_slit_sim.y_c = 0.5;
    single_slit_sim.p_x = 200.0;
    single_slit_sim.p_y = 0.0;
    single_slit_sim.s_x = 0.05;
    single_slit_sim.s_y = 0.1;

    sim_params triple_slit_sim;
    triple_slit_sim.outfile = "./data/triple_slit_simulation.csv";
    triple_slit_sim.potential="triple-slit";
    triple_slit_sim.v0 = 1.0e10;
    triple_slit_sim.h = 0.005;
    triple_slit_sim.dt = 2.5e-5; 
    triple_slit_sim.T = 0.008;
    triple_slit_sim.x_c = 0.25;
    triple_slit_sim.y_c = 0.5;
    triple_slit_sim.p_x = 200.0;
    triple_slit_sim.p_y = 0.0;
    triple_slit_sim.s_x = 0.05;
    triple_slit_sim.s_y = 0.1;

    // Problem 7
    run_simulation(zero_potential_sim);
    run_simulation(double_slit_sim);

    // Problem 8
    run_simulation(problem8_sim);

    // Problem 9
    run_simulation(single_slit_sim);
    run_simulation(triple_slit_sim);

    return 0;
}
