#include "Particle.h"
#include "PenningTrap.h"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <armadillo>

double const T = 9.64852558;
double const V = 9.64852558e7;
double const B0 = 9.65e1;
double const V0 = 2.41e6;
double const d = 500;


void problem_8() {
    arma::vec r1 = arma::vec({20.0, 0.0, 20.0});
    arma::vec v1 = arma::vec({0.0, 25.0, 0.0});
    arma::vec r2 = arma::vec({25.0, 25.0, 0.0});
    arma::vec v2 = arma::vec({0.0, 40.0, 5.0});

    double q = 1.0;
    double m = 40.078;
    double dt = 0.01;
    double duration = 50.0;
    int n = int(duration/dt);

    Particle particle1 = Particle(q, m, r1, v1);
    Particle particle2 = Particle(q, m, r2, v2);

    // Single particle
    PenningTrap penning_trap = PenningTrap();
    penning_trap.add_particle(particle1);
    penning_trap.forward_euler(n, dt);
    penning_trap.particles[0].save_history("./data/one_particle/single_particle_forward_euler.csv");

    penning_trap = PenningTrap();
    penning_trap.add_particle(particle1);
    penning_trap.rk4(n, dt);
    penning_trap.particles[0].save_history("./data/one_particle/single_particle_rk4.csv");



    // Two particles without interations
    penning_trap = PenningTrap();
    penning_trap.add_particle(particle1);
    penning_trap.add_particle(particle2);
    penning_trap.rk4(50, dt);
    penning_trap.particles[0].save_history("./data/two_particles_without_interactions/particle0.csv");
    penning_trap.particles[1].save_history("./data/two_particles_without_interactions/particle1.csv");

    // Two particles with interations
    penning_trap = PenningTrap();
    penning_trap.toggle_coulomb();
    penning_trap.add_particle(particle1);
    penning_trap.add_particle(particle2);
    penning_trap.rk4(50, dt);
    penning_trap.particles[0].save_history("./data/two_particles_with_interactions/particle0.csv");
    penning_trap.particles[1].save_history("./data/two_particles_with_interactions/particle1.csv");

    // Single particle for different ns
    std::vector<int> ns = {4000, 8000, 16000, 32000};
    std::string path = "./data/one_particle/";
    for (int i = 0; i < ns.size(); i++) {
        n = ns[i];
        dt = duration/((double)(n));

        // Forward Euler
        penning_trap = PenningTrap();
        penning_trap.add_particle(particle1);
        penning_trap.forward_euler(n, dt);
        penning_trap.particles[0].save_history(path+"forward_euler_"+std::to_string(n) +".csv");

        // Fourth-Order Runge-Kutta
        penning_trap = PenningTrap();
        penning_trap.add_particle(particle1);
        penning_trap.rk4(n, dt);
        penning_trap.particles[0].save_history(path+"rk4_"+std::to_string(n) +".csv");
    }
}

void perform_frequency_scan(arma::vec freqs, std::string outfile, bool coulomb) {
    PenningTrap penning_trap = PenningTrap();
    arma::vec amps = arma::vec({0.1, 0.4, 0.7});
    
    arma::mat graphs = arma::zeros(freqs.n_elem, amps.n_elem+1);
    graphs.col(0) = freqs;
    for (int j = 0; j < amps.n_elem; j++) {
        for (int i = 0; i < freqs.n_elem; i++) {
            penning_trap.add_random_particles(100);
            penning_trap.set_amplitude(amps(j));
            penning_trap.set_frequency(freqs(i));
            penning_trap.toggle_time_dependence();
            if (coulomb) {
                penning_trap.toggle_coulomb();
            }

            penning_trap.rk4(5000, 0.1);

            graphs(i,j+1) = penning_trap.get_num_particles_inside_trap();

            std::cout << "(" << std::to_string(i) << ", " << std::to_string(j) << ")" << std::endl;
            penning_trap = PenningTrap();
        }
    }
    arma::field<std::string> header(graphs.n_cols);
    header(0) = "Frequencies";
    for (int i = 0; i < amps.n_elem; i++) {
        header(i+1) = "Remaining particles (amplitude "+std::to_string(amps(i))+")";
    }
    graphs.save(arma::csv_name(outfile, header));
}

void problem9_frequency_scan() {
    std::string outfile = "./data/multiple_particles/num_particles_in_trap.csv";
    arma::vec frequencies = arma::linspace(0.2, 2.5, 20);
    perform_frequency_scan(frequencies, outfile, false);
}
void problem9_resonance_scan() {
    double resonance = 1.61;
    std::string folder = "./data/multiple_particles/";
    arma::vec freqs = arma::linspace(resonance-0.2, resonance+0.2, 5);
    //perform_frequency_scan(freqs, folder+"explore_resonance_without_coulomb.csv", false);
    perform_frequency_scan(freqs, folder+"explore_resonance_with_coulomb.csv", true);
}

void problem9() {
    //problem9_frequency_scan();
    problem9_resonance_scan();
    // Explore chosen resonance.
   
    // perform_frequency_scan(frequencies, folder+"explore_resonance_with_coulomb.csv", true);
}

int main() {
    arma::arma_rng::set_seed(2022);
    // problem_8();
    // problem9();

    return 0;
}
