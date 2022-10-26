#ifndef PENNING_TRAP_H_
#define PENNING_TRAP_H_

#include "Particle.h"

#include <vector>
#include <armadillo>

class PenningTrap {
public:
    PenningTrap();
    PenningTrap(
        double magnetic_field_strength, 
        double electric_field_strength, 
        double characteristic_dimension
    );
    void set_amplitude(double amp);
    void set_frequency(double freq);
    void toggle_coulomb();
    void toggle_time_dependence();
    void add_particle(Particle particle);
    void add_random_particles(int num_particles);
    int get_num_particles_inside_trap();
    arma::vec E(arma::vec r, double t);
    arma::vec B(arma::vec r);
    arma::vec F_lorentz(int i, double t);
    arma::vec F_ij(int i, int j);
    arma::vec F_coulomb(int i);
    arma::vec F(int i, double t);
    arma::vec f(double t, arma::vec r, arma::vec v, double q, double m, int i);
    void forward_euler(int steps, double dt);
    void rk4(int steps, double dt);

    std::vector<Particle> particles;
private:
    double B0;
    double V0;
    double d;
    double t;
    double amp;
    double freq;
    bool is_time_dependent;
    bool is_coulomb_enabled;
};

#endif  // PENNING_TRAP_H_
