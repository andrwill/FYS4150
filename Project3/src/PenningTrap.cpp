#include "Particle.h"
#include "PenningTrap.h"

#include <vector>
#include <ctgmath>
#include <armadillo>

double const k_e = 1.38935333e5;

PenningTrap::PenningTrap() {
    this->B0 = 9.65e1;
    this->V0 = 2.41e6;
    this->d = 500.0;
    this->is_time_dependent = false;
    this->is_coulomb_enabled = false;
    this->t = 0.0;
    this->amp = 0.0;
    this->freq = 0.0;
}
PenningTrap::PenningTrap(
    double magnetic_field_strength, 
    double electric_field_strength, 
    double characteristic_dimension
) {
    this->B0 = magnetic_field_strength;
    this->V0 = electric_field_strength;
    this->d = characteristic_dimension;
    this->is_time_dependent = false;
    this->is_coulomb_enabled = false;
    this->t = 0.0;
    this->amp = 0.0;
    this->freq = 0.0;
}
void PenningTrap::set_amplitude(double amp) {
    this->amp = amp;
}
void PenningTrap::set_frequency(double freq) {
    this->freq = freq;
}
int PenningTrap::get_num_particles_inside_trap() {
    int num_particles_inside_trap = 0;
    for (int i = 0; i < this->particles.size(); i++) {
        if (arma::norm(this->particles[i].r > this->d)) {
            num_particles_inside_trap++;
        }
    }
    return num_particles_inside_trap;
}
void PenningTrap::add_random_particles(int num_particles) {
    arma::vec r = arma::zeros(3);
    arma::vec v = arma::zeros(3);
    for (int i = 0; i < num_particles; i++) {
        r = 0.1*this->d*arma::vec(3).randn();
        v = 0.1*this->d*arma::vec(3).randn();
        this->particles.push_back(Particle(1.0, 40.078, r, v));
    }
}
void PenningTrap::toggle_coulomb() {
    this->is_coulomb_enabled = not this->is_coulomb_enabled;
}
void PenningTrap::toggle_time_dependence() {
    this->is_time_dependent = not this->is_time_dependent;
}
void PenningTrap::add_particle(Particle particle) {
    this->particles.push_back(particle);
}
arma::vec PenningTrap::E(arma::vec r, double t) {
    double d = this->d;
    if (arma::norm(r) > d) {
        return arma::zeros(3);
    }

    double _V0 = this->V0;
    if (this->is_time_dependent) {
        _V0 *= (1.0 + this->amp*cos(this->freq*t));
    }

    return 0.5*_V0/(d*d)*arma::vec({r[0], r[1], -2.0*r[2]});
}
arma::vec PenningTrap::B(arma::vec r) {
    if (arma::norm(r) > this->d) {
        return arma::zeros(3);
    }

    return arma::vec({0.0, 0.0, this->B0});
}
arma::vec PenningTrap::F_lorentz(int i, double t) {
    Particle& p = this->particles[i];
    double q = p.q;
    arma::vec r = p.r;
    arma::vec v = p.v;

    return q*this->E(r, t) + q*arma::cross(v, this->B(r));
}
arma::vec PenningTrap::F_ij(int i, int j) {
    Particle& p_i = this->particles[i];
    Particle& p_j = this->particles[j];

    double q1 = p_i.q;
    double q2 = p_j.q;
    arma::vec r1 = p_i.r;
    arma::vec r2 = p_j.r;

    double r = arma::norm(r1-r2);

    return k_e*q1*q2/(r*r*r) * (r1 - r2);
}
arma::vec PenningTrap::F_coulomb(int i) {
    arma::vec F = arma::zeros(3);
    for (int j = 0; j < i; j++) {
        F += this->F_ij(i,j);
    }
    for (int j = i+1; j < this->particles.size(); j++) {
        F += this->F_ij(i,j);
    }

    return F;
}
arma::vec PenningTrap::F(int i, double t) {
    arma::vec total_force = this->F_lorentz(i, t);
    if (this->is_coulomb_enabled) {
        total_force += this->F_coulomb(i);
    }
    return total_force;
}
arma::vec PenningTrap::f(double t, arma::vec r, arma::vec v, double q, double m, int i) {
    arma::vec f_lorentz = q*(this->E(r, t) + arma::cross(v, this->B(r)));

    double other_q = 0.0;
    double R = 0.0;
    arma::vec other_r = arma::zeros(3);
    arma::vec f_coulomb = arma::zeros(3);

    if (this->is_coulomb_enabled) {
        for (int j = 0; j < i; j++) {
            other_q = this->particles[j].q;
            other_r = this->particles[j].r;
            R = arma::norm(r - other_r);
            f_coulomb += k_e*q*other_q/(R*R*R) * (r - other_r);
        }
        for (int j = i+1; j < this->particles.size(); j++) {
            other_r = this->particles[j].r;
            other_q = this->particles[j].q;
            R = arma::norm(r - other_r);
            f_coulomb += k_e*q*other_q/(R*R*R) * (r - other_r);
        }
    }

    return 1.0/m * (f_coulomb + f_lorentz);
}
void PenningTrap::forward_euler(int steps, double dt) {
    int n = this->particles.size();

    int l = this->particles[0].history.n_rows;
    for (int i = 0; i < n; i++) {
        this->particles[i].history.reshape(l+steps, 3);
    }

    double t = this->t;
    for (int h = 0; h < steps; h++) {
        for (int i = 0; i < n; i++) {
            this->particles[i].r += dt*this->particles[i].v;
            this->particles[i].v += dt/this->particles[i].m * this->F(i, t);
            this->particles[i].history.row(l+h) = this->particles[i].r.t();
        }
        t += dt;
    }

    this->t = t;
}
void PenningTrap::rk4(int steps, double dt) {
    int n = this->particles.size();

    int l = this->particles[0].history.n_rows;
    for (int i = 0; i < n; i++) {
        this->particles[i].history.reshape(l+steps, 3);
    }

    double t = this->t;
    double q = 0.0;
    double m = 0.0;
    arma::vec r = arma::zeros(3);
    arma::vec v = arma::zeros(3);
    arma::vec k1 = arma::zeros(3);
    arma::vec k2 = arma::zeros(3);
    arma::vec k3 = arma::zeros(3);
    arma::vec k4 = arma::zeros(3);
    for (int h = 0; h < steps; h++) {
        for (int i = 0; i < n; i++) {
            q = this->particles[i].q;
            m = this->particles[i].m;
            r = this->particles[i].r;
            v = this->particles[i].v;

            k1 = this->f(t, r, v, q, m, i);
            k2 = this->f(t + 0.5*dt, r + 0.5*dt*v, v + 0.5*dt*k1, q, m, i);
            k3 = this->f(t + 0.5*dt, r + 0.5*dt*v, v + 0.5*dt*k2, q, m, i);
            k4 = this->f(t + dt, r + dt*v, v + dt*k3, q, m, i);

            this->particles[i].r += dt*this->particles[i].v;
            this->particles[i].v += dt/6.0 *(k1 + 2.0*k2 + 2.0*k3 + k4);
            this->particles[i].history.row(l+h) = this->particles[i].r.t();
        }
        t += dt;
    }

    this->t = t;        
}
