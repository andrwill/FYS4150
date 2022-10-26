#include "Particle.h"

#include <armadillo>

Particle::Particle(
    double charge, 
    double mass, 
    arma::vec position, 
    arma::vec velocity
) {
    this->q = charge;
    this->m = mass;
    this->r = position;
    this->v = velocity;

    this->history = arma::zeros(1,3);
    this->history.row(0) = this->r.t();
}

void Particle::save_history(std::string filepath) {
    arma::field<std::string> header(3);
    header(0) = "x";
    header(1) = "y";
    header(2) = "z";
    this->history.save(arma::csv_name(filepath, header));
}
