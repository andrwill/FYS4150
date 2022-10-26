#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <string>
#include <vector>
#include <armadillo>

class Particle {
public:
    double q;
    double m;
    arma::vec r;
    arma::vec v;
    arma::mat history;
    Particle(double charge, double mass, arma::vec position, arma::vec velocity);
    void save_history(std::string filepath);
};

#endif  // PARTICLE_H_
