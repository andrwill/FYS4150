import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    k_e = 1.38935333e5
    T = 9.64852558e1
    V = 9.64852558e7
    B0 = 9.65e1
    V0 = 2.41e6
    d = 500    

    v0 = 2.0
    x0 = 1.0
    y0 = v0
    z0 = 3.0

    q = 1.0
    m = 1.0

    omega_0 = q*B0/m
    omega_z = np.sqrt(2.0*q*V0/(m*d**2))
    omega_p = 0.5*(omega_0 + np.sqrt(omega_0**2 - 2.0*omega_z**2))
    omega_m = 0.5*(omega_0 - np.sqrt(omega_0**2 - 2.0*omega_z**2))

    a_p = (v0 + omega_m*x0)/(omega_m - omega_p)
    a_m = (v0 + omega_p*x0)/(omega_p - omega_m)

    steps = 1000
    t_start, t_end = 0.0, 50
    t = np.linspace(t_start, t_end, steps)

    def f(t):
        return a_p*np.exp(-1.0j*omega_p*t)+a_m*np.exp(-1.0j*omega_m*t)
    def z(t):
        return z0*np.cos(omega_z*t)

    f = f(t)
    x, y = f.real, f.imag
    z = z(t)

    ax = plt.axes(projection="3d")

    plt.title("Single-Particle Analytical Solution")
    ax.scatter3D(x[0], y[0], z[0], label="Starting point", s=40, c="lightgreen")
    plt.plot(x, y, z, label="Trajectory")
    ax.scatter3D(x[-1], y[-1], z[-1], label="Endpoint", s=40, color="r")
    plt.legend()
    plt.savefig("./figures/single-particle_analytical_solution.pdf")
    plt.show()
