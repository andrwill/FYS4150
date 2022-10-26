import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k_e = 1.38935333e5
T = 9.64852558e1
V = 9.64852558e7
q = 1.0
m = 40.078
d = 500.0
v0 = 25.0
x0 = 20.0
ydot0 = v0
z0 = 20.0
B0 = 9.65e1
V0 = 2.41e6
omega_0 = q*B0/m
omega_z = np.sqrt(2.0*q*V0/(m*d**2))
omega_p = 0.5*(omega_0 + np.sqrt(omega_0**2 - 2.0*omega_z**2))
omega_m = 0.5*(omega_0 - np.sqrt(omega_0**2 - 2.0*omega_z**2))
A_p = (v0 + omega_m*x0)/(omega_m - omega_p)
A_m = (v0 + omega_p*x0)/(omega_p - omega_m)

def f(t):
    return A_p*np.exp(-1.0j*omega_p*t) + A_m*np.exp(-1.0j*omega_m*t)

def z(t):
    return z0*np.cos(omega_z*t)

def r(t):
    _ = f(t)
    return np.vstack((_.real, _.imag, z(t))).T

def plot_testplot(): 
    r_fe = pd.read_csv("./data/one_particle/single_particle_forward_euler.csv")
    r_rk4 = pd.read_csv("./data/one_particle/single_particle_rk4.csv")

    t = np.linspace(0.0, 50.0, r_fe["z"].size)
    
    plt.title("Single Particle Motion in $z$-direction")
    plt.xlabel("$t$")
    plt.ylabel("$z$")
    plt.plot(t, r_fe["z"], label="Forward Euler")
    plt.plot(t, r_rk4["z"], label="Fourth-Order Runge Kutta")
    plt.plot(t, z(t), label="Exact")#, linestyle="dashed")

    #_ = f(t)
    #x, y = _.real, _.imag
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(x, y)
    # ax[1].plot(r_rk4["x"], r_rk4["y"])
    # plt.plot(r_fe["x"], r_fe["y"], label="Forward Euler")
    # plt.plot(r_rk4["x"], r_rk4["y"], label="RK4")
    # plt.plot(x, y, label="Exact")

    #ax = plt.axes(projection="3d")
    ##ax.plot(r_fe["x"], r_fe["y"], r_fe["z"], label="Forward Euler")
    #ax.plot(r_rk4["x"], r_rk4["y"], r_rk4["z"], label="RK4")
    #ax.plot(x, y, z(t), label="Exact")

    plt.legend()

def plot_single_particle():
    r = pd.read_csv("./data/one_particle/single_particle_rk4.csv")

    # ax = plt.axes(projection="3d")
    # ax.plot(r["x"], r["y"], r["z"], label="Trajectory", color="black")
    # ax.scatter(r["x"][0], r["y"][0], r["z"][0], color="cyan", label="Starting point")
    # ax.scatter(r["x"].iloc[-1], r["y"].iloc[-1], r["z"].iloc[-1], color="red", label="Endpoint")

    t = np.linspace(0.0, 50.0, r["z"].size)
    
    plt.title("Single Particle Motion in $z$-direction")
    plt.xlabel("$t$")
    plt.ylabel("$z$")
    plt.plot(t, r["z"], label="Approximate")
    plt.plot(t, z(t), label="Exact")#, linestyle="dashed")
    plt.legend()

    plt.savefig("./figures/single_particle_z_component.pdf")

def plot_two_particles_xy():
    r1_int = pd.read_csv("./data/two_particles_with_interactions/particle0.csv")
    r2_int = pd.read_csv("./data/two_particles_with_interactions/particle1.csv")

    r1_no_int = pd.read_csv("./data/two_particles_without_interactions/particle0.csv")
    r2_no_int = pd.read_csv("./data/two_particles_without_interactions/particle1.csv")

    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Two Particles")
    ax[0].set_title("With Interactions")
    ax[0].plot(r1_int["x"], r1_int["y"])
    ax[0].plot(r2_int["x"], r2_int["y"])

    ax[1].set_title("Without Interactions")
    ax[1].plot(r1_no_int["x"], r1_no_int["y"])
    ax[1].plot(r2_no_int["x"], r2_no_int["y"])

    plt.savefig("./figures/two_particles_xy_plane.pdf")

def plot_two_particles_phase_plane():
    r1_int = pd.read_csv("./data/two_particles_with_interactions/particle0.csv")
    r2_int = pd.read_csv("./data/two_particles_with_interactions/particle1.csv")
    v1_int = r1_int[1:]-r1_int[:-1]
    v2_int = r2_int[1:]-r2_int[:-1]

    r1_no_int = pd.read_csv("./data/two_particles_without_interactions/particle0.csv")
    r2_no_int = pd.read_csv("./data/two_particles_without_interactions/particle1.csv")
    v1_no_int = r1_no_int[1:]-r1_no_int[:-1]
    v2_no_int = r2_no_int[1:]-r2_no_int[:-1]

    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Two Particles")
    ax[0,0].set_title("Without Interaction")
    ax[0,0].set_xlabel("$x$")
    ax[0,0].set_ylabel("$v_x$")
    ax[0,0].plot(r1_no_int["x"], v1_no_int["x"], label="Particle 0")
    ax[0,0].plot(r2_no_int["x"], v2_no_int["x"], label="Particle 1")

    ax[1,0].set_xlabel("$z$")
    ax[1,0].set_ylabel("$v_z$")
    ax[1,0].plot(r1_no_int["z"], v1_no_int["z"], label="Particle 0")
    ax[1,0].plot(r2_no_int["z"], v2_no_int["z"], label="Particle 1")
    
    ax[0,1].set_title("With Interaction")

    ax[0,1].plot(r1_int["x"], v1_int["x"], label="Particle 0")
    ax[0,1].plot(r2_int["x"], v2_int["x"], label="Particle 1")

    ax[1,1].plot(r1_int["z"], v1_int["z"], label="Particle 0")
    ax[1,1].plot(r2_int["z"], v2_int["z"], label="Particle 1")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.savefig("./figures/two_particles_phase_plane.pdf")

def plot_two_particles_3d():
    r1_int = pd.read_csv("./data/two_particles_with_interactions/particle0.csv")
    r2_int = pd.read_csv("./data/two_particles_with_interactions/particle1.csv")

    r1_no_int = pd.read_csv("./data/two_particles_without_interactions/particle0.csv")
    r2_no_int = pd.read_csv("./data/two_particles_without_interactions/particle1.csv")

    fig = plt.figure()
    fig.suptitle("Two Particles")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("With Interactions")
    ax.plot(r1_int["x"], r1_int["y"], r1_int["z"])
    ax.plot(r2_int["x"], r2_int["y"], r2_int["z"])

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_title("Without Interactions")
    ax.plot(r1_no_int["x"], r1_no_int["y"], r1_no_int["z"])
    ax.plot(r2_no_int["x"], r2_no_int["y"], r2_no_int["z"])

    plt.savefig("./figures/two_particles_3d.pdf")


def plot_single_particle_errors():
    folder = "./data/one_particle/"

    ns = [ 4000, 8000, 16000, 32000 ]

    r_forward_eulers = [ pd.read_csv(folder+f"forward_euler_{n}.csv").to_numpy() for n in ns ]
    r_rk4s = [ pd.read_csv(folder+f"rk4_{n}.csv").to_numpy() for n in ns ]

    ts = [ np.linspace(0.0, 50.0, r.shape[0]) for r in r_rk4s ]
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle("Single Particle Relative Errors")
    ax[0].set_title("Forward Euler")
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("Absolute Error")

    ax[1].set_title("Fourth-Order Runge Kutta Method")
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("Absolute Error")

    max_err_fe = np.zeros(4)
    max_err_rk4 = np.zeros(4)

    dt = np.array([ 50.0/n for n in ns ])

    for i in range(len(ns)):
        t = ts[i]
        n = ns[i]
        r_exact = r(t)
        r_fe = r_forward_eulers[i]
        r_rk4 = r_rk4s[i]

        abs_err_fe = np.apply_along_axis(
            np.linalg.norm, 
            1, 
            r_exact - r_fe
        )
        abs_err_rk4 = np.apply_along_axis(
            np.linalg.norm, 
            1, 
            r_exact - r_rk4
        )
        ax[0].plot(t, abs_err_fe/np.linalg.norm(r_exact), label=f"n = {n}")
        ax[1].plot(t, abs_err_rk4/np.linalg.norm(r_exact), label=f"n = {n}")

        max_err_fe[i] = np.max(abs_err_fe)
        max_err_rk4[i] = np.max(abs_err_rk4)

    r_err_fe = 0.0
    r_err_rk4 = 0.0
    for i in range(1,len(ns)):
        r_err_fe += np.log(max_err_fe[i]/max_err_fe[i-1])/np.log(dt[i]/dt[i-1])
        r_err_rk4 += np.log(max_err_rk4[i]/max_err_rk4[i-1])/np.log(dt[i]/dt[i-1])

    r_err_fe /= 3.0
    r_err_rk4 /= 3.0

    print("Estimated Error Convergence Rate")
    print(f"Forward Euler: {r_err_fe}")
    print(f"Fourth-Order Runge Kutta: {r_err_rk4}")
    ax[0].legend()
    ax[1].legend()

    plt.savefig("./figures/single_particle_relative_errors.pdf")


def plot_num_particles_in_trap(title, infile, outfile):
    df = pd.read_csv(infile)
    frequencies = df["Frequencies"]

    plt.title(title)

    print(df.columns.values)

    for label in df.columns.values[1:]:
        amplitude = float(label[-9:-1])
        num_particles = df[label]

        plt.plot(frequencies, num_particles, label=f"Amplitude = {amplitude}")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Number of Particles")
    plt.legend()

    plt.savefig(outfile)

def plot_num_particles_without_coulomb():

    plt.savefig("./figures/")
def plot_num_particles_with_coulomb():
    plt.savefig("")

if __name__ == "__main__":
    #plot_single_particle()
    #plot_two_particles_xy()
    #plot_two_particles_phase_plane()
    #plot_two_particles_3d()
    plot_single_particle_errors()
    #plot_num_particles_in_trap(
    #    "Number of Particles After 500 μs",
    #    "./data/multiple_particles/num_particles_in_trap.csv",
    #    "./figures/num_particles_in_penning_trap.pdf"
    #)
    #plot_num_particles_in_trap(
    #    "Number of Particles After 500 μs (without interactions)",
    #    "./data/multiple_particles/explore_resonance_without_coulomb.csv",
    #    "./figures/num_particles__coulomb_off.pdf"
    #)
    #plot_num_particles_in_trap(
    #    "Number of Particles After 500 μs (with interactions)",
    #    "./data/multiple_particles/explore_resonance_with_coulomb.csv",
    #    "./figures/num_particles__coulomb_on.pdf"
    #)
    plt.show()
