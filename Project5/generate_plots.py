import numpy as np
import pyarma as arma
import matplotlib.pyplot as plt

from math import sqrt

def load_cx_mat(filename):
    U = arma.cx_mat()
    U.load(filename, arma.csv_ascii)
    U = np.array(U)
    return U

def problem7():
    U = load_cx_mat('./data/zero_potential_simulation.csv')
    abs_err_zero_potential = np.abs(1.0 - np.linalg.norm(U, axis=0))

    U = load_cx_mat('./data/double_slit_simulation.csv')
    abs_err_double_slit = np.abs(1.0 - np.linalg.norm(U, axis=0))

    t = np.linspace(0.0, 0.008, U.shape[1])

    plt.title('Deviation of Total Probability from 1')
    plt.xlabel('$t$')
    plt.ylabel('$|1 - \Sigma_{i,j} p_{i,j}|$')
    plt.ylim(0.0, 1.1*np.maximum(np.max(abs_err_zero_potential), np.max(abs_err_double_slit)))
    plt.plot(t, abs_err_zero_potential, label='Zero potential')
    plt.plot(t, abs_err_double_slit, label='Double-slit potential')
    plt.legend()
    plt.savefig('./figures/prob_dev_from_1.pdf')
    plt.show()

def problem8():
    U = load_cx_mat('./data/problem8_simulation.csv')
    t = np.linspace(0.0, 0.008, U.shape[1])

    n = round(sqrt(U.shape[0]))
    
    def plot_prob_dist(t0):
        i = np.where(np.abs(t - t0) < 2.5e-5)[0][0]

        u = U[:,i]
        P = (np.conj(u)*u).real.reshape(n, n)

        plt.title(f'Probability Distribution at $t = {t0}$')
        plt.axis('off')
        plt.imshow(P)
        plt.savefig(f'./figures/prob_dist_t{t0:.3f}.pdf')
        plt.show()

    def plot_re_u(t0):
        i = np.where(np.abs(t - t0) < 2.5e-5)[0][0]

        u = U[:,i]

        plt.title(f'Real Component at $t = {t0}$')
        plt.axis('off')
        plt.imshow(u.real.reshape(n, n))
        plt.savefig(f'./figures/u_real_t{t0:.3f}.pdf')
        plt.show()

    def plot_im_u(t0):
        i = np.where(np.abs(t - t0) < 2.5e-5)[0][0]

        u = U[:,i]

        plt.title(f'Imaginary Component at $t = {t0}$')
        plt.axis('off')
        plt.imshow(u.imag.reshape(n, n))
        plt.savefig(f'./figures/u_imag_t{t0:.3f}.pdf')
        plt.show()

    t0s = [0.0, 0.001, 0.002]

    for t0 in t0s:
        plot_prob_dist(t0)
        plot_re_u(t0)
        plot_im_u(t0)

def problem9():
    def plot_p_y(x, t, potential, save_plot=True):
        U = load_cx_mat('./data/'+potential+'_simulation.csv')
        n = round(sqrt(U.shape[0]))
        ts = np.arange(0.0, 0.002+2.5e-5, 2.5e-5)
        i = np.where(np.abs(ts - t) < 2.5e-5)[0][0]
        
        u = U[:,i].reshape(n, n)
        
        u = u[:,int(x*n)]
        u /= np.linalg.norm(u)
        
        p = (np.conj(u)*u).real
        assert np.isclose(np.sum(p), 1.0)

        y = np.linspace(0.0, 1.0, n+2)

        formatted_potential = potential.replace("_", " ").title()
        plt.title(f'$p(y|x={x:.1f},t={t:.4f})$ ({formatted_potential} Potential)')
        plt.xlabel('y')
        plt.ylabel(f'p(y|x={x:.1f},t={t:.4f})')
        plt.plot(y[1:-1], p)
        if save_plot:
            plt.savefig(f'./figures/{potential}_simulation_{t:.4f}.pdf')
        plt.show()

    plot_p_y(x=0.8, t=0.0016, potential='single_slit')
    plot_p_y(x=0.8, t=0.0016, potential='double_slit')
    plot_p_y(x=0.8, t=0.0016, potential='triple_slit')

    plot_p_y(x=0.8, t=0.002, potential='single_slit')
    plot_p_y(x=0.8, t=0.002, potential='double_slit')
    plot_p_y(x=0.8, t=0.002, potential='triple_slit')


if __name__ == '__main__':
    problem7()
    problem8()
    problem9()
