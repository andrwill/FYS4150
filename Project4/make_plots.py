import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def problem5():
    ordered_T_1point0 = pd.read_csv("./data/prob5_ordered_T10.csv").to_numpy()
    ordered_T_2point4 = pd.read_csv("./data/prob5_ordered_T24.csv").to_numpy()
    disordered_T_1point0 = pd.read_csv("./data/prob5_disordered_T10.csv").to_numpy()
    disordered_T_2point4 = pd.read_csv("./data/prob5_disordered_T24.csv").to_numpy()
    
    cycles = range(ordered_T_1point0.shape[0])
    
    def plot_average_energy():
        plt.title('Estimated Average Energy')
        plt.xlabel('Monte Carlo Cycle')
        plt.ylabel('$\epsilon$')
        plt.plot(cycles, ordered_T_1point0[:,0], label='Initially ordered, $T=1.0$')
        plt.plot(cycles, disordered_T_1point0[:,0], label='Initially disordered, $T=1.0$')
        plt.plot(cycles, ordered_T_2point4[:,0], label='Initially ordered, $T=2.4$')
        plt.plot(cycles, disordered_T_2point4[:,0], label='Initially disordered, $T=2.4$')
        plt.legend()
        plt.savefig('./figures/estimated_average_energy_20x20.pdf')
        plt.show()

    def plot_average_abs_magnetization():
        plt.title('Estimated Average Absolute Magnetization')
        plt.xlabel('Monte Carlo Cycle')
        plt.ylabel('$m$')
        plt.plot(cycles, ordered_T_1point0[:,1], label='Initially ordered, $T=1.0$')
        plt.plot(cycles, disordered_T_1point0[:,1], label='Initially disordered, $T=1.0$')
        plt.plot(cycles, ordered_T_2point4[:,1], label='Initially ordered, $T=2.4$')
        plt.plot(cycles, disordered_T_2point4[:,1], label='Initially disordered, $T=2.4$')
        plt.legend()
        plt.savefig('./figures/estimated_average_abs_magnetization_20x20.pdf')
        plt.show()

    plot_average_energy()
    plot_average_abs_magnetization()

def problem6():
    
    def plot_dist_with_T_1point0():
        disordered_T_1point0 = pd.read_csv("./data/prob5_disordered_T10.csv").to_numpy()

        epsilons = disordered_T_1point0[:,0]
        num_cycles = disordered_T_1point0.shape[0]
        plt.title('Estimated pdf of $\epsilon$ with $T=1.0$')
        plt.xlabel('$\epsilon$')
        plt.ylabel('$p(\epsilon;T)$')
        plt.hist(epsilons/num_cycles, bins=30)
        plt.savefig('./figures/eps_pdf_T_1point0.pdf')
        plt.show()
    def plot_dist_with_T_2point4():
        disordered_T_2point4 = pd.read_csv("./data/prob5_disordered_T24.csv").to_numpy()

        epsilons = disordered_T_2point4[:,0]
        num_cycles = disordered_T_2point4.shape[0]
        plt.title('Estimated pdf of $\epsilon$ with $T=2.4$')
        plt.xlabel('$\epsilon$')
        plt.ylabel('$p(\epsilon;T)$')
        plt.hist(epsilons/num_cycles, bins=30)
        plt.savefig('./figures/eps_pdf_T_2point4.pdf')
        plt.show()

    plot_dist_with_T_1point0()
    plot_dist_with_T_2point4()

def problem8():
    Ts = [2.1, 2.2, 2.25, 2.3, 2.4]
    Ls = [40, 60, 80, 100]

    E = [[None for j in range(len(Ts))] for i in range(len(Ls))]
    abs_M = [[None for j in range(len(Ts))] for i in range(len(Ls))]
    for L in Ls:
        for T in Ts:
            df = pd.read_csv(f'./data/prob8_T{T:.2f}_L{L}.csv').to_numpy()
            E[i][j] = df[:,0]
            abs_M[i][j] = df[:,1]

    #for i in range(len(Ls)):
    #    plt.title('')
    #    for j in range(len(Ts)):

    for i in range(len(Ls)):
        plt.title(f'Suceptibility ($L={Ls[i]}$')
        plt.xlabel('$T$')
        plt.ylabel('$\chi$')
        for j in range(len(Ts)):
            pass

def problem9():
    pass

if __name__ == '__main__':
    #problem5()
    problem6()
    #problem8()
