import numpy as np
import pyarma as arma
import matplotlib.pyplot as plt

def plot_potential(potential):
    V = arma.mat()
    V.load(f'./data/potentials/{potential}.csv')

    plt.title(potential.replace('_', ' ').title())
    plt.axis('off')
    plt.imshow(V.t())
    plt.savefig(f'./figures/potentials/{potential}.pdf')
    plt.show()

if __name__ == '__main__':
    plot_potential('zero_potential')
    plot_potential('single_slit_potential')
    plot_potential('double_slit_potential')
    plot_potential('triple_slit_potential')
