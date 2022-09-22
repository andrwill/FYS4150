import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_three_smallest_eigvecs(N):
    eigvecs = pd.read_csv(f"./tables/three_last_eigvecs_{N}_by_{N}.csv").to_numpy()
   
    rows, cols = eigvecs.shape
    solutions = np.zeros((rows+2, cols))
    solutions[1:-1,:] = eigvecs

    i = range(rows+2)

    for j in range(cols):
        plt.plot(i, solutions[:,j], label=f"eigvec{j}", marker='o')

    plt.xlabel("Index")
    plt.ylabel("Component")
    plt.title(f"Three Smallest Eigenvectors of A (N={N})")
    plt.legend()
    plt.savefig(f"./figures/three_smallest_eigvecs_N_{N}_.pdf")

if __name__ == "__main__":
    plot_three_smallest_eigvecs(100)
    plt.show()
