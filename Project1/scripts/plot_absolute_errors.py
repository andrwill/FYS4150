import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def v_exact(x):
    return 1.0 - (1.0 - np.exp(-10.0))*x - np.exp(-10.0*x)

if __name__ == "__main__":
    df = pd.read_csv("./data/approximations.csv").fillna(0.0)

    M = int(df.columns[-1][1:])

    x = np.linspace(0.0, 1.0, int(float(f"1e{len(df.columns)}")))

    v_exact = v_exact(x)

    for v in df.columns[:5]:
        v_formatted = f"$n_{{steps}}={v[1:]}$"

        m = int(v[1:])
        k = M // m
        print(df[v])

        plt.plot(x[::k], np.log10(np.abs(df[v][:m] - v_exact[::k])), 
                 label=f"$n_{{steps}}={m}$")

    plt.title("Absolute error")
    plt.xlabel("$x$")
    plt.ylabel("$\log_{10}|v_{exact}(x) - v_{approx}(x)|$")
    plt.legend()
    plt.savefig("./figures/absolute_errors.pdf")
