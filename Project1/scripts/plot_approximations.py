import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def v_exact(x):
    return 1.0 - (1.0 - np.exp(-10.0))*x - np.exp(-10.0*x)



if __name__ == "__main__":
    df = pd.read_csv("./data/approximations.csv").fillna(0.0)

    M = int(df.columns[-1][1:])#int(float(f"1e{len(df.columns)}"))
    x = np.linspace(0.0, 1.0, M)

    for v in df.columns[:5]:
        v_formatted = f"$n_{{steps}}={v[1:]}$"

        m = int(v[1:])
        k = M // m

        plt.plot(x[::k], df[v][:m], label=v_formatted)

    plt.plot(x, v_exact(x), label="Exact solution")
    plt.title("Approximations")
    plt.xlabel("$x$")
    plt.ylabel("$v(x)$")
    plt.legend()
    plt.savefig("./figures/approximations.pdf")
