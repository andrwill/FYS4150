import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def v_exact(x):
    return 1.0 - (1.0 - np.exp(-10.0))*x - np.exp(-10.0*x)


if __name__ == "__main__":
    df = pd.read_csv("./data/approximations.csv")

    M = int(df.columns[-1][1:])
    x = np.linspace(0.0, 1.0, M)

    v_exact = v_exact(x)

    maximum_relative_errors = []
    for v in df.columns[:5]:
        v_formatted = f"$n_{{steps}}={v[1:]}$"

        m = int(v[1:])
        k = M // m

        relative_errors = np.abs(np.divide(df[v][:m] - v_exact[::k], v_exact[::k]))
        maximum_relative_errors.append(f"{np.max(relative_errors):.3e}")
        
        relative_errors = np.log(relative_errors)
        
        plt.plot(x[::k], relative_errors, label=f"$n_{{steps}} = {m}$")

    steps = [10**k for k in range(1,6)]

    # Generate table of maximum relative errors.
    pd.DataFrame({
        "Number of steps": steps, 
        "Maximum relative error": maximum_relative_errors
    }).to_csv("./data/maximum_relative_errors.csv", index=False)

    plt.title("Relative errors")
    plt.xlabel("$x$")
    plt.ylabel("$\log_{10}|(v_{exact}(x) - v_{approx}(x))/v_{exact}(x)|$")
    plt.legend()
    plt.savefig("./figures/relative_errors.pdf")
