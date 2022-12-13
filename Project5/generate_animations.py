import numpy as np
import pyarma as arma
import matplotlib
import matplotlib.pyplot as plt

from math import sqrt
from matplotlib.animation import FuncAnimation


def load_cx_mat(filename):
    U = arma.cx_mat()
    U.load(filename, arma.csv_ascii)
    U = np.array(U)
    return U

def animate(potential):
    U = load_cx_mat(f'./data/{potential}_simulation.csv')

    n = round(sqrt(U.shape[0]))
    t = np.arange(0.0, 1.0, 2.5e-5)
    x = np.linspace(0.0, 1.0, n+2)
    y = np.linspace(0.0, 1.0, n+2)

    fontsize = 12

    fig = plt.figure()
    ax = plt.gca()

    u = U[:,0]
    p = (np.conj(u)*u).reshape(n, n).real

    norm = matplotlib.cm.colors.Normalize(vmin=0.0, vmax=np.max(p))
    img = ax.imshow(
        p,
        extent = [x[0], x[-1], y[0], y[-1]],
        cmap = plt.get_cmap('viridis'),
        norm=norm
    )

    plt.xlabel('$x$', fontsize=fontsize)
    plt.ylabel('$y$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    colorbar = fig.colorbar(img, ax=ax)
    colorbar.set_label('$u(x, y, t)$', fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=fontsize)

    time_txt = plt.text(
        0.95, 0.95, 
        f't = {t[0]:.5f}', color='white', 
        horizontalalignment='right', verticalalignment='top',
        fontsize=fontsize
    )

    def animation(i):
        u = U[:, i]
        p = (np.conj(u)*u).reshape(n, n).real

        color_scale = matplotlib.cm.colors.Normalize(vmin=0.0, vmax=np.max(p))
        img.set_norm(norm)
        img.set_data(p)

        time_txt.set_text(f't = {t[i]:.5f}')

        return img

    anim = FuncAnimation(
        fig, 
        animation, 
        interval=1, 
        frames=U.shape[1],
        repeat=False,
        blit=0
    )

    anim.save(
        f'./animations/{potential}_animation.mp4', 
        writer='ffmpeg',
        fps=60
    )

    plt.show()

if __name__ == '__main__':
    #animate('zero_potential')
    animate('problem8')
    #animate('single_slit')
    #animate('double_slit')
    #animate('triple_slit')
