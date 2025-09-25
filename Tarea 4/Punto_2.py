import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simpson
import os
# malla 2D periódica
L = 2.0
N = 192
dx = L / N
x = np.linspace(0, L-dx, N)
y = np.linspace(0, L-dx, N)
X, Y = np.meshgrid(x, y, indexing='xy')

# números de onda 2D
kx = 2*np.pi * np.fft.fftfreq(N, d=dx)
ky = 2*np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky, indexing='xy')
K2 = KX*KX + KY*KY

def etd1_step(U, V, dt, alpha, beta, Ffun, Gfun):
    E_u = np.exp(-alpha * K2 * dt)
    E_v = np.exp(-beta  * K2 * dt)
    Fu = Ffun(U, V)
    Gv = Gfun(U, V)

    with np.errstate(divide='ignore', invalid='ignore'):
        phi_u = (E_u - 1.0) / (-alpha * K2)
        phi_v = (E_v - 1.0) / (-beta  * K2)
    phi_u[K2==0] = dt
    phi_v[K2==0] = dt

    Uhat = np.fft.fft2(U); Vhat = np.fft.fft2(V)
    Fu_hat = np.fft.fft2(Fu)
    Gv_hat = np.fft.fft2(Gv)

    Uhat_new = E_u * Uhat + phi_u * Fu_hat
    Vhat_new = E_v * Vhat + phi_v * Gv_hat

    U_new = np.fft.ifft2(Uhat_new).real
    V_new = np.fft.ifft2(Vhat_new).real
    return U_new, V_new

def add_caption(ax, title, alpha, beta, Ftxt, Gtxt):
    ax.set_title(title, fontsize=11)
    ax.text(0.01, 0.02, f"α={alpha:.6f}, β={beta:.6f}\nF(u,v)={Ftxt}\nG(u,v)={Gtxt}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

def run_and_save(title, alpha, beta, Ffun, Gfun, Ftxt, Gtxt, T=15.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    U = 0.1 * rng.standard_normal((N,N))
    V = 0.1 * rng.standard_normal((N,N))
    steps = int(T/dt)
    for n in range(steps):
        U, V = etd1_step(U, V, dt, alpha, beta, Ffun, Gfun)
    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    im = ax.imshow(U, origin='lower', extent=(0,L,0,L),
                   cmap="cividis", interpolation="none")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_caption(ax, f"2_{title}", alpha, beta, Ftxt, Gtxt)
    fig.tight_layout()
    fig.savefig(f"2_{title}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# --------- Ejemplos ---------
def F1(u, v): return u - u**3 - v - 0.05
def G1(u, v): return 10.0*(u - v)

A, B = 1.0, 3.2
def F2(u, v): return A - (B+1)*u + u*u*v
def G2(u, v): return B*u - u*u*v

F, k = 0.0367, 0.0649
def F3(u, v): return -u*v*v + F*(1-u)
def G3(u, v): return  u*v*v - (F+k)*v

# --- Ejecución directa ---
run_and_save("turing_base",
             alpha=2.8e-4, beta=5.0e-2,
             Ffun=F1, Gfun=G1,
             Ftxt="u - u^3 - v - 0.05",
             Gtxt="10(u - v)",
             T=15.0, dt=0.01, seed=1)

run_and_save("turing_beta_mayor",
             alpha=2.8e-4, beta=0.12,
             Ffun=F1, Gfun=G1,
             Ftxt="u - u^3 - v - 0.05",
             Gtxt="10(u - v)",
             T=15.0, dt=0.01, seed=2)

run_and_save("brusselator",
             alpha=1.5e-4, beta=1.5e-3,
             Ffun=F2, Gfun=G2,
             Ftxt="A-(B+1)u+u^2 v",
             Gtxt="B u - u^2 v",
             T=15.0, dt=0.005, seed=3)

run_and_save("grayscott_manchas",
             alpha=1.6e-4, beta=8.0e-4,
             Ffun=F3, Gfun=G3,
             Ftxt="-u v^2 + F(1-u)",
             Gtxt="u v^2 - (F+k) v",
             T=15.0, dt=0.01, seed=4)

