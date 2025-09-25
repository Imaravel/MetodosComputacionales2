import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simpson
import os

# --- Parámetros Generales de la Simulación ---
L = 40.0
Nx = 1024
x = np.linspace(-L / 2, L / 2, Nx)
dx = x[1] - x[0]
ALPHA = 0.1

# --- Condición Inicial ---
def get_initial_psi():
    posicion_inicial = np.exp(-2.0 * (x + 10.0)**2)
    velocidad_inicial = np.exp(1j * 2.0 * x)
    psi0 = posicion_inicial * velocidad_inicial
    norm = np.sqrt(simpson(np.abs(psi0)**2, x))
    return psi0 / norm

# --- Potenciales ---
def V_harmonic(x):
    return -x**2 / 50.0

def V_quartic(x):
    return (x / 5.0)**4

def V_hat(x):
    return (1.0 / 50.0) * ((x**4 / 100.0) - x**2)

# --- Split-Step Fourier Method ---
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

def solve_schrodinger(psi0, V_func, t_final, dt):
    psi = psi0.copy()
    V = V_func(x)
    V_op = np.exp(-1j * V * dt / 2.0)
    T_op_dt = np.exp(-1j * ALPHA * k**2 * dt)

    psi_t = [psi0]
    mu_t, sigma_t = [], []

    num_steps = int(t_final / dt)

    for i in range(num_steps):
        psi = V_op * psi
        psi_k = np.fft.fft(psi)
        psi_k = T_op_dt * psi_k
        psi = np.fft.ifft(psi_k)
        psi = V_op * psi
        psi_t.append(psi.copy())

        prob_density = np.abs(psi)**2
        mu = simpson(x * prob_density, x)
        mu_t.append(mu)
        x2_mean = simpson(x**2 * prob_density, x)
        sigma = np.sqrt(x2_mean - mu**2)
        sigma_t.append(sigma)

    return np.array(psi_t), np.linspace(0, t_final, num_steps), np.array(mu_t), np.array(sigma_t)

# --- Funciones para Graficar y Animar ---
def _resolve_animation_target(filename):
    root, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == '.mp4':
        if animation.writers.is_available('ffmpeg'):
            return filename, 'ffmpeg'
        fallback = root + '.gif'
        print('FFmpeg no está disponible; se guardará la animación como GIF en', fallback)
        return fallback, 'pillow'

    if ext in ('.gif', '.apng'):
        return filename, 'pillow'

    fallback = root + '.gif'
    print('Extensión desconocida para animación, se usará GIF en', fallback)
    return fallback, 'pillow'


def create_animation(psi_t, V_func, filename, t_final):
    fig, ax = plt.subplots()
    prob_density_t = np.abs(psi_t)**2

    line, = ax.plot(x, prob_density_t[0], lw=2, color='black')
    pot, = ax.plot(x, V_func(x) / (2 * np.max(np.abs(V_func(x)))) +
                   0.5 * np.max(prob_density_t), 'r--', label='V(x) (escalado)')

    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(0, 1.1 * np.max(prob_density_t))
    ax.set_xlabel('Posición (x)')
    ax.set_ylabel(r'Densidad de Probabilidad $|\psi|^2$')
    ax.set_title(f'Evolución del Paquete de Ondas en {filename.split(".")[1]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    def animate(i):
        line.set_ydata(prob_density_t[i])
        return line,

    frame_step = max(1, len(psi_t) // 300)
    frames = range(0, len(psi_t), frame_step)
    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)
    filename, writer = _resolve_animation_target(filename)
    anim.save(filename, writer=writer, fps=30, dpi=150)
    plt.close(fig)

def create_uncertainty_plot(t_array, mu_t, sigma_t, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_array, mu_t, 'b-', lw=2, label=r'Posición media $\mu(t)$')
    ax.fill_between(t_array, mu_t - sigma_t, mu_t + sigma_t,
                    color='blue', alpha=0.2, label=r'Incertidumbre $\mu \pm \sigma$')
    ax.set_xlabel('Tiempo (t)')
    ax.set_ylabel('Posición (x)')
    ax.set_title(f'Posición e Incertidumbre vs. Tiempo en {filename.split(".")[1]}')
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close(fig)

# --- Ejecución ---
# 1.a Oscilador Armónico
T_FINAL_A, DT_A = 150.0, 0.05
psi0 = get_initial_psi()
psi_t_a, t_a, mu_a, sigma_a = solve_schrodinger(psi0, V_harmonic, T_FINAL_A, DT_A)
create_animation(psi_t_a, V_harmonic, '1.a.mp4', T_FINAL_A)
create_uncertainty_plot(t_a, mu_a, sigma_a, '1.a.pdf')

# 1.b Oscilador Cuártico
T_FINAL_B, DT_B = 50.0, 0.02
psi0 = get_initial_psi()
psi_t_b, t_b, mu_b, sigma_b = solve_schrodinger(psi0, V_quartic, T_FINAL_B, DT_B)
create_animation(psi_t_b, V_quartic, '1.b.mp4', T_FINAL_B)
create_uncertainty_plot(t_b, mu_b, sigma_b, '1.b.pdf')

# 1.c Potencial Sombrero
T_FINAL_C, DT_C = 150.0, 0.05
psi0 = get_initial_psi()
psi_t_c, t_c, mu_c, sigma_c = solve_schrodinger(psi0, V_hat, T_FINAL_C, DT_C)
create_animation(psi_t_c, V_hat, '1.c.mp4', T_FINAL_C)
create_uncertainty_plot(t_c, mu_c, sigma_c, '1.c.pdf')
