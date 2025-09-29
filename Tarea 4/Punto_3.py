import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simpson
import os

# --- Parámetros globales ---
L = 40.0        # longitud del dominio
N = 256         # número de puntos espaciales
x = np.linspace(0, L, N, endpoint=False)
delta = 0.22    # parámetro de dispersión
steps = 20000   # pasos para animación principal
steps_cfl = 2000  # pasos más cortos para cálculo CFL

# Espacio de Fourier
dx = L / N
k = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Condición inicial: solitón (sech^2)
phi0 = 2 * (1/np.cosh(x - L/3))**2

# --- Definición de esquema RK4 en espacio de Fourier ---
def rhs(phi_hat, k, delta):
    phi = np.fft.ifft(phi_hat).real
    phi_x = np.fft.ifft(1j*k*phi_hat).real
    nonlinear = -np.fft.fft(phi * phi_x)
    dispersive = -(1j*delta**2)*(k**3)*phi_hat
    return nonlinear + dispersive

def rk4(phi_hat, dt, k, delta):
    k1 = rhs(phi_hat, k, delta)
    k2 = rhs(phi_hat + 0.5*dt*k1, k, delta)
    k3 = rhs(phi_hat + 0.5*dt*k2, k, delta)
    k4 = rhs(phi_hat + dt*k3, k, delta)
    return phi_hat + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# --- Simulación principal para la animación ---
dt = 1e-3
phi_hat = np.fft.fft(phi0)

frames = []
for n in range(steps):
    phi_hat = rk4(phi_hat, dt, k, delta)
    if n % 50 == 0:   # guardar cada 50 pasos
        frames.append(np.fft.ifft(phi_hat).real)

# --- Animación ---
fig, ax = plt.subplots()
line, = ax.plot(x, frames[0])
ax.set_ylim(-2, 2)   # rango fijo para claridad
ax.set_title("Evolución de la ecuación KdV")

def update(i):
    line.set_ydata(frames[i])
    ax.set_xlabel(f"t = {i*dt*50:.2f}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30)
ani.save("3_kdv_soliton.mp4", writer="ffmpeg", dpi=150)
plt.close()

print("Simulación guardada en '3_kdv_soliton.mp4'")

# --- Grupos de parámetros (solo dt y dx) ---
parametros = [
    {"dt": 0.001, "dx": 0.1},
    {"dt": 0.002, "dx": 0.05},
    {"dt": 0.005, "dx": 0.2},
    {"dt": 0.001, "dx": 0.05},
]

# --- Cálculo de CFL ---
with open("3_cfl_resultados.txt", "w") as f:
    f.write("Resultados del cálculo de CFL para la ecuación de KdV\n")
    f.write("----------------------------------------------------\n\n")

    for i, p in enumerate(parametros, start=1):
        dt_param = p["dt"]
        dx_param = p["dx"]

        # Inicialización en Fourier
        phi_hat = np.fft.fft(phi0)

        # Simulación corta para obtener u_max
        u_max = 0.0
        for n in range(steps_cfl):
            phi_hat = rk4(phi_hat, dt_param, k, delta)
            if n % 50 == 0:
                phi = np.fft.ifft(phi_hat).real
                u_max = max(u_max, np.max(np.abs(phi)))

        if u_max == 0:
            f.write(f"Conjunto {i}: CFL = no existe (u_max = 0)\n")
        else:
            c = 6 * u_max
            cfl = c * dt_param / dx_param
            f.write(f"Conjunto {i}: dt={dt_param}, dx={dx_param}, u_max={u_max:.4f} → CFL = {cfl:.4f}\n")

print("Archivo '3_cfl_resultados.txt' generado con éxito.")


# --- Parámetros globales ---
L = 40.0        # longitud del dominio
N = 256         # número de puntos espaciales
x = np.linspace(0, L, N, endpoint=False)
delta = 0.22    # parámetro de dispersión
steps = 20000    # pasos para animación
dt = 1e-3

# Espacio de Fourier
dx = L / N
k = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Condición inicial (A=2 al inicio)
A0 = 2.0
phi0 = A0 * (1/np.cosh(x - L/3))**2

# --- Definición de esquema RK4 en espacio de Fourier ---
def rhs(phi_hat, k, delta):
    phi = np.fft.ifft(phi_hat).real
    phi_x = np.fft.ifft(1j*k*phi_hat).real
    nonlinear = -np.fft.fft(phi * phi_x)
    dispersive = -(1j*delta**2)*(k**3)*phi_hat
    return nonlinear + dispersive

def rk4(phi_hat, dt, k, delta):
    k1 = rhs(phi_hat, k, delta)
    k2 = rhs(phi_hat + 0.5*dt*k1, k, delta)
    k3 = rhs(phi_hat + 0.5*dt*k2, k, delta)
    k4 = rhs(phi_hat + dt*k3, k, delta)
    return phi_hat + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# --- Simulación ---
phi_hat = np.fft.fft(phi0)
frames = []

for n in range(steps):
    phi_hat = rk4(phi_hat, dt, k, delta)
    if n % 20 == 0:
        # Definir A(t) que crece lentamente
        A_t = 1.0 + 0.0002*n
        phi = np.fft.ifft(phi_hat).real
        frames.append((A_t * phi, A_t))

# --- Animación ---
fig, ax = plt.subplots()
line, = ax.plot(x, frames[0][0])
ax.set_ylim(-10, 10)
ax.set_title("Evolución de la ecuación KdV con amplitud creciente")

def update(i):
    y, A_t = frames[i]
    line.set_ydata(y)
    ax.set_xlabel(f"t = {i*dt*20:.2f}, A = {A_t:.2f}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30)
ani.save("3_kdv_soliton_A_creciente.mp4", writer="ffmpeg", dpi=150)
plt.close()

print("Animación con A creciente guardada en '3_kdv_soliton_A_creciente.mp4'")





























