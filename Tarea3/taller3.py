import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.special import betainc
import pandas as pd


def integrate(fun, t_span, y0, rtol=1e-9, atol=1e-12, max_step=np.inf, events=None):
    return solve_ivp(fun, t_span, y0, method="RK45",
                     rtol=rtol, atol=atol, max_step=max_step,
                     events=events, dense_output=False)
#1
# 1.a Lotka-Volterra
def lotka_volterra(t, z, a=2, b=1.5, g=0.3, d=0.4):
    x, y = z
    return np.array([a*x - b*x*y, -g*y + d*x*y])

def lotka_V(x, y, a=2, b=1.5, g=0.3, d=0.4):
    return d*x - g*np.log(np.maximum(x,1e-15)) + b*y - a*np.log(np.maximum(y,1e-15))

sol = integrate(lambda t,z: lotka_volterra(t, z, 2.0,1.5,0.3,0.4),
                (0.0, 50.0), np.array([3, 2], float),
                rtol=1e-10, atol=1e-12, max_step=0.02)
t = sol.t; x = sol.y[0]; y = sol.y[1]
V = lotka_V(x, y, 2.0,1.5,0.3,0.4)
fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
axs[0].plot(t,x); axs[0].set(xlabel='t', ylabel='x(t)', title='Lotka-Volterra: x(t)')
axs[1].plot(t,y); axs[1].set(xlabel='t', ylabel='y(t)', title='Lotka-Volterra: y(t)')
axs[2].plot(t,V); axs[2].set(xlabel='t', ylabel='V(t)', title='Cantidad conservada V')
fig.savefig('1.a.pdf', bbox_inches='tight'); plt.close(fig)

# 1.b Landau
def landau_ode(t, z, q, B0, E0, m, k):
    x, y, vx, vy = z
    ex_term = np.sin(k*x) + k*x*np.cos(k*x)
    ax = (q*E0*ex_term - q*B0*vy)/m
    ay = (q*B0*vx)/m
    return np.array([vx, vy, ax, ay])

def landau_conserved(z, q, B0, E0, m, k):
    x, y, vx, vy = z
    Pi_y = m*vy - q*B0*x
    K = 0.5*m*(vx*vx + vy*vy)
    U = - q*E0*x*np.sin(k*x)
    return Pi_y, K+U

q,B0,E0,m,k = 7.5284, 0.438, 0.7423, 3.8428, 1.0014
sol = integrate(lambda t,z: landau_ode(t, z, q,B0,E0,m,k),
                (0.0, 30.0), np.array([0.0,0.0,0.0,0.1], float),
                rtol=1e-9, atol=1e-12, max_step=0.01)
t = sol.t; x, y, vx, vy = sol.y
Pi_y = np.empty_like(t); E_tot = np.empty_like(t)
for i in range(len(t)): Pi_y[i], E_tot[i] = landau_conserved([x[i],y[i],vx[i],vy[i]], q,B0,E0,m,k)
fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
axs[0].plot(t,x,label='x'); axs[0].plot(t,y,label='y'); axs[0].legend(); axs[0].set(title='Landau: posiciones')
axs[1].plot(t,Pi_y); axs[1].set(ylabel='Π_y')
axs[2].plot(t,E_tot); axs[2].set(xlabel='t', ylabel='K+U')
fig.savefig('1.b.pdf', bbox_inches='tight'); plt.close(fig)

# 1.c Binario gravitacional
def binary_ode(t, z, G=1.0, m=1.7):
    x1,y1,vx1,vy1,x2,y2,vx2,vy2 = z
    rx, ry = x2-x1, y2-y1
    r2 = rx*rx + ry*ry; inv_r3 = 1.0/(r2*np.sqrt(r2)+1e-15)
    ax1, ay1 =  G*m*rx*inv_r3,  G*m*ry*inv_r3
    ax2, ay2 = -G*m*rx*inv_r3, -G*m*ry*inv_r3
    return np.array([vx1,vy1,ax1,ay1, vx2,vy2,ax2,ay2])

def binary_conserved(z, G=1.0, m=1.7):
    x1,y1,vx1,vy1,x2,y2,vx2,vy2 = z
    rx, ry = x2-x1, y2-y1
    r = np.sqrt(rx*rx + ry*ry)
    K = 0.5*m*(vx1*vx1+vy1*vy1+vx2*vx2+vy2*vy2)
    U = -G*m*m/(r+1e-15)
    Lz = m*(x1*vy1 - y1*vx1 + x2*vy2 - y2*vx2)
    return K+U, Lz

z0 = np.array([0.0,0.0,0.0,0.5, 1.0,1.0,0.0,-0.5], float)
sol = integrate(lambda t,z: binary_ode(t, z, 1.0,1.7),
                (0.0, 10.0), z0, rtol=1e-10, atol=1e-12, max_step=0.01)
t = sol.t; Z = sol.y
E = np.empty_like(t); L = np.empty_like(t)
for i in range(len(t)): E[i], L[i] = binary_conserved(Z[:,i], 1.0,1.7)
fig, axs = plt.subplots(3,1, figsize=(8,9), constrained_layout=True)
axs[0].plot(t,Z[0],label='x1'); axs[0].plot(t,Z[1],label='y1')
axs[0].plot(t,Z[4],label='x2'); axs[0].plot(t,Z[5],label='y2'); axs[0].legend(ncol=2,fontsize=8)
axs[1].plot(t,E); axs[2].plot(t,L); axs[2].set(xlabel='t')
fig.savefig('1.c.pdf', bbox_inches='tight'); plt.close(fig)
#2
g = 9.773; m_bala = 10.01; A_beta = 1.642; B_beta = 40.624; C_beta = 2.36


def beta_y_numba(y):
    val = 1.0 - y/B_beta
    return 0.0 if val <= 0.0 else A_beta*(val**C_beta)

def beta_y(y):  
    return beta_y_numba(y)

def proj_ode(t, z):
    x,y,vx,vy = z
    v = np.sqrt(vx*vx + vy*vy)
    b = beta_y(y)
    ax = -(b/m_bala)*v*vx
    ay = -g - (b/m_bala)*v*vy
    return np.array([vx, vy, ax, ay])

def simulate_shot(v0, theta_deg, tmax=200.0):
    th = np.deg2rad(theta_deg)
    y0 = np.array([0.0, 0.0, v0*np.cos(th), v0*np.sin(th)], float)
    def hit_ground(t,z): return z[1]
    hit_ground.terminal = True; hit_ground.direction = -1.0
    sol = integrate(proj_ode, (0.0,tmax), y0, rtol=1e-8, atol=1e-10, max_step=0.05, events=hit_ground)
    if sol.t_events[0].size>0:
        return sol.y_events[0][0][0], sol.t_events[0][0], sol
    return sol.y[0,-1], sol.t[-1], sol

def y_at_x_for_angle(v0, theta_deg, x_target, tmax=200.0):
    th = np.deg2rad(theta_deg)
    y0 = np.array([0.0, 0.0, v0*np.cos(th), v0*np.sin(th)], float)
    def pass_xt(t,z): return z[0]-x_target
    pass_xt.terminal=True; pass_xt.direction=0.0
    sol = integrate(proj_ode, (0.0,tmax), y0, rtol=1e-8, atol=1e-10, max_step=0.05, events=pass_xt)
    return None if sol.t_events[0].size==0 else sol.y_events[0][0][1]

def angle_to_hit(v0, x_target, y_target, th_min=10.0, th_max=80.0, tol_y=1e-2, max_iter=40):
    a,b = th_min, th_max
    fa = y_at_x_for_angle(v0,a,x_target); fb = y_at_x_for_angle(v0,b,x_target)
    if (fa is None) or (fb is None) or ((fa-y_target)*(fb-y_target)>0.0):
        thetas = np.linspace(th_min, th_max, 51)
        vals = [y_at_x_for_angle(v0, th, x_target) for th in thetas]
        good = [(th,yy) for th,yy in zip(thetas, vals) if yy is not None]
        if len(good)<2: raise ValueError("No se alcanza el objetivo con los ángulos de búsqueda")
        for i in range(1,len(good)):
            th0,y0 = good[i-1]; th1,y1 = good[i]
            if (y0-y_target)*(y1-y_target) <= 0.0:
                a,fa = th0,y0; b,fb = th1,y1; break
    for _ in range(max_iter):
        c = 0.5*(a+b); fc = y_at_x_for_angle(v0,c,x_target)
        if fc is None: a=c; continue
        if abs(fc-y_target) < tol_y: return c
        if (fa-y_target)*(fc-y_target) <= 0.0: b,fb = c,fc
        else: a,fa = c,fc
    return 0.5*(a+b)

# 2.a 
v0s = np.linspace(20.0, 140.0, 25)
thetas_scan = np.linspace(10.0, 80.0, 71)
x_max = []
for v0 in v0s:
    best = -1.0
    for th in thetas_scan:
        x_end, _, _ = simulate_shot(v0, th)
        best = max(best, x_end)
    x_max.append(best)
x_max = np.array(x_max)
fig, ax = plt.subplots(figsize=(7,4.8))
ax.plot(v0s, x_max, marker='o'); ax.set(xlabel='v0 [m/s]', ylabel='x_max [m]', title='2.a Alcance máximo vs v0')
fig.savefig('2.a.pdf', bbox_inches='tight'); plt.close(fig)

# 2.b 
v0_b      = 60.0
x_t_b     = 12.0
y_t_b     = 0.0
theta_b   = angle_to_hit(v0_b, x_t_b, y_t_b, th_min=10.0, th_max=80.0, tol_y=1e-2, max_iter=50)
y_verif   = y_at_x_for_angle(v0_b, theta_b, x_t_b)
xe, te, solb = simulate_shot(v0_b, theta_b)
tx, xx, yy = solb.t, solb.y[0], solb.y[1]
with open('2.b.txt','w',encoding='utf-8') as f:
    f.write('v0[m/s]\ttheta[deg]\tx_target[m]\ty_target[m]\ty(x_target)[m]\n')
    f.write(f'{v0_b:.6f}\t{theta_b:.6f}\t{x_t_b:.6f}\t{y_t_b:.6f}\t{(np.nan if y_verif is None else y_verif):.6f}\n')
fig, ax = plt.subplots(figsize=(7,4.6))
ax.plot(xx,yy,lw=1.5,label=f'Trayectoria (v0={v0_b:.1f}, θ={theta_b:.2f}°)')
ax.scatter([x_t_b],[y_t_b],s=50,marker='x',label='Objetivo',zorder=5)
if y_verif is not None:
    ax.scatter([x_t_b],[y_verif],s=36,facecolors='none',edgecolors='k',label='y(x_target)')
ax.set(xlabel='x [m]', ylabel='y [m]', title='2.b Ángulo para impactar el objetivo')
ax.grid(True, ls='--', alpha=0.4); ax.legend()
fig.savefig('2.b.pdf', bbox_inches='tight'); plt.close(fig)

#  2.c 
def hits_target(v0, th, x_target=12.0, y_target=0.0, tol_y=0.05):
    yy = y_at_x_for_angle(v0, th, x_target)
    return (yy is not None) and (abs(yy-y_target) <= tol_y)

v0s_grid  = np.linspace(20.0, 140.0, 31)
thetas_g  = np.linspace(10.0, 80.0, 71)
V, TH = [], []
for v in v0s_grid:
    for th in thetas_g:
        if hits_target(v, th, 12.0, 0.0, 0.05):
            V.append(v); TH.append(th)
V, TH = np.array(V), np.array(TH)
fig, ax = plt.subplots(figsize=(6.4,4.8))
ax.scatter(V, TH, s=8)
ax.set(xlabel='v0 [m/s]', ylabel='theta0 [deg]', title='2.c Soluciones que atinan a (12 m, 0)')
fig.savefig('2.c.pdf', bbox_inches='tight'); plt.close(fig)
hbar = 0.1
a_M  = 0.8
x0_M = 10.0


def V_morse_numba(x):                                         
    return (1.0 - np.exp(-a_M*(x - x0_M)))**2 - 1.0

def V_morse(x):  
    return (1.0 - np.exp(-a_M*(x - x0_M)))**2 - 1.0


def rk4_step(x, psi, phi, h, eps):
    k1_psi = phi
    k1_phi = (V_morse_numba(x) - eps)/(hbar*hbar) * psi

    x2 = x + 0.5*h
    psi2 = psi + 0.5*h*k1_psi
    phi2 = phi + 0.5*h*k1_phi
    k2_psi = phi2
    k2_phi = (V_morse_numba(x2) - eps)/(hbar*hbar) * psi2

    psi3 = psi + 0.5*h*k2_psi
    phi3 = phi + 0.5*h*k2_phi
    k3_psi = phi3
    k3_phi = (V_morse_numba(x2) - eps)/(hbar*hbar) * psi3

    x4 = x + h
    psi4 = psi + h*k3_psi
    phi4 = phi + h*k3_phi
    k4_psi = phi4
    k4_phi = (V_morse_numba(x4) - eps)/(hbar*hbar) * psi4

    psi_next = psi + (h/6.0)*(k1_psi + 2.0*k2_psi + 2.0*k3_psi + k4_psi)
    phi_next = phi + (h/6.0)*(k1_phi + 2.0*k2_phi + 2.0*k3_phi + k4_phi)
    return psi_next, phi_next

def integrate_rk4_schr(eps, xs, psi0, phi0):
    n = xs.shape[0]
    psi = np.empty(n, np.float64)
    phi = np.empty(n, np.float64)
    psi[0] = psi0; phi[0] = phi0
    for i in range(n-1):
        h = xs[i+1]-xs[i]
        psi[i+1], phi[i+1] = rk4_step(xs[i], psi[i], phi[i], h, eps)
    return psi, phi

def find_turning_points(eps, x_min=2.0, x_max=20.0, nx=3001):
    xs = np.linspace(x_min, x_max, nx)
    s = V_morse(xs) - eps
    idx = []
    for i in range(1, nx):
        if s[i-1]*s[i] <= 0.0:
            idx.append(i-1)
    if len(idx) >= 2:
        i1, i2 = idx[0], idx[-1]
        x1 = xs[i1] - s[i1]*(xs[i1+1]-xs[i1])/(s[i1+1]-s[i1]+1e-30)
        x2 = xs[i2] - s[i2]*(xs[i2+1]-xs[i2])/(s[i2+1]-s[i2]+1e-30)
        return x1, x2
    return 6.0, 16.0

def normalize_psi(xs, psi):
    s = 0.0
    for i in range(1,len(xs)):
        dx = xs[i]-xs[i-1]; s += 0.5*dx*(psi[i]**2 + psi[i-1]**2)
    return psi/np.sqrt(s) if s>0 else psi

def theoretical_morse_energies(nmax=50):
    lam = 1.0/(a_M*hbar)
    vals = []
    for n in range(nmax):
        if (n+0.5) >= lam: break
        vals.append(-(1.0 - (n+0.5)/lam)**2)
    return np.array(vals)

#3.a
eps_grid = np.linspace(-0.99, -0.01, 500)
norms = np.empty_like(eps_grid)
for i, eps in enumerate(eps_grid):
    x1, x2 = find_turning_points(eps)
    xs = np.linspace(x1-2.0, x2+2.0, 4001)
    psi, phi = integrate_rk4_schr(eps, xs, 0.0, 1e-10)
    norms[i] = np.hypot(psi[-1], phi[-1])

mins = [i for i in range(1,len(norms)-1) if norms[i]<=norms[i-1] and norms[i]<=norms[i+1]]
eps_found = eps_grid[np.array(mins, int)]


curves = []
for eps in eps_found:
    x1, x2 = find_turning_points(eps)
    xs = np.linspace(x1-2.0, x2+2.0, 4001)
    psi, phi = integrate_rk4_schr(eps, xs, 0.0, 1e-10)
    curves.append((eps, xs, normalize_psi(xs, psi)))


fig, ax = plt.subplots(figsize=(6,6))
Xp = np.linspace(2.0, 20.0, 1000)
ax.plot(Xp, V_morse(Xp), color='k', lw=1.2, label='Morse potential')

colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(curves))) if len(curves)>0 else []
for (c,(eps, xs, psi_n)) in zip(colors, curves):
    ax.plot(xs, psi_n + eps, lw=1.0, color=c)
    ax.hlines(eps, xs.min(), xs.max(), colors=c, linestyles='dotted', lw=0.6)
ax.set_xlim(0.0, 12.0)
ax.set_ylim(-1.05, 0.05)
ax.set_xlabel('x'); ax.set_ylabel('ψ(x) + ε')
ax.set_title('3. Estados ligados en potencial de Morse (shooting)')
ax.legend(loc='lower left', frameon=False)
fig.savefig('3.pdf', bbox_inches='tight'); plt.close(fig)

#3.b
eps_theo = theoretical_morse_energies(60)
n_min = min(len(eps_found), len(eps_theo))
with open('3.txt','w',encoding='utf-8') as f:
    f.write('n\tE_found\tE_theoretical\tpercent_diff\n')
    for n in range(n_min):
        Ef, Et = eps_found[n], eps_theo[n]
        f.write(f'{n}\t{Ef:.8f}\t{Et:.8f}\t{100.0*abs(Ef-Et)/(abs(Et)+1e-30):.4f}\n')

# Problema 4: Caos del Péndulo Elástico Planar
def problema_4():
    """
    Simula el péndulo elástico planar y crea secciones de Poincaré
    """
    def pendulum_equations(t, y, alpha):
        theta, r, P_theta, P_r = y
        dtheta = P_theta / (r + 1)**2
        dr = P_r
        dP_theta = -alpha**2 * (r + 1) * np.sin(theta)
        dP_r = alpha**2 * np.cos(theta) - r + P_theta**2 / (1 + r)**3
        return [dtheta, dr, dP_theta, dP_r]
    
    def event_crossing(t, y):
        """Detecta cuando theta = 0 (mod 2π)"""
        return np.sin(y[0])
    
    event_crossing.direction = -1  # Solo cruces descendentes
    
    # Parámetros
    alphas = np.linspace(1, 1.2, 20)
    t_max = 10000
    y0 = [np.pi/2, 0, 0, 0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    
    for i, alpha in enumerate(alphas):
        # Resolver con detección de eventos
        sol = solve_ivp(
            lambda t, y: pendulum_equations(t, y, alpha),
            [0, t_max], y0,
            events=event_crossing,
            dense_output=True,
            max_step=0.1,
            rtol=1e-10
        )
        
        if sol.t_events[0].size > 10:  # Solo si hay suficientes cruces
            # Obtener estados en los cruces
            poincare_states = []
            for t_event in sol.t_events[0][1:]:  # Ignorar el primer cruce
                state = sol.sol(t_event)
                poincare_states.append([state[1], state[3]])  # r, P_r
            
            poincare_states = np.array(poincare_states)
            ax.scatter(poincare_states[:, 0], poincare_states[:, 1], 
                      c=[colors[i]]*len(poincare_states), s=0.5, alpha=0.6)
    
    ax.set_xlabel('r')
    ax.set_ylabel('$P_r$')
    ax.set_title('Sección de Poincaré del Péndulo Elástico Planar')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('4.pdf')
    plt.close()


# Problema 6: Teoría de vigas quasiestáticas
def problema_6():
    """
    Resuelve las ecuaciones de Timoshenko-Ehrenfest para deformación de viga
    """
    def beam_equations(x, y):
        # y = [phi, dphi/dx, w, dw/dx]
        phi, dphi, w, dw = y
        
        # Parámetros
        E, I, kappa, A, G = 1, 1, 5/6, 1, 1
        q = betainc(3, 6, x)
        
        # Sistema de ecuaciones
        d2phi = q / (E * I)
        ddphi = d2phi
        
        dwdx = phi - (1/(kappa*A*G)) * E * I * d2phi
        ddw = dwdx
        
        return [dphi, ddphi, dw, ddw]
    
    # Resolver BVP como IVP
    x_span = [0, 1]
    x_eval = np.linspace(0, 1, 100)
    y0 = [0, 0, 0, 0]  # Condiciones iniciales en x=0
    
    sol = solve_ivp(beam_equations, x_span, y0, t_eval=x_eval, method='RK45')
    
    # Extraer soluciones
    phi = sol.y[0]
    w = sol.y[2]
    
    # Coordenadas de la viga
    y_top = 0.2
    y_bottom = -0.2
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Viga sin deformar
    ax1.fill_between(x_eval, y_bottom*np.ones_like(x_eval), 
                     y_top*np.ones_like(x_eval), 
                     alpha=0.3, color='blue', label='Sin deformar')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Viga sin deformar')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Viga deformada
    x_deformed = x_eval - y_top * phi
    y_top_deformed = y_top + w
    y_bottom_deformed = y_bottom + w
    
    ax2.fill_between(x_eval, y_bottom*np.ones_like(x_eval), 
                     y_top*np.ones_like(x_eval), 
                     alpha=0.3, color='blue', label='Sin deformar')
    ax2.fill_between(x_deformed, y_bottom_deformed, y_top_deformed, 
                     alpha=0.5, color='red', label='Deformada')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Viga deformada bajo carga')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('6.pdf')
    plt.close()


problema_4()
problema_6()