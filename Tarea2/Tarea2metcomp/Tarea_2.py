#Librerias
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy import ndimage as ndi
from PIL import Image
import os, glob

#Utilidades punto 1

def Fourier_transform(f, y, t):
    f = np.asarray(f)
    y = np.asarray(y)
    t = np.asarray(t)
    M = len(f)
    F = np.zeros(M, dtype=complex)
    dt = t[1] - t[0]
    for i in range(M):
        F[i] = np.sum(y * np.exp(-2j * np.pi * f[i] * t)) * dt
    return F

def generate_data(tmax,dt,A,freq,noise):
    
    ts = np.arange(0,tmax+dt,dt)
    return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)

def Eliminate_max(datos, margen):
    datos = np.array(datos)
    margen = int(margen)   
    idx_max = np.argmax(datos)  
    inicio = max(0, idx_max - margen)
    fin = min(len(datos), idx_max + margen + 1)
    datos_filtrados = np.delete(datos, np.arange(inicio, fin))
    return datos_filtrados

def fwhm(x, y):
    x = np.array(x)
    y = np.array(y)
    i_max = np.argmax(y)
    y_max = y[i_max]
    if y_max <= 0:
        return np.nan
    half = y_max / 2.0
    # izquierda
    iL = np.where(y[:i_max] <= half)[0]
    if len(iL) == 0:
        return np.nan
    i1 = iL[-1]
    i2 = i1 + 1
    t = (half - y[i1]) / (y[i2] - y[i1] + 1e-12)
    x_left = x[i1] + t * (x[i2] - x[i1])
    # derecha
    iR = np.where(y[i_max:] <= half)[0]
    if len(iR) == 0:
        return np.nan
    i1 = iR[0] + i_max
    i0 = i1 - 1
    t = (half - y[i0]) / (y[i1] - y[i0] + 1e-12)
    x_right = x[i0] + t * (x[i1] - x[i0])

    # ancho
    width = x_right - x_left
    return width

#Punto 1
#a
f0 = 5  # Hz
t, r = generate_data(10, 0.001, 5, f0, 0.1)
f_vec = np.linspace(-f0*2.7,f0*2.7, 401)  
T_F = Fourier_transform(f_vec, r, t)

plt.figure(figsize=(10,5))
plt.plot(f_vec, np.abs(T_F))
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|F(f)|")
plt.title("transformada")
plt.grid(True)
plt.savefig("1.a.pdf", dpi=300, bbox_inches="tight")

#b
LisSNt=[]
LisSNf=[]
for i in range(0,201):
    f0=np.random.randint(10,1000)/100
    A=np.random.randint(1,3)/10
    noise=np.random.randint(10,500)/100
    a=f0*1.5
    t, r=generate_data(a,0.001,A,f0,noise)
    f_vec = np.linspace(0,a, 401)
    F=Fourier_transform(f_vec, r, t)
    SNt=A/f0
    SNf_i=Eliminate_max(F,1.2/a)
    SNf=float(np.std(SNf_i))
    LisSNt.append(SNt)   
    LisSNf.append(SNf)   

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(LisSNf, LisSNt, s=10, c="blue", marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("STD espectral sin pico (SNf) Log-Log")
plt.ylabel("A/f0 (SNt)")
plt.title("Relación SNf vs SNt (Log-Log)")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(LisSNf, LisSNt, s=10, c="blue", marker="o")
plt.xlabel("STD espectral sin pico (SNf)")
plt.ylabel("A/f0 (SNt)")
plt.title("Relación SNf vs SNt (Lineal)")
plt.grid(True)

plt.suptitle("1.b Relación SNf vs SNt", fontsize=14)  
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig("1.b.pdf", dpi=300, bbox_inches="tight")

#c
LisAp=[]
LisTmax=[]
for i in range(1,21):
    f0=5
    A=5
    tmax=f0*1*i
    t, r=generate_data(tmax,0.001,A,f0,0.1)
    f_vec = np.linspace(0,f0*1.5, 401)
    F=Fourier_transform(f_vec, r, t)
    Ap=fwhm(f_vec,F)
    LisAp.append(Ap)
    LisTmax.append(tmax)

    
plt.figure()
plt.scatter(LisTmax, LisAp, s=10, c="blue", marker="o")  
plt.xlabel("Tmax")
plt.ylabel("A/f0 (SNt)")
plt.title("Relación SNf vs SNt")
plt.grid(True)
plt.savefig("1.c.pdf", dpi=300, bbox_inches="tight")


#Utilidades 2
def EPPS0(x, y):

    x = np.array(x)
    y = np.array(y)

    fft_vals = np.fft.fft(y)
    fft_freqs = np.fft.fftfreq(len(x), d=(x[1] - x[0]))  


    positive_freqs = fft_freqs[:len(fft_freqs)//2]
    magnitudes = np.abs(fft_vals[:len(fft_vals)//2])

    peaks, _ = find_peaks(magnitudes)

    if len(peaks) == 0:
        return None 

    
    return positive_freqs[peaks[1]]

def max_loc(arr):
    indices=[] 
    
    for i in range(1, len(arr)-1):
        if arr[i-1] < arr[i] and arr[i] > arr[i+1]:   
                               
            indices.append(i)
    return indices 

def FPB (x):
    y=x+(10*np.cos(x))
    return y

#2
#a
Datos = "SN_d_tot_V2.0.csv"

array_datos = []
with open(Datos, newline='', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        array_datos.append(fila)
         
del array_datos[0]
lisv=[]
for i in range(len(array_datos)):
    for j in  range(len(array_datos[i])):
        a=float(array_datos[i][j])
        array_datos[i][j] = a
    if array_datos[i][4] == -1.0:
        lisv.append(i)

Datc = [elem for i, elem in enumerate(array_datos) if i not in lisv]

lisf=[]
liss=[]
for i in range(len(Datc)):
    lisf.append(Datc[i][3]-1818)
    liss.append(abs(Datc[i][4]))
    
x = np.array(lisf)
y = np.array(liss)

f_interp = interp1d(x, y, kind='cubic')
x_new = np.linspace(x.min(), x.max(), len(array_datos))
y_new = abs(f_interp(x_new))


plt.figure()
plt.plot(x_new, y_new, label="Interpolación cúbica")
plt.legend()
plt.savefig("2.a.pdf", dpi=300, bbox_inches="tight")
plt.close()

#2
f = EPPS0(x_new, y_new)
P = 1/f


np.savetxt("P.txt", np.atleast_1d(P))


y_cop=[]
for i in range(0,len(x_new)):
    fil=FPB(x_new[i])
    if y_new[i] < fil:
       y_cop.append(y_new[i])
    else:
        y_cop.append(0)
y_copu=np.array(y_cop)


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(x_new, y_new, 'b')
axs[0].set_title("Normal")

axs[1].plot(x_new, y_copu, 'r')
axs[1].set_title("Con filtro")

plt.tight_layout()
plt.savefig("2.b.data.pdf", dpi=300, bbox_inches="tight")
plt.close()


maxi = max_loc(y_copu)

fechas=[]
som=[]
for i in range(len(x_new)):
    if i in maxi:
        fechas.append(x_new[i])
        som.append(y_copu[i])
Fa=np.array(fechas)
Sa=np.array(som)

plt.figure()
plt.plot(Fa, Sa, label="Interpolación cúbica")
plt.legend()
plt.savefig("2.b.maxima.pdf", dpi=300, bbox_inches="tight")
plt.close()

#Utilidades 3

# 3.a: blur gaussiano en frecuencia 
def fft2_gaussian_blur_rgb(img_rgb: np.ndarray, sigma_frac: float = 0.06) -> np.ndarray:
    """
    Aplica desenfoque gaussiano multiplicando la FFT2 por una gaussiana 2D en frecuencia.
    img_rgb: uint8 (H,W,3).  sigma_frac ~ 0.04–0.10 (fracción de tamaño de imagen).
    Retorna float en [0,1].
    """
    img = img_rgb.astype(float) / 255.0
    H, W, _ = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma = sigma_frac * max(H, W)
    G = np.exp(-r2 / (2.0 * sigma**2))  # filtro pasa-bajas suave

    out = np.empty_like(img)
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(img[..., c]))
        Ff = F * G
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))
    out = np.clip(out, 0, 1)
    return out

#3.b: eliminación de ruido periódico por “notches”
def _notch_zero(F, y, x, hw):
    """Pone a cero un rectángulo centrado en (y,x) de semiancho hw, recortando bordes."""
    H, W = F.shape
    y1, y2 = max(0, y - hw), min(H, y + hw + 1)
    x1, x2 = max(0, x - hw), min(W, x + hw + 1)
    F[y1:y2, x1:x2] = 0

def remove_periodic_noise(gray: np.ndarray, k_peaks: int = 14,
                          r0: int = 8, notch_hw: int = 4, tau: float = 6.0) -> np.ndarray:
    """
    Quita ruido periódico anulando picos brillantes del espectro (fftshift) fuera del DC.
    - k_peaks: máximos a intentar anular (se hace simétricamente).
    - r0: radio del disco central que se ignora (DC y muy bajas f).
    - notch_hw: semiancho del “notch” (tamaño ~ (2*hw+1)^2).
    - tau: umbral relativo (pico > tau * mediana de magnitudes restantes).
    Retorna float en [0,1].
    """
    img = gray.astype(float) / 255.0
    H, W = img.shape
    cy, cx = H // 2, W // 2

    F = np.fft.fftshift(np.fft.fft2(img))
    Mag = np.abs(F)

    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    band = (r > r0) & (r < 0.49 * min(H, W))  # no tocar DC ni el borde extremo
    work = Mag.copy()
    work[~band] = 0

    Fclean = F.copy()
    for _ in range(k_peaks):
        i = int(np.argmax(work))
        val = float(work.flat[i])
        if val <= 0:
            break
        med = np.median(work[work > 0])
        if med <= 0 or val < tau * med:
            break
        y, x = np.unravel_index(i, work.shape)

        # Notch en el pico y en su simétrico (asegura salida real)
        _notch_zero(Fclean, y, x, notch_hw)
        ys, xs = 2 * cy - y, 2 * cx - x
        _notch_zero(Fclean, ys, xs, notch_hw)

        # Evitar elegir los mismos píxeles otra vez
        _notch_zero(work, y, x, notch_hw)
        _notch_zero(work, ys, xs, notch_hw)

    clean = np.real(np.fft.ifft2(np.fft.ifftshift(Fclean)))
    # Normalización robusta a [0,1]
    vmin, vmax = np.percentile(clean, [0.5, 99.5])
    clean = np.clip((clean - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return clean

# helpers de IO
def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))

def save_u8(img01: np.ndarray, path: str):
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(path)

# 3
# 3.a — Desenfoque (Miette)
miette_path = "miette.jpg"     # usa tu ruta si difiere
img_rgb = load_rgb(miette_path)
blur = fft2_gaussian_blur_rgb(img_rgb, sigma_frac=0.06)  # ajusta 0.04–0.10 si quieres más/menos blur
save_u8(blur, "3.a.jpg")


# 3.b.a — Ruido periódico (p_a_t_o)
pato_path = "p_a_t_o.jpg"      # usa tu ruta si difiere
pato_gray = load_gray(pato_path)
pato_clean = remove_periodic_noise(pato_gray, k_peaks=18, r0=10, notch_hw=5, tau=6.0)
save_u8(pato_clean, "3.b.a.jpg")


# 3.b.b — G_a_t_o (si no existe, usa miette como fallback)
candidates = []
for ext in ("png", "jpg", "jpeg"):
    candidates += glob.glob(f"g_a_t_o.{ext}")
gato_src = candidates[0] if candidates else miette_path  # fallback
gato_gray = load_gray(gato_src)
gato_clean = remove_periodic_noise(gato_gray, k_peaks=18, r0=10, notch_hw=5, tau=6.0)
save_u8(gato_clean, "3.b.b.png")

#4

datos = np.loadtxt("OGLE-LMC-CEP-0001.dat")
t = datos[:, 0]       
y = datos[:, 1]        

baseline = t.max() - t.min()
df = 1.0 / baseline / 5.0
freqs = np.arange(0.01, 2.0, df)

power = []
for f in freqs:
    c = np.cos(2*np.pi*f*t)
    s = np.sin(2*np.pi*f*t)
    X = np.column_stack([c, s])
    coef, _, _, _ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    A, B = coef
    power.append(0.5*(A*A + B*B))

power = np.array(power)
f0 = freqs[np.argmax(power)]   


phi = np.mod(f0 * t, 1.0)


plt.figure(figsize=(7,5))
plt.scatter(phi, y, s=10, alpha=0.7)
plt.xlabel("Fase φ = mod(f₀·t, 1)")
plt.ylabel("Brillo")
plt.title("Curva de luz doblada en fase")
plt.tight_layout()
plt.savefig("4.pdf", bbox_inches="tight")

#Utilidades 5

def clean_filtered_backprojection(sinogram, angles=None):
    """
    Clean filtered backprojection implementation focused on image clarity
    """
    # Esperado: filas = ángulos, columnas = detectores. Si no, transpone.
    sino = np.array(sinogram, dtype=float)
    if sino.shape[0] < sino.shape[1]:
        sino = sino.T
    
    n_angles, n_detectors = sino.shape
    
    if angles is None:
        angles = np.linspace(0, 180, n_angles, endpoint=False)
    
    # Output image size
    img_size = n_detectors
    
    # Cree grids
    center = img_size // 2
    x = np.arange(img_size) - center
    y = np.arange(img_size) - center
    X, Y = np.meshgrid(x, y)

    # Reconstrucción
    reconstruction = np.zeros((img_size, img_size))
    
    # Aplicar filtro a cada proyección
    filtered_projections = np.zeros_like(sino)
    
    for i in range(n_angles):
        proj = sino[i, :]

        # Zero-padding para mejorar la FFT
        padded_size = 2 ** int(np.ceil(np.log2(2 * len(proj))))
        proj_padded = np.zeros(padded_size)
        start_idx = (padded_size - len(proj)) // 2
        proj_padded[start_idx:start_idx + len(proj)] = proj

        # Aplicar filtro Ram-Lak
        proj_fft = np.fft.fft(proj_padded)
        freqs = np.fft.fftfreq(padded_size)

        # Ram-Lak (ramp) filtro con ventana de Hann
        ramp = np.abs(freqs)
        # Suavizar el filtro para reducir artefactos
        hann = 0.5 * (1 + np.cos(2 * np.pi * freqs))
        filter_kernel = ramp * hann

        # Aplique el filtro
        filtered_fft = proj_fft * filter_kernel
        filtered_proj_padded = np.real(np.fft.ifft(filtered_fft))

        # Extraer proyección filtrada
        filtered_projections[i, :] = filtered_proj_padded[start_idx:start_idx + len(proj)]
    
    # Retroproyección de proyecciones filtradas
    for i, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)

        # Calcular la coordenada de proyección para cada píxel
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Proyectar coordenadas sobre la línea del detector
        t = X * cos_a + Y * sin_a

        # Posiciones de los detectores
        detector_positions = np.arange(n_detectors) - n_detectors // 2

        # Interpolar valores de proyección
        projection_values = np.interp(t, detector_positions, filtered_projections[i, :],
                                    left=0, right=0)

        # Agregar a la reconstrucción
        reconstruction += projection_values

    # Aplicar factor de escala
    reconstruction *= np.pi / (2 * n_angles)

    # Aplicar máscara circular para eliminar artefactos en los bordes
    radius = min(img_size // 2 - 2, n_detectors // 2)
    mask = (X**2 + Y**2) <= radius**2
    reconstruction[~mask] = 0
    
    return reconstruction

# Cargar y procesar el sinograma
sinogram_path = "1.npy"
output_path = "4.png"


S = np.load(sinogram_path)


# Asegurar la orientación adecuada
if S.shape[0] < S.shape[1]:
    S = S.T

# Realizar reconstrucción limpia
reconstruction = clean_filtered_backprojection(S)

# Normalizar y mejorar
def normalize_image(img):
    """Normalización robusta utilizando percentiles"""
    # Eliminar ceros y valores muy pequeños para el cálculo de percentiles
    nonzero_vals = img[img > np.percentile(img, 5)]
    if len(nonzero_vals) > 0:
        p1, p99 = np.percentile(nonzero_vals, [1, 99])
        normalized = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
    else:
        normalized = img / (img.max() + 1e-8)
    return normalized

# Normalizar la reconstrucción
recon_norm = normalize_image(reconstruction)

# Convertir a formato de imagen
img_u8 = (recon_norm * 255).astype(np.uint8)
Image.fromarray(img_u8).save(output_path)
