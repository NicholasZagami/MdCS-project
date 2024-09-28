import numpy as np

from src.customdct.customdct import *
import matplotlib.pyplot as plt
from time import perf_counter
import scipy.fft as fft
from PIL import Image
from scipy.optimize import curve_fit

def func_N3(N, a):
    return a * N**3

def func_NlogN(N, b):
    return b * N * np.log(N)

def measure_custom_dct(matrix, win_size=8):
    start_time = perf_counter()
    compute_dct2(matrix, win_size=win_size)
    end_time = perf_counter()
    return end_time - start_time

def measure_fft_dct(matrix):
    start_time = perf_counter()
    fft.dct(matrix, type=2,axis=0)
    fft.dct(matrix, type=2, axis=1)
    end_time = perf_counter()
    return end_time - start_time

sizes = [20, 40, 80, 160, 320, 640]
images = ["20x20","40x40","80x80", "160x160","320x320","640x640"]
#sizes = sizes[:4]
values_custom = []
values_fft = []
for image in images[:]:
    filename = "img/"+image+".bmp"
    matrix = np.array(Image.open(filename))
    time_custom = measure_custom_dct(matrix)
    time_fft = measure_fft_dct(matrix)
    values_custom.append(time_custom)
    values_fft.append(time_fft)
    print("Immagine: ", image)
    print("Tempo impiegato dalla dct custom: ", time_custom,"s")
    print("Tempo impiegato dalla dct fft: ", time_fft,"s")

# Fit per il tuo algoritmo
popt_my_dct2, _ = curve_fit(func_N3, sizes, values_custom)
# Fit per l'algoritmo della libreria
popt_lib_dct2, _ = curve_fit(func_NlogN, sizes, values_fft)

plt.figure(figsize=(10, 5))
plt.semilogy(sizes, values_custom, 'o-', label='Custom DCT2')
plt.semilogy(sizes, func_N3(np.array(sizes), *popt_my_dct2), '--', label='Fit N^3')
plt.semilogy(sizes, values_fft, 's-', label='FFT DCT2')
plt.semilogy(sizes, func_NlogN(np.array(sizes), *popt_lib_dct2), '--', label='Fit N log(N)')
plt.minorticks_off()
plt.xticks(sizes)
plt.yticks(values_custom + values_fft, labels=[f"{i:.1e}" for i in (values_custom + values_fft)])
plt.xlabel('Dimensione N degli array (N x N)')
plt.ylabel('Tempo impiegato (s)')
plt.title('Tempo impiegato per DCT2 vs Dimensione N')
plt.legend()
plt.grid(True)
plt.savefig('result.png', dpi=300)