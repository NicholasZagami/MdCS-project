import numpy as np
import scipy.fft as fft

test_matrix = np.array(
    [[231, 32, 233, 161, 24, 71, 140, 245],
     [247, 40, 248, 245, 124, 204, 36, 107],
     [234, 202, 245, 167, 9, 217, 239, 173],
     [193, 190, 100, 167, 43, 180, 8, 70],
     [11, 24, 210, 177, 81, 243, 8, 112],
     [97, 195, 203, 47, 125, 114, 165, 181],
     [193, 70, 174, 167, 41, 30, 127, 245],
     [87, 149, 57, 192, 65, 129, 178, 228]])

matrix_first_row = [231, 32, 233, 161, 24, 71, 140, 245]

matrix_dct2 = fft.dct(test_matrix, axis=0, norm='ortho', type=2)
matrix_dct2 = fft.dct(matrix_dct2, axis=1, norm='ortho', type=2)

monodimensional_dct = fft.dct(matrix_first_row, type=2, norm='ortho')

print("DCT2 con fft: \n", matrix_dct2)
print("DCT2 con fft su riga: \n", monodimensional_dct)
