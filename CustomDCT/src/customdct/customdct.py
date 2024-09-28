import numpy as np
from math import ceil, sqrt


def compute_dct(values):
    num_of_values = len(values)
    steps = np.array([((2 * step + 1) / (2 * num_of_values)) * np.pi
                      for step in range(num_of_values)])
    cosine_base = np.array([np.cos(steps * freq) for freq in range(num_of_values)])
    cos_values = np.zeros(num_of_values)
    cos_values[0] = np.dot(values, cosine_base[0]) / num_of_values
    cos_values[1:num_of_values] = [np.dot(values, cosine_base[i]) * 2 / num_of_values
                                   for i in range(1, num_of_values)]
    return cos_values


def compute_dct2(matrix, win_size=8):
    rows, cols = matrix.shape
    approx_matrix = (matrix.astype("float64") - 128)

    #Aggiungo un padding alla matrice approssimata qualora
    #le sue dimensioni non fossero divisibili per 8
    padding_rows = win_size * ceil(rows / win_size) - rows
    padding_cols = win_size * ceil(cols / win_size) - cols
    if (padding_cols != 0 or padding_rows != 0):
        approx_matrix = np.pad(approx_matrix, ((0, padding_rows),
                                               (0, padding_cols)),
                                                mode='edge')

    #Scorro l'immagine a blocchi 8x8 partendo da (0,0) e discendendo l'immagine
    for row in range(0, approx_matrix.shape[0], win_size):
        for col in range(0, approx_matrix.shape[1], win_size):
            #Recupero una porzione (finestra) dell'immagine
            for sample_col in range(col, col + win_size):
                approx_matrix[row:row + win_size, sample_col] = compute_dct(
                    approx_matrix[row:row + win_size, sample_col])
            for sample_row in range(row, row + win_size):
                approx_matrix[sample_row, col:col + win_size] = compute_dct(
                    approx_matrix[sample_row, col:col + win_size])
    approx_matrix = approx_matrix[:rows, :cols]
    return (approx_matrix).astype('int64')


def compute_idct(values):
    num_of_values = len(values)
    canonical_values = np.zeros(num_of_values)
    steps = np.array([((2 * step + 1) / (2 * num_of_values)) * np.pi
                      for step in range(num_of_values)])
    cosine_base = np.array([np.cos(steps * freq)
                            for freq in range(num_of_values)])
    for i in range(num_of_values):
        canonical_values += values[i] * cosine_base[i]
    return canonical_values


def compute_idct2(encoded_matrix, win_size=8):
    rows, cols = encoded_matrix.shape
    approx_matrix = encoded_matrix.astype("float64")

    padding_rows = win_size * ceil(rows / win_size) - rows
    padding_cols = win_size * ceil(cols / win_size) - cols
    if (padding_cols != 0 or padding_rows != 0):
        approx_matrix = np.pad(approx_matrix, ((0, padding_rows),
                                               (0, padding_cols)),
                               mode='edge')

    # Scorro l'immagine a blocchi 8x8 partendo da (0,0) e discendendo l'immagine
    for row in range(0, approx_matrix.shape[0], win_size):
        for col in range(0, approx_matrix.shape[1], win_size):
            for sample_row in range(row, row + win_size):
                approx_matrix[sample_row, col:col + win_size] = compute_idct(
                    approx_matrix[sample_row, col:col + win_size])
            for sample_col in range(col, col + win_size):
                approx_matrix[row:row + win_size, sample_col] = compute_idct(
                    approx_matrix[row:row + win_size, sample_col])
    approx_matrix = approx_matrix[:rows, :cols]
    approx_matrix = approx_matrix + 128
    #Riporto i valori nel range [0,255]
    min = approx_matrix.min()
    max = approx_matrix.max()
    approx_matrix = ((approx_matrix - min) / (max - min)) * 255
    return approx_matrix.astype("uint8")
