import csv
from time import perf_counter
import numpy as np
import src as lss


def measure_time_jacobi(matrix, method, max_iters=50000, tol=1e-4):
    start_time = perf_counter()
    target_vector = np.ones(matrix.shape[0])
    counter = method(matrix, target_vector, max_iters, tol)
    end_time = perf_counter()
    tot_time = end_time - start_time
    print(f"Time elapsed (Jacobi): {round(tot_time, 2)}s")
    print("Number of iterations: {}".format(counter))
    return counter, tot_time


def measure_time_gaub_seidel(matrix, method, max_iters=50000, tol=1e-4):
    start_time = perf_counter()
    target_vector = np.ones(matrix.shape[0])
    counter = method(matrix, target_vector, max_iters, tol)
    end_time = perf_counter()
    tot_time = end_time - start_time
    print(f"Time elapsed (Gaub Seidel): {round(tot_time, 2)}s")
    print("Number of iterations: {}".format(counter))
    return counter, tot_time


def measure_time_gradient(matrix, method, max_iters=20000, tol=1e-4):
    start_time = perf_counter()
    target_vector = np.ones(matrix.shape[0])
    counter = method(matrix, target_vector, max_iters, tol)
    end_time = perf_counter()
    tot_time = end_time - start_time
    print(f"Time elapsed (Gradient): {round(tot_time, 2)}s")
    print("Number of iterations: {}".format(counter))
    return counter, tot_time


def measure_time_conjugate_gradient(matrix, method, max_iters=20000, tol=1e-4):
    start_time = perf_counter()
    target_vector = np.ones(matrix.shape[0])
    counter = method(matrix, target_vector, max_iters, tol)
    end_time = perf_counter()
    tot_time = end_time - start_time
    print(f"Time elapsed (Conjugate Gradient): {round(tot_time, 2)}s")
    print("Number of iterations: {}".format(counter))
    return counter, tot_time


matrices = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]

stat_solver = lss.StationarySystemSolver()
non_stat_solver = lss.NonStationarySystemSolver()

headers = ["Matrice", "Tolleranza", "Tempo", "Iterazioni"]

measure_time_methods = {
    'Jacobi': measure_time_jacobi,
    'Gaub Seidel': measure_time_gaub_seidel,
    'Gradiente': measure_time_gradient,
    'Gradiente coniugato': measure_time_conjugate_gradient
}


def run_and_write_results(matrix, method, method_name, max_iters=25000, tol_values=[1e-4, 1e-6, 1e-8, 1e-10]):
    measure_time = measure_time_methods[method_name]

    for tol in tol_values:
        print(f"============== Tol = {tol} ======================")
        tot_iteration, tot_time = measure_time(matrix, method, max_iters=max_iters, tol=tol)
        with open(f"results/results.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([method_name, matrix_name, tol, tot_time, tot_iteration])

headers = ["Metodo", "Matrice", "Tolleranza", "Tempo", "Iterazioni"]

with open(f"results/results.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

for matrix_name in matrices:
    matrix_name = "data/" + matrix_name
    print("#################", matrix_name, "#################")
    matrix = stat_solver.create_matrix_from_file(matrix_name)
    matrix_cond = stat_solver.compute_cond(matrix)

    run_and_write_results(matrix, stat_solver.solve_with_jacobi, 'Jacobi')
    run_and_write_results(matrix, stat_solver.solve_with_gaub_seidel, 'Gaub Seidel')
    run_and_write_results(matrix, non_stat_solver.solve_with_gradient, 'Gradiente')
    run_and_write_results(matrix, non_stat_solver.solve_with_conjugate_gradient, 'Gradiente coniugato')