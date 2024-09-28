import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


class LinearSolver:

    @staticmethod
    def create_matrix_from_file(filename):
        matrix = mmread(filename).toarray()
        return matrix

    @staticmethod
    def compute_cond(matrix):
        return np.linalg.cond(matrix)


class StationarySystemSolver(LinearSolver):
    def __stationary_solve(self, matrix, factor, operation, target_vector=None, max_iters=20000, tol=1e-4):
        result_vector = np.zeros(matrix.shape[0])  # x^0
        if target_vector is None:
            target_vector = np.ones(matrix.shape[0])  # b
        else:
            assert len(target_vector) == matrix.shape[0], "Target length should be equal to the number of rows in matrix!"

        counter = 0

        residual_denominator = 1 / np.linalg.norm(target_vector)  # Calcolo la norma del vettore target (||b||)

        while counter < max_iters:
            dot_product = matrix @ result_vector  # Calcolo il prodotto scalare matrice * vettore risultato (Ax^k)
            residue = target_vector - dot_product  # Calcolo il residuo come b - Axk
            scalar_residual = np.linalg.norm(residue) * residual_denominator  # Calcolo il residuo scalato

            if scalar_residual < tol:  # Criterio di arresto
                return counter, False

            else:
                result_vector = result_vector + operation(factor, residue)  # trovo il nuovo passo iterativo (x^k+1)

            counter += 1
        return counter, True

    def solve_with_jacobi(self, matrix, target=None, max_iters=20000, tol=1e-4):
        splitting_ls = 1 / np.diag(matrix)  # Estraggo la diagonale di A e trovo la sua inversa (P^-1)
        operator = lambda x, y: (x * y)
        counter = self.__stationary_solve(matrix, splitting_ls, operator, target, max_iters, tol)
        return counter

    def solve_with_gaub_seidel(self, matrix, target=None, max_iters=20000, tol=1e-4):
        L = csr_matrix(np.tril(matrix))  # Estraggo la parte triangolare inferiore di A (L)
        operator = lambda x, y: self.solve_triangular(L, y)  # Risolvo il sistema Py = r(k)
        counter = self.__stationary_solve(matrix, None, operator, target, max_iters, tol)
        return counter

    def solve_triangular(self, L, b, lower=True):
        n = L.shape[0]

        x = np.zeros_like(b, dtype=L.dtype)

        for i in range(n):
            row_start = L.indptr[i]
            row_end = L.indptr[i + 1]

            row_values = L.data[row_start:row_end]
            column_indices = L.indices[row_start:row_end]

            dot_product = sum(value * x[col] for value, col in zip(row_values, column_indices) if col < i)

            diagonal_element = row_values[column_indices == i][0] if column_indices[column_indices == i].size else None

            if diagonal_element is not None:
                x[i] = (b[i] - dot_product) / diagonal_element
            else:
                x[i] = 0
        return x


class NonStationarySystemSolver(LinearSolver):

    def solve_with_gradient(self, matrix, target_vector=None, max_iters=20000, tol=1e-4):
        result_vector = np.zeros(matrix.shape[0])  # x^0
        if target_vector is None:
            target_vector = np.ones(matrix.shape[0])  # b
        else:
            assert len(target_vector) == matrix.shape[
                0], "Target length should be equal to the number of rows in matrix!"

        counter = 0
        scalar_residual = 0

        residual_denominator = 1 / np.linalg.norm(target_vector)  # Calcolo la norma del vettore target (||b||)

        while counter < max_iters:
            dot_product = np.dot(matrix,
                                 result_vector)  # Calcolo il prodotto scalare matrice * vettore risultato (Ax^k)
            residue = target_vector - dot_product  # Calcolo il residuo come b - Axk
            alpha = np.dot(residue, residue) / np.dot(residue, np.dot(matrix, residue))  # Calcolo il coefficiente alpha
            scalar_residual = np.linalg.norm(residue) * residual_denominator  # Calcolo il residuo scalato
            if scalar_residual < tol:  # Criterio di arresto
                return counter, False
            else:
                result_vector = result_vector + (alpha * residue)  # trovo il nuovo passo iterativo (x^k+1)
            counter += 1
        return counter, True

    def solve_with_conjugate_gradient(self, matrix, target_vector=None, max_iters=20000, tol=1e-4):
        result_vector = np.zeros(matrix.shape[0])  # x^0
        if target_vector is None:
            target_vector = np.ones(matrix.shape[0])  # b
        else:
            assert len(target_vector) == matrix.shape[
                0], "Target length should be equal to the number of rows in matrix!"

        counter = 0
        scalar_residual = 0

        # Calcolo del residuo e della direzione (che è uguale al residuo)
        residual_denominator = 1 / np.linalg.norm(target_vector)
        dot_product = np.dot(matrix, result_vector)
        residue = target_vector - dot_product
        direction = residue

        #Calcolo dell'intensità di movimento e il passo iterativo
        alpha = np.dot(direction, residue) / np.dot(direction, np.dot(matrix, direction))
        result_vector = result_vector + (alpha * direction)

        # Calcolo il residuo e verifico la convergenza
        scalar_residual = np.linalg.norm(residue) * residual_denominator
        if scalar_residual < tol:
            return counter, False
        else:
            # Calcolo il nuovo residuo e la nuova direzione
            dot_product = np.dot(matrix, result_vector)
            residue = target_vector - dot_product
            beta = np.dot(direction, np.dot(matrix, residue)) / np.dot(direction, np.dot(matrix, direction))
            direction = residue - (beta * direction)

        # Itero fino alla convergenza
        while counter < max_iters:
            dot_product = np.dot(matrix, result_vector)
            residue = target_vector - dot_product
            alpha = np.dot(direction, residue) / np.dot(direction, np.dot(matrix, direction))
            result_vector = result_vector + (alpha * direction)
            scalar_residual = np.linalg.norm(residue) * residual_denominator
            if scalar_residual < tol:
                print(f"Converged in {counter} iterations")
                return counter, False
            else:
                dot_product = np.dot(matrix, result_vector)
                residue = target_vector - dot_product
                beta = np.dot(direction, np.dot(matrix, residue)) / np.dot(direction, np.dot(matrix, direction))
                direction = residue - (beta * direction)

            counter += 1

        return counter, True
