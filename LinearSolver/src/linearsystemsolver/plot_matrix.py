import matplotlib.pylab as plt
from scipy.io import mmread

matrices = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
base_path = "../../tests/data/"

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, matrix in enumerate(matrices):
    result_matrix = mmread(base_path + matrix).toarray()
    axs[i//2, i%2].spy(result_matrix, markersize=0.025)
    axs[i//2, i%2].set_title(matrix)

plt.tight_layout()
plt.savefig(f"../../tests/results/matrix.png")