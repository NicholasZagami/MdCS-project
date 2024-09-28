import matplotlib.pyplot as plt
import pandas as pd

def plot_results(method_name, column_name, title):
    results = pd.read_csv(f"../../tests/results/results.csv")
    results['Matrice'] = results['Matrice'].apply(lambda x: x.split('/')[-1].split('.')[0])
    results['Tempo'] = results['Tempo'].apply(lambda x: round(x, 2))

    results = results[results['Metodo'] == method_name]
    pivot_df = results.pivot(index='Matrice', columns='Tolleranza', values=column_name)

    plt.figure(figsize=(50, 50))
    ax = pivot_df.plot(kind='bar', rot=0, width=0.8)
    ax.set_xlabel("Matrici")
    ax.set_ylabel(column_name)
    ax.set_title(title)

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.savefig(f"../../tests/results/{method_name}-{column_name}.png", dpi=300)

def plot_overall_results(column_name, title):
    results = pd.read_csv(f"../../tests/results/results.csv")
    results['Matrice'] = results['Matrice'].apply(lambda x: x.split('/')[-1].split('.')[0])
    results['Tempo'] = results['Tempo'].apply(lambda x: round(x, 2))
    matrices = results['Matrice'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, matrix in enumerate(matrices):
        subset_results = results[results['Matrice'] == matrix]
        pivot_df = subset_results.pivot_table(index='Tolleranza', columns='Metodo', values=column_name)

        if column_name == 'Iterazioni':
            pivot_df = pivot_df.astype(int)

        ax = pivot_df.plot(kind='bar', rot=0, ax=axs[i], width=0.8)
        ax.set_xlabel("Tolleranza")
        ax.set_ylabel(column_name)
        ax.set_title(f"{title} per la matrice {matrix}")

        for p in ax.patches:
            ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"../../tests/results/overall-{column_name}.png", dpi=300)


methods = ["Jacobi", "Gaub Seidel", "Gradiente", "Gradiente coniugato"]
matrices = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]

for method in methods:
    plot_results(method, 'Iterazioni', f"Numero di iterazioni richieste per il metodo di {method}")
    plot_results(method, 'Tempo', f"Tempi di esecuzioni totale per il metodo di {method}")

plot_overall_results('Iterazioni', "Numero di iterazioni richieste")
plot_overall_results('Tempo', "Tempo di esecuzione richiesto")