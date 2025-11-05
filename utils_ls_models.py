# utils_ls_models.py
import numpy as np

def build_dataset_ls14pp(rows, rep_map=None, window=50):
    """
    Constrói datasets híbridos LS14PP com padding seguro.
    Shapes esperados:
        X_seq: (n_samples, window, 25)
        X_hist, X_freq, X_atraso, X_global: (n_samples, 15)
        y: (n_samples, 15)
    """
    n_samples = len(rows) - window
    n_numbers = 25  # total de dezenas da Lotofácil
    n_output = 15   # número de dezenas por input

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_hist = np.zeros((n_samples, n_output), dtype=np.float32)
    X_freq = np.zeros((n_samples, n_output), dtype=np.float32)
    X_atraso = np.zeros((n_samples, n_output), dtype=np.float32)
    X_global = np.zeros((n_samples, n_output), dtype=np.float32)
    y = np.zeros((n_samples, n_output), dtype=np.float32)

    for i in range(n_samples):
        window_rows = rows[i:i+window]

        # Preencher X_seq
        for j, row in enumerate(window_rows):
            for num in row['numbers']:
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0

        # Última linha do window
        last_row = rows[i + window]
        nums = last_row['numbers'][:n_output]
        
        # Padding seguro: garante exatamente 15 posições
        padded = np.zeros(n_output, dtype=np.float32)
        padded[:len(nums)] = 1.0

        X_hist[i] = padded
        X_freq[i] = padded
        X_atraso[i] = padded
        X_global[i] = padded
        y[i] = padded

    return X_seq, X_hist, X_freq, X_atraso, X_global, y


def build_dataset_ls15pp(rows, window=50):
    """
    Constrói datasets LS15PP com padding seguro.
    Shapes esperados:
        X_seq: (n_samples, window, 25)
        X_freq, X_atraso, X_global: (n_samples, 15)
        y: (n_samples, 15)
    """
    n_samples = len(rows) - window
    n_numbers = 25
    n_output = 15

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_freq = np.zeros((n_samples, n_output), dtype=np.float32)
    X_atraso = np.zeros((n_samples, n_output), dtype=np.float32)
    X_global = np.zeros((n_samples, n_output), dtype=np.float32)
    y = np.zeros((n_samples, n_output), dtype=np.float32)

    for i in range(n_samples):
        window_rows = rows[i:i+window]

        # Preencher X_seq
        for j, row in enumerate(window_rows):
            for num in row['numbers']:
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0

        last_row = rows[i + window]
        nums = last_row['numbers'][:n_output]

        padded = np.zeros(n_output, dtype=np.float32)
        padded[:len(nums)] = 1.0

        X_freq[i] = padded
        X_atraso[i] = padded
        X_global[i] = padded
        y[i] = padded

    return X_seq, X_freq, X_atraso, X_global, y
