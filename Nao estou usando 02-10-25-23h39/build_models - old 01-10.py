# utils_ls_models.py
import numpy as np

def build_dataset_ls14pp(rows, rep_map=None, window=50):
    """
    Constrói os datasets híbridos LS14PP.
    Garantindo que todos os inputs tenham shapes corretos:
    X_seq: (n_samples, window, 25)
    X_hist: (n_samples, 15)
    X_freq: (n_samples, 15)
    X_atraso: (n_samples, 15)
    X_global: (n_samples, 15)
    """
    n_samples = len(rows) - window
    n_numbers = 25  # total de dezenas da Lotofácil

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_hist = np.zeros((n_samples, 15), dtype=np.float32)
    X_freq = np.zeros((n_samples, 15), dtype=np.float32)
    X_atraso = np.zeros((n_samples, 15), dtype=np.float32)
    X_global = np.zeros((n_samples, 15), dtype=np.float32)
    y = np.zeros((n_samples, 15), dtype=np.float32)

    for i in range(n_samples):
        window_rows = rows[i:i+window]
        # Preencher X_seq
        for j, row in enumerate(window_rows):
            for num in row['numbers']:
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0
        
        # Exemplo de preenchimento seguro: pegar últimas 15 dezenas do row mais recente
        last_row = rows[i+window]
        nums = last_row['numbers'][:15]  # garante 15 números
        X_hist[i, :len(nums)] = 1.0
        X_freq[i, :len(nums)] = 1.0
        X_atraso[i, :len(nums)] = 1.0
        X_global[i, :len(nums)] = 1.0
        y[i, :len(nums)] = 1.0

    return X_seq, X_hist, X_freq, X_atraso, X_global, y


def build_dataset_ls15pp(rows, window=50):
    """
    Constrói datasets LS15PP
    X_seq: (n_samples, window, 25)
    X_freq: (n_samples, 15)
    X_atraso: (n_samples, 15)
    X_global: (n_samples, 15)
    """
    n_samples = len(rows) - window
    n_numbers = 25

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_freq = np.zeros((n_samples, 15), dtype=np.float32)
    X_atraso = np.zeros((n_samples, 15), dtype=np.float32)
    X_global = np.zeros((n_samples, 15), dtype=np.float32)
    y = np.zeros((n_samples, 15), dtype=np.float32)

    for i in range(n_samples):
        window_rows = rows[i:i+window]
        # Preencher X_seq
        for j, row in enumerate(window_rows):
            for num in row['numbers']:
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0

        last_row = rows[i+window]
        nums = last_row['numbers'][:15]
        X_freq[i, :len(nums)] = 1.0
        X_atraso[i, :len(nums)] = 1.0
        X_global[i, :len(nums)] = 1.0
        y[i, :len(nums)] = 1.0

    return X_seq, X_freq, X_atraso, X_global, y
