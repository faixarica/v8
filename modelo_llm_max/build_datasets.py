# build_datasets.py
import numpy as np
from modelo_llm_max.utils_ls_models import to_binary

def build_dataset_ls15pp(rows, window):
    X_seq, X_freq, X_atraso, X_global, y = [], [], [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        target_row = rows[i + window]

        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(target_row[1:])

        freq_vec = np.sum([to_binary(r[1:]) for r in seq_rows], axis=0) / float(window)

        atraso_vec = np.zeros(25, dtype=np.float32)
        for d in range(1, 26):
            atraso = 0
            for past in reversed(seq_rows):
                atraso += 1
                if d in past[1:]:
                    break
            atraso_vec[d-1] = atraso / float(window)

        soma = sum(target_row[1:]) / (25.0 * 15.0)
        pares = sum(1 for x in target_row[1:] if x % 2 == 0) / 15.0

        X_seq.append(seq)
        X_freq.append(freq_vec)
        X_atraso.append(atraso_vec)
        X_global.append([soma, pares])
        y.append(target)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_freq, dtype=np.float32),
        np.array(X_atraso, dtype=np.float32),
        np.array(X_global, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )

def build_dataset_ls14pp(rows, rep_map, window):
    X_seq, X_hist, X_freq, X_atraso, X_global, y = [], [], [], [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        target_row = rows[i + window]

        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(target_row[1:])
        concurso_target = int(target_row[0])

        repeticoes = rep_map.get(concurso_target, 0) / 15.0
        freq_vec = np.sum([to_binary(r[1:]) for r in seq_rows], axis=0) / float(window)

        atraso_vec = np.zeros(25, dtype=np.float32)
        for d in range(1, 26):
            atraso = 0
            for past in reversed(seq_rows):
                atraso += 1
                if d in past[1:]:
                    break
            atraso_vec[d-1] = atraso / float(window)

        soma = sum(target_row[1:]) / (25.0 * 15.0)
        pares = sum(1 for x in target_row[1:] if x % 2 == 0) / 15.0

        X_seq.append(seq)
        X_hist.append([repeticoes])
        X_freq.append(freq_vec)
        X_atraso.append(atraso_vec)
        X_global.append([soma, pares])
        y.append(target)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_hist, dtype=np.float32),
        np.array(X_freq, dtype=np.float32),
        np.array(X_atraso, dtype=np.float32),
        np.array(X_global, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )