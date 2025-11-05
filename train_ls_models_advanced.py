# train_ls_models_advanced.py
import os
import argparse
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from db_utils import fetch_history

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------
def build_features(rows, n_numbers=25):
    """Calcula frequência, atraso e estatísticas globais de cada número."""
    n_rows = len(rows)
    freq_matrix = np.zeros((n_rows, n_numbers), dtype=np.float32)
    atraso_matrix = np.zeros((n_rows, n_numbers), dtype=np.float32)
    global_matrix = np.zeros((n_rows, n_numbers), dtype=np.float32)

    last_seen = np.zeros(n_numbers, dtype=int)
    count_total = np.zeros(n_numbers, dtype=int)

    for i, row in enumerate(rows):
        nums = row['numbers'] if isinstance(row, dict) else row
        for num in nums:
            idx = num - 1
            count_total[idx] += 1
            last_seen[idx] = 0
        for idx in range(n_numbers):
            atraso_matrix[i, idx] = last_seen[idx]
            freq_matrix[i, idx] = count_total[idx]
            global_matrix[i, idx] = count_total[idx] / (i+1)
            last_seen[idx] += 1

    # Normaliza frequências e atrasos
    freq_matrix /= (freq_matrix.max() + 1e-8)
    atraso_matrix /= (atraso_matrix.max() + 1e-8)
    return freq_matrix, atraso_matrix, global_matrix

# ------------------------------
# BUILD DATASETS
# ------------------------------
def build_dataset(rows, window=50, n_numbers=25):
    n_samples = len(rows) - window
    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_freq = np.zeros((n_samples, n_numbers), dtype=np.float32)
    X_atraso = np.zeros((n_samples, n_numbers), dtype=np.float32)
    X_global = np.zeros((n_samples, n_numbers), dtype=np.float32)
    y = np.zeros((n_samples, n_numbers), dtype=np.float32)

    freq_matrix, atraso_matrix, global_matrix = build_features(rows, n_numbers)

    for i in range(n_samples):
        window_rows = rows[i:i+window]
        for j, row in enumerate(window_rows):
            nums = row['numbers'] if isinstance(row, dict) else row
            for num in nums:
                X_seq[i, j, num-1] = 1.0
        last_row = rows[i+window]
        nums = last_row['numbers'] if isinstance(last_row, dict) else last_row
        for num in nums:
            y[i, num-1] = 1.0

        X_freq[i] = freq_matrix[i+window]
        X_atraso[i] = atraso_matrix[i+window]
        X_global[i] = global_matrix[i+window]

    return X_seq, X_freq, X_atraso, X_global, y

# ------------------------------
# BUILD MODEL
# ------------------------------
def build_lstm_model(window=50, n_numbers=25, lr=0.001, dropout=0.2):
    X_seq_input = Input(shape=(window, n_numbers), name="X_seq")
    X_freq_input = Input(shape=(n_numbers,), name="X_freq")
    X_atraso_input = Input(shape=(n_numbers,), name="X_atraso")
    X_global_input = Input(shape=(n_numbers,), name="X_global")

    x = LSTM(128)(X_seq_input)
    x = Dropout(dropout)(x)
    x = Concatenate()([x, X_freq_input, X_atraso_input, X_global_input])
    output = Dense(n_numbers, activation="sigmoid")(x)

    model = Model(inputs=[X_seq_input, X_freq_input, X_atraso_input, X_global_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

# ------------------------------
# TRAIN MODEL
# ------------------------------
def train_model(last_n=1500, window=50, epochs=100, batch=16, out="./tmp"):
    print(f"Treinando modelo avançado | last_n={last_n}, window={window}, epochs={epochs}, batch={batch}")
    rows = fetch_history(last_n=last_n)
    if not rows or len(rows) <= window:
        raise ValueError("[ERRO] Nenhum dado retornado para treino!")

    X_seq, X_freq, X_atraso, X_global, y = build_dataset(rows, window=window)
    model = build_lstm_model(window=window)

    model.fit(
        x=[X_seq, X_freq, X_atraso, X_global],
        y=y,
        epochs=epochs,
        batch_size=batch,
        verbose=2
    )

    os.makedirs(out, exist_ok=True)
    out_path = os.path.join(out, f"ls_model_advanced.keras")
    model.save(out_path)
    print(f"[OK] Modelo salvo em: {out_path}")
    return model

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_n", type=int, default=1500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=str, default="./tmp")
    args = parser.parse_args()

    train_model(
        last_n=args.last_n,
        window=args.window,
        epochs=args.epochs,
        batch=args.batch,
        out=args.out
    )
