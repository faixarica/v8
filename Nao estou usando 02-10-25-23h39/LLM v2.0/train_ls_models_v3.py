# train_ls_models_v3.py
import os
import argparse
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from db_utils import fetch_history  # precisa estar implementado corretamente

# ------------------------------
# BUILD DATASETS
# ------------------------------
def build_dataset_ls14pp(rows, window=50):
    n_samples = len(rows) - window
    n_numbers = 25

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_hist = np.zeros((n_samples, 15), dtype=np.float32)
    X_freq = np.zeros((n_samples, 15), dtype=np.float32)
    X_atraso = np.zeros((n_samples, 15), dtype=np.float32)
    X_global = np.zeros((n_samples, 15), dtype=np.float32)
    y = np.zeros((n_samples, n_numbers), dtype=np.float32)  # agora 25 posições

    for i in range(n_samples):
        window_rows = rows[i:i+window]
        for j, row in enumerate(window_rows):
            for num in (row['numbers'] if isinstance(row, dict) else row):
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0

        last_row = rows[i+window]
        nums = (last_row['numbers'] if isinstance(last_row, dict) else last_row)
        for num in nums:
            if 1 <= num <= n_numbers:
                y[i, num-1] = 1.0

        # features auxiliares (placeholder simples)
        X_hist[i, :len(nums[:15])] = 1.0
        X_freq[i, :len(nums[:15])] = 1.0
        X_atraso[i, :len(nums[:15])] = 1.0
        X_global[i, :len(nums[:15])] = 1.0

    return X_seq, X_hist, X_freq, X_atraso, X_global, y


def build_dataset_ls15pp(rows, window=50):
    n_samples = len(rows) - window
    n_numbers = 25

    X_seq = np.zeros((n_samples, window, n_numbers), dtype=np.float32)
    X_freq = np.zeros((n_samples, 15), dtype=np.float32)
    X_atraso = np.zeros((n_samples, 15), dtype=np.float32)
    X_global = np.zeros((n_samples, 15), dtype=np.float32)
    y = np.zeros((n_samples, n_numbers), dtype=np.float32)  # agora 25 posições

    for i in range(n_samples):
        window_rows = rows[i:i+window]
        for j, row in enumerate(window_rows):
            for num in (row['numbers'] if isinstance(row, dict) else row):
                if 1 <= num <= n_numbers:
                    X_seq[i, j, num-1] = 1.0

        last_row = rows[i+window]
        nums = (last_row['numbers'] if isinstance(last_row, dict) else last_row)
        for num in nums:
            if 1 <= num <= n_numbers:
                y[i, num-1] = 1.0

        # features auxiliares (placeholder simples)
        X_freq[i, :len(nums[:15])] = 1.0
        X_atraso[i, :len(nums[:15])] = 1.0
        X_global[i, :len(nums[:15])] = 1.0

    return X_seq, X_freq, X_atraso, X_global, y


# ------------------------------
# BUILD MODEL LS14PP HYBRID
# ------------------------------
def build_lstm_ls14pp_hybrid(input_shape, lr=0.001):
    X_seq_input = Input(shape=input_shape, name="X_seq")
    X_hist_input = Input(shape=(15,), name="X_hist")
    X_freq_input = Input(shape=(15,), name="X_freq")
    X_atraso_input = Input(shape=(15,), name="X_atraso")
    X_global_input = Input(shape=(15,), name="X_global")

    x = LSTM(64)(X_seq_input)
    x = Concatenate()([x, X_hist_input, X_freq_input, X_atraso_input, X_global_input])
    output = Dense(25, activation="sigmoid")(x)  # saída 25

    model = Model(
        inputs=[X_seq_input, X_hist_input, X_freq_input, X_atraso_input, X_global_input],
        outputs=output
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


# ------------------------------
# TRAIN LS14PP
# ------------------------------
def train_ls14pp(last_n=1500, window=50, epochs=100, batch=16):
    print(f"Treinando LS14PP | last_n={last_n}, window={window}, epochs={epochs}, batch={batch}")
    rows = fetch_history(last_n=last_n)
    if not rows or len(rows) <= window:
        raise ValueError("[ERRO] Nenhum dado retornado para treino LS14PP!")

    X_seq, X_hist, X_freq, X_atraso, X_global, y = build_dataset_ls14pp(rows, window=window)
    model = build_lstm_ls14pp_hybrid(input_shape=(window, 25))

    model.fit(
        x=[X_seq, X_hist, X_freq, X_atraso, X_global],
        y=y,
        epochs=epochs,
        batch_size=batch,
        verbose=2
    )
    return model


# ------------------------------
# TRAIN LS15PP
# ------------------------------
def train_ls15pp(last_n=1500, window=50, epochs=100, batch=16):
    print(f"Treinando LS15PP | last_n={last_n}, window={window}, epochs={epochs}, batch={batch}")
    rows = fetch_history(last_n=last_n)
    if not rows or len(rows) <= window:
        raise ValueError("[ERRO] Nenhum dado retornado para treino LS15PP!")

    X_seq, X_freq, X_atraso, X_global, y = build_dataset_ls15pp(rows, window=window)

    X_seq_input = Input(shape=(window, 25), name="X_seq")
    X_freq_input = Input(shape=(15,), name="X_freq")
    X_atraso_input = Input(shape=(15,), name="X_atraso")
    X_global_input = Input(shape=(15,), name="X_global")

    x = LSTM(64)(X_seq_input)
    x = Concatenate()([x, X_freq_input, X_atraso_input, X_global_input])
    output = Dense(25, activation="sigmoid")(x)  # saída 25

    model = Model(
        inputs=[X_seq_input, X_freq_input, X_atraso_input, X_global_input],
        outputs=output
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")

    model.fit(
        x=[X_seq, X_freq, X_atraso, X_global],
        y=y,
        epochs=epochs,
        batch_size=batch,
        verbose=2
    )
    return model

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["ls14pp", "ls15pp"])
    parser.add_argument("--last_n", type=int, default=1500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=str, default="./tmp")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.model == "ls14pp":
        model = train_ls14pp(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch)
    else:
        model = train_ls15pp(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch)

    out_path = os.path.join(args.out, f"{args.model}_model.keras")
    model.save(out_path)
    print(f"[OK] Modelo salvo em: {out_path}")
