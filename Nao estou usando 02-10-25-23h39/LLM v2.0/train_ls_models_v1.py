import argparse
import logging
import os
import numpy as np

from sqlalchemy import text
from db import Session  # usa seu db.py (Neon/SQLite/etc)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------- utilidades --------------------
def to_binary(jogo):
    b = np.zeros(25, dtype=np.uint8)
    for n in jogo:
        try:
            idx = int(n) - 1
            if 0 <= idx < 25:
                b[idx] = 1
            else:
                logging.warning(f"[to_binary] Valor fora do intervalo 1-25 detectado: {n}")
        except Exception as e:
            logging.error(f"[to_binary] Erro ao processar valor {n}: {e}")
    return b

def fetch_history(last_n=None, include_repeats=False):
    db = Session()
    try:
        if last_n:
            q = text("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso DESC
                LIMIT :lim
            """)
            rows = db.execute(q, {"lim": last_n}).fetchall()
            rows = list(reversed(rows))
        else:
            q = text("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso ASC
            """)
            rows = db.execute(q).fetchall()

        rows = [[int(v) for v in r] for r in rows]

        if include_repeats:
            rep_rows = db.execute(text("SELECT concurso_atual, qtd_repetidos FROM repete")).fetchall()
            rep_map = {int(r[0]): int(r[1]) for r in rep_rows}
            return rows, rep_map

        return rows
    finally:
        try:
            db.close()
        except Exception:
            pass

# -------------------- datasets --------------------
def build_dataset_ls15(rows, window):
    X, y = [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(rows[i + window][1:])
        X.append(seq)
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_dataset_ls14(rows, rep_map, window):
    X_seq, X_hist, y = [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]
        target_row = rows[i + window]
        target = to_binary(target_row[1:])
        concurso_target = int(target_row[0])
        repeticoes = rep_map.get(concurso_target, 0) / 15.0
        X_seq.append(seq)
        X_hist.append([repeticoes])
        y.append(target)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_hist, dtype=np.float32),
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

# -------------------- modelos --------------------
def weighted_bce(pos_weight=3.0):
    def loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        loss_pos = - pos_weight * y_true * K.log(y_pred)
        loss_neg = - (1.0 - y_true) * K.log(1.0 - y_pred)
        return K.mean(loss_pos + loss_neg)
    return loss

def build_lstm_ls15(seq_shape, lr=1e-3):
    seq_input = Input(shape=seq_shape, name="seq_input")
    x = LSTM(128, return_sequences=False)(seq_input)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(25, activation="sigmoid")(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

def build_lstm_ls14_hybrid(seq_shape, hist_shape, lr=1e-3, pos_weight=3.0):
    seq_input = Input(shape=seq_shape, name="seq_input")
    hist_input = Input(shape=hist_shape, name="hist_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    h = Dense(8, activation="relu")(hist_input)

    z = Concatenate()([x, h])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.2)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, hist_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=weighted_bce(pos_weight=pos_weight))
    return model

def build_lstm_ls14pp_hybrid(seq_shape, lr=1e-3, pos_weight=3.0):
    seq_input = Input(shape=seq_shape, name="seq_input")
    hist_input = Input(shape=(1,), name="hist_input")
    freq_input = Input(shape=(25,), name="freq_input")
    atraso_input = Input(shape=(25,), name="atraso_input")
    global_input = Input(shape=(2,), name="global_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    h = Dense(8, activation="relu")(hist_input)
    f = Dense(32, activation="relu")(freq_input)
    a = Dense(32, activation="relu")(atraso_input)
    g = Dense(8, activation="relu")(global_input)

    z = Concatenate()([x, h, f, a, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, hist_input, freq_input, atraso_input, global_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=weighted_bce(pos_weight=pos_weight))
    return model

def build_lstm_ls15pp(seq_shape, lr=1e-3):
    seq_input = Input(shape=seq_shape, name="seq_input")
    freq_input = Input(shape=(25,), name="freq_input")
    atraso_input = Input(shape=(25,), name="atraso_input")
    global_input = Input(shape=(2,), name="global_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    f = Dense(32, activation="relu")(freq_input)
    a = Dense(32, activation="relu")(atraso_input)
    g = Dense(8, activation="relu")(global_input)

    z = Concatenate()([x, f, a, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, freq_input, atraso_input, global_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

# -------------------- treino --------------------
def train_ls15(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, out="."):
    os.makedirs(out, exist_ok=True)
    rows = fetch_history(last_n=last_n)
    X, y = build_dataset_ls15(rows, window)
    if len(X) < 10:
        raise RuntimeError("Dados insuficientes para LS15 com esses parâmetros.")

    model = build_lstm_ls15((window, 25), lr=lr)
    chk_path = os.path.join(out, "modelo_ls15.h5")
    cbs = [
        ModelCheckpoint(chk_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    logging.info("Treinando LS15...")
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch, callbacks=cbs, verbose=2)
    logging.info(f"LS15 salvo em {chk_path}")
    return model

def train_ls14(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, pos_weight=3.0, out="."):
    os.makedirs(out, exist_ok=True)
    rows, rep_map = fetch_history(last_n=last_n, include_repeats=True)
    X_seq, X_hist, y = build_dataset_ls14(rows, rep_map, window)
    if len(X_seq) < 10:
        raise RuntimeError("Dados insuficientes para LS14 com esses parâmetros.")

    model = build_lstm_ls14_hybrid((window, 25), (1,), lr=lr, pos_weight=pos_weight)
    chk_path = os.path.join(out, "modelo_ls14.h5")
    cbs = [
        ModelCheckpoint(chk_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    logging.info("Treinando LS14 híbrido...")
    model.fit([X_seq, X_hist], y, validation_split=0.2, epochs=epochs, batch_size=batch, callbacks=cbs, verbose=2)
    logging.info(f"LS14 híbrido salvo em {chk_path}")
    return model

def train_ls14pp(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, pos_weight=3.0, out="."):
    os.makedirs(out, exist_ok=True)
    rows, rep_map = fetch_history(last_n=last_n, include_repeats=True)
    X_seq, X_hist, X_freq, X_atraso, X_global, y = build_dataset_ls14pp(rows, rep_map, window)
    if len(X_seq) < 10:
        raise RuntimeError("Dados insuficientes para LS14++ com esses parâmetros.")

    model = build_lstm_ls14pp_hybrid((window, 25), lr=lr, pos_weight=pos_weight)
    chk_path = os.path.join(out, "modelo_ls14pp.keras")
    cbs = [
        ModelCheckpoint(chk_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    logging.info("Treinando LS14++ (híbrido avançado)...")
    model.fit([X_seq, X_hist, X_freq, X_atraso, X_global], y, validation_split=0.2, epochs=epochs, batch_size=batch, callbacks=cbs, verbose=2)
    logging.info(f"LS14++ salvo em {chk_path}")
    return model

def train_ls15pp(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, out="."):
    os.makedirs(out, exist_ok=True)
    rows = fetch_history(last_n=last_n)
    X_seq, X_freq, X_atraso, X_global, y = build_dataset_ls15pp(rows, window)
    if len(X_seq) < 10:
        raise RuntimeError("Dados insuficientes para LS15++ com esses parâmetros.")

    model = build_lstm_ls15pp((window, 25), lr=lr)
    chk_path = os.path.join(out, "modelo_ls15pp.keras")
    cbs = [
        ModelCheckpoint(chk_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    logging.info("Treinando LS15++ (avançado)...")
    model.fit([X_seq, X_freq, X_atraso, X_global], y, validation_split=0.2, epochs=epochs, batch_size=batch, callbacks=cbs, verbose=2)
    logging.info(f"LS15++ salvo em {chk_path}")
    return model

# -------------------- API compatível --------------------
def train(model_name, last_n=None, window=50, epochs=50, batch=32, lr=1e-3, pos_weight=3.0, out="."):
    if model_name.lower() == "ls15":
        return train_ls15(
            last_n=last_n or 500,
            window=window,
            epochs=epochs,
            batch=batch,
            lr=lr,
            out=out,
        )
    elif model_name.lower() == "ls14":
        return train_ls14(
            last_n=last_n or 500,
            window=window,
            epochs=epochs,
            batch=batch,
            lr=lr,
            pos_weight=pos_weight,
            out=out,
        )
    elif model_name.lower() == "ls14pp":
        return train_ls14pp(
            last_n=last_n or 500,
            window=window,
            epochs=epochs,
            batch=batch,
            lr=lr,
            pos_weight=pos_weight,
            out=out,
        )
    elif model_name.lower() == "ls15pp":
        return train_ls15pp(
            last_n=last_n or 500,
            window=window,
            epochs=epochs,
            batch=batch,
            lr=lr,
            out=out,
        )
    else:
        raise ValueError("model_name deve ser 'ls15', 'ls14', 'ls14pp' ou 'ls15pp'.")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ls15", "ls14", "ls14pp", "ls15pp", "both"])
    parser.add_argument("--last_n", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=3.0, help="Peso positivo para LS14/LS14++")
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    if args.model in ("ls15", "both"):
        train_ls15(
            last_n=args.last_n,
            window=args.window,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            out=args.out,
        )

    if args.model in ("ls14", "both"):
        train_ls14(
            last_n=args.last_n,
            window=args.window,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            pos_weight=args.pos_weight,
            out=args.out,
        )

    if args.model == "ls14pp":
        train_ls14pp(
            last_n=args.last_n,
            window=args.window,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            pos_weight=args.pos_weight,
            out=args.out,
        )

    if args.model == "ls15pp":
        train_ls15pp(
            last_n=args.last_n,
            window=args.window,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            out=args.out,
        )
