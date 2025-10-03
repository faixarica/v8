import argparse
import logging
import os
import numpy as np
from sqlalchemy import text
from db import Session  # usa seu db.py (Neon/SQLite/etc)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
from build_models import build_lstm_ls15pp, build_lstm_ls14pp_hybrid
import tensorflow as tf
from utils_ls_models import fetch_history, build_dataset_ls15pp, build_dataset_ls14pp
from build_models import build_lstm_ls15pp, build_lstm_ls14pp_hybrid


OUT_DIR = "./backtest_models"
os.makedirs(OUT_DIR, exist_ok=True)

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

#-------- nova loss e callback --------------------
def bce_with_topk_proxy(k=15, tau=0.05, alpha=1.0, pos_weight=1.0):
    def loss(y_true, y_pred):
        eps = K.epsilon()
        p = K.clip(y_pred, eps, 1.0 - eps)
        loss_pos = - pos_weight * y_true * K.log(p)
        loss_neg = - (1.0 - y_true) * K.log(1.0 - p)
        bce = K.mean(loss_pos + loss_neg, axis=-1)
        logit = K.log(p / (1.0 - p + eps) + eps)
        soft = K.softmax(logit / (tau + eps), axis=-1)
        exp_hits = K.sum(soft * y_true, axis=-1) * K.cast(k, K.floatx())
        reward = exp_hits / K.cast(k, K.floatx())
        combined = bce - alpha * reward
        return combined
    return loss

class LoteriaMetricsCallback(Callback):
    def __init__(self, val_data, k=15, save_best_path=None, verbose=1):
        super().__init__()
        self.val_X, self.val_y = val_data
        self.k = k
        self.save_best_path = save_best_path
        self.best_mean_hits = -1.0
        self.verbose = verbose

    def _predict(self):
        preds = self.model.predict(self.val_X, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return np.asarray(preds)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self._predict()
        topk_idx = np.argsort(-preds, axis=1)[:, :self.k]
        n = preds.shape[0]
        hits = np.zeros(n, dtype=np.int32)
        for i in range(n):
            selected = np.zeros(preds.shape[1], dtype=np.uint8)
            selected[topk_idx[i]] = 1
            hits[i] = int(np.sum(selected * self.val_y[i]))

        mean_hits = float(hits.mean())
        pct_ge_11 = float((hits >= 11).sum()) / n
        pct_ge_12 = float((hits >= 12).sum()) / n
        pct_ge_13 = float((hits >= 13).sum()) / n
        pct_ge_14 = float((hits >= 14).sum()) / n

        logs['val_mean_hits'] = mean_hits
        logs['val_pct_ge_11'] = pct_ge_11
        logs['val_pct_ge_12'] = pct_ge_12
        logs['val_pct_ge_13'] = pct_ge_13
        logs['val_pct_ge_14'] = pct_ge_14

        if self.verbose:
            print(f"\n[LOT] ep {epoch+1}: mean_hits={mean_hits:.4f} | >=11: {pct_ge_11:.2%} >=12: {pct_ge_12:.2%} >=13: {pct_ge_13:.2%} >=14: {pct_ge_14:.2%}")

        if self.save_best_path and mean_hits > self.best_mean_hits:
            self.best_mean_hits = mean_hits
            self.model.save(self.save_best_path)
            if self.verbose:
                print(f"[LOT] Novo melhor salvo em {self.save_best_path} (mean_hits={mean_hits:.4f})")

def train_val_split(X, y, val_frac=0.2, seed=42):
    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(n * (1 - val_frac))
    train_idx = idx[:split]
    val_idx = idx[split:]
    if isinstance(X, list):
        X_train = [a[train_idx] for a in X]
        X_val = [a[val_idx] for a in X]
    else:
        X_train = X[train_idx]
        X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    return (X_train, y_train), (X_val, y_val)

# -------------------- funções de treino --------------------

from build_models import build_lstm_ls14pp_hybrid
from utils_ls_models import fetch_history, build_dataset_ls14pp
from tensorflow.keras.optimizers import Adam

def train_ls14pp(last_n=1500, window=50, epochs=200, batch=16, lr=1e-3, pos_weight=3.0, out="."):
    """
    Treina o modelo LS14PP híbrido e salva em arquivo .keras
    """
    print(f"Treinando LS14PP | last_n={last_n}, window={window}, epochs={epochs}, batch={batch}")

    # ------------------- Preparar dados -------------------
    rows, rep_map = fetch_history(last_n=last_n, include_repeats=True)
    X_seq, X_hist, X_freq, X_atraso, X_global, y = build_dataset_ls14pp(rows, rep_map, window)
    X = [X_seq, X_hist, X_freq, X_atraso, X_global]

    # Split treino/teste
    TEST_SIZE = min(150, len(y)//3)
    n = len(y)
    train_idx = list(range(0, n - TEST_SIZE))
    test_idx = list(range(n - TEST_SIZE, n))

    X_train = [a[train_idx] for a in X]
    X_test = [a[test_idx] for a in X]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Amostras treino: {len(y_train)}, teste: {len(y_test)}")

    # ------------------- Construir modelo -------------------
    model = build_lstm_ls14pp_hybrid((window, 25), lr=lr)

    # ------------------- Callbacks -------------------
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # ------------------- Treino -------------------
    # Aplica pos_weight via class_weight
    class_weight = {0: 1.0, 1: pos_weight}

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch,
        callbacks=[es, rlrop],
        class_weight=class_weight,
        verbose=2
    )

    # ------------------- Salvar modelo -------------------
    import os
    os.makedirs(out, exist_ok=True)
    model_file = os.path.join(out, "ls14pp_model.keras")
    model.save(model_file)
    print(f"Modelo LS14PP salvo em {model_file}")

    return model

def train_ls15pp(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, alpha_topk=1.0, tau=0.05, out="."):
    os.makedirs(out, exist_ok=True)
    rows = fetch_history(last_n=last_n)
    X_seq, X_freq, X_atraso, X_global, y = build_dataset_ls15pp(rows, window)
    if len(X_seq) < 10:
        raise RuntimeError("Dados insuficientes para LS15++ com esses parâmetros.")

    X = [X_seq, X_freq, X_atraso, X_global]
    (X_train, y_train), (X_val, y_val) = train_val_split(X, y)

    model = build_lstm_ls15pp((window, 25), lr=lr)
    loss_fn = bce_with_topk_proxy(k=15, tau=tau, alpha=alpha_topk, pos_weight=1.0)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn)

    chk_path = os.path.join(out, "modelo_ls15pp.keras")
    best_by_metric = os.path.join(out, "best_ls15pp_by_meanhits.keras")

    cbs = [
        ModelCheckpoint(chk_path, save_best_only=True, monitor="val_loss", verbose=1),
        LoteriaMetricsCallback(val_data=(X_val, y_val), k=15, save_best_path=best_by_metric, verbose=1),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch, callbacks=cbs, verbose=2)
    return model

# ------------------ Gerador de palpites a partir de probabilidades ------------------
def generate_games_from_probs(probs, k=15, n_games=5, method='diversified', seed=None):
    """
    probs: vetor (25,) de probabilidades
    k: dezenas por jogo (15)
    n_games: quantos palpites gerar
    method:
      - 'topk_plus_samples': primeiro jogo = top-k, outros = amostragem ponderada
      - 'diversified': cria jogos adicionando ruído aos logits e pegando top-k
    retorna lista de arrays com dezenas (1..25)
    """
    rng = np.random.RandomState(seed)
    probs = np.asarray(probs).astype(float)
    if probs.ndim != 1 or probs.shape[0] != 25:
        raise ValueError("probs deve ser vetor de 25 elementos")

    games = []
    # determinístico top-k
    topk = np.argsort(-probs)[:k]
    games.append(np.sort(topk + 1))  # +1 para dezenas 1..25

    if method == 'topk_plus_samples':
        pnorm = probs / probs.sum()
        for _ in range(n_games - 1):
            u = rng.rand(25)
            keys = -np.log(u) / (pnorm + 1e-12)
            idx = np.argsort(keys)[:k]
            games.append(np.sort(idx + 1))
    else:  # diversified
        logit = np.log(probs / (probs.sum() + 1e-12) + 1e-12)
        for _ in range(n_games - 1):
            noise = rng.normal(scale=0.3, size=25)
            pert = logit + noise
            idx = np.argsort(-pert)[:k]
            games.append(np.sort(idx + 1))

    # remove duplicados
    unique_games = []
    seen = set()
    for g in games:
        tup = tuple(g.tolist())
        if tup not in seen:
            seen.add(tup)
            unique_games.append(g)
        if len(unique_games) >= n_games:
            break
    return unique_games

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ls15", "ls14", "ls14pp", "ls15pp", "both"])
    parser.add_argument("--last_n", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    if args.model in ("ls15", "both"):
        train_ls15(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr, out=args.out)

    if args.model in ("ls14", "both"):
        train_ls14(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr, pos_weight=args.pos_weight, out=args.out)

    if args.model == "ls14pp":
        train_ls14pp(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr, pos_weight=args.pos_weight, out=args.out)

    if args.model == "ls15pp":
        train_ls15pp(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr, out=args.out)
