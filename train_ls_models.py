# train_ls_models.py
"""
Treinador para LS15 (sequência) e LS14 (híbrido: sequência + qtd_repetidos).

Compatibilidade:
- Mantém funções: train_ls14(...), train_ls15(...), e train(...)
- CLI: --model {ls14, ls15, both}
- Salva: modelo_ls14.h5 e modelo_ls15.h5 (compatível com carregar_modelo_ls14/15)
- Entrada LS14 = [seq_input=(1,window,25), hist_input=(1,1)] com qtd_repetidos/15.0

Uso típico:
python train_ls_models.py --model ls14 --last_n 500 --window 50 --epochs 120 --batch 32 --lr 0.001 --pos_weight 4.0 --out ./models
"""

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
    """Converte 15 dezenas -> vetor binário de 25 posições."""
    b = np.zeros(25, dtype=np.uint8)
    for n in jogo:
        b[int(n) - 1] = 1
    return b


def fetch_history(last_n=None, include_repeats=False):
    """
    Retorna resultados em ordem cronológica (mais antigo -> mais recente).
    rows: cada item = [concurso, n1..n15].
    Se include_repeats=True, retorna (rows, rep_map) com {concurso_atual: qtd_repetidos}.
    """
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
    """
    LS15: X=(N, window, 25), y=(N,25)
    """
    X, y = [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]       # usa n1..n15
        target = to_binary(rows[i + window][1:])
        X.append(seq)
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_ls14(rows, rep_map, window):
    """
    LS14 híbrido: (seq_input, hist_input) -> y
    seq_input = (N, window, 25)
    hist_input = (N, 1)  onde valor = qtd_repetidos/15.0 do CONCURSO TARGET
    """
    X_seq, X_hist, y = [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]
        target_row = rows[i + window]
        target = to_binary(target_row[1:])
        concurso_target = int(target_row[0])
        repeticoes = rep_map.get(concurso_target, 0) / 15.0  # normalizado
        X_seq.append(seq)
        X_hist.append([repeticoes])
        y.append(target)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_hist, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )


# -------------------- modelos --------------------
def weighted_bce(pos_weight=3.0):
    """Binary cross-entropy com peso para classe positiva (para LS14)."""
    def loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        loss_pos = - pos_weight * y_true * K.log(y_pred)
        loss_neg = - (1.0 - y_true) * K.log(1.0 - y_pred)
        return K.mean(loss_pos + loss_neg)
    return loss


def build_lstm_ls15(seq_shape, lr=1e-3):
    """
    seq_shape=(window,25). Saída 25 sigmoid.
    """
    seq_input = Input(shape=seq_shape, name="seq_input")
    x = LSTM(128, return_sequences=False)(seq_input)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(25, activation="sigmoid")(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


def build_lstm_ls14_hybrid(seq_shape, hist_shape, lr=1e-3, pos_weight=3.0):
    """
    Híbrido: entrada 1 (seqência binária), entrada 2 (qtd_repetidos normalizado).
    """
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


# -------------------- treino (API compatível) --------------------
def train_ls15(last_n=500, window=50, epochs=100, batch=32, lr=1e-3, out="."):
    """
    Mantido para compatibilidade.
    """
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
    """
    Mantido para compatibilidade. LS14 híbrido (duas entradas).
    """
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


def train(model_name, last_n=None, window=50, epochs=50, batch=32, lr=1e-3, pos_weight=3.0, out="."):
    """
    Mantém a API anterior baseada em 'model_name'.
    """
    if model_name.lower() == "ls15":
        return train_ls15(last_n=last_n or 500, window=window, epochs=epochs, batch=batch, lr=lr, out=out)
    elif model_name.lower() == "ls14":
        return train_ls14(last_n=last_n or 500, window=window, epochs=epochs, batch=batch, lr=lr, pos_weight=pos_weight, out=out)
    else:
        raise ValueError("model_name deve ser 'ls15' ou 'ls14'.")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ls15", "ls14", "both"])
    parser.add_argument("--last_n", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=3.0, help="Peso positivo para LS14 (apenas LS14).")
    parser.add_argument("--out", type=str, default=".")
    args = parser.parse_args()

    if args.model in ("ls15", "both"):
        train_ls15(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr, out=args.out)

    if args.model in ("ls14", "both"):
        train_ls14(last_n=args.last_n, window=args.window, epochs=args.epochs, batch=args.batch, lr=args.lr,
                   pos_weight=args.pos_weight, out=args.out)
