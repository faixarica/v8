import argparse
import logging
import os
import numpy as np

from sqlalchemy import text
from db import Session

from tensorflow.keras.models import load_model

from train_ls_models import (
    train_ls14,
    train_ls14pp,
    train_ls15,
    train_ls15pp,
    to_binary,
    fetch_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------- avaliação --------------------
def top15_from_probs(probs):
    return np.argsort(probs)[-15:] + 1

def evaluate_model(model, model_name, lookback=200, window=50):
    """
    Backtest simples: roda últimos `lookback` concursos, gera top-15,
    compara com resultado real e coleta métricas.
    """
    rows = fetch_history(last_n=lookback + window)
    hits = []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i+window]
        target_row = rows[i+window]
        target = set(target_row[1:])

        if model_name in ("ls14", "ls14pp"):
            # ls14 precisa de repetição / extras
            # aqui simplificamos: só sequência binária
            seq = np.array([to_binary(r[1:]) for r in seq_rows], dtype=np.float32)
            X = np.expand_dims(seq, axis=0)
            probs = model.predict(X, verbose=0)[0]
        else:
            # ls15 e ls15pp usam só sequência (ou extras)
            seq = np.array([to_binary(r[1:]) for r in seq_rows], dtype=np.float32)
            X = np.expand_dims(seq, axis=0)
            probs = model.predict(X, verbose=0)[0]

        pred = set(top15_from_probs(probs))
        acertos = len(pred.intersection(target))
        hits.append(acertos)

    hits = np.array(hits)
    taxa13 = np.mean(hits >= 13)
    taxa14 = np.mean(hits >= 14)
    return {
        "taxa13": float(taxa13),
        "taxa14": float(taxa14),
        "hist": hits.tolist(),
    }

# -------------------- pipeline --------------------
def auto_retrain(model_name, last_n=2000, window=50, epochs=80, batch=32, lr=1e-3, pos_weight=3.0, eval_lookback=300, out="./models"):
    os.makedirs(out, exist_ok=True)
    chk_path = os.path.join(out, f"modelo_{model_name}.h5")

    # 1. carregar modelo existente se houver
    old_model, old_metrics = None, None
    if os.path.exists(chk_path):
        old_model = load_model(chk_path, compile=False)
        old_metrics = evaluate_model(old_model, model_name, lookback=eval_lookback, window=window)
        logging.info(f"Métricas atuais {model_name}: {old_metrics}")

    # 2. treinar novo modelo
    logging.info(f"Treinando novo {model_name}...")
    if model_name == "ls14":
        model = train_ls14(last_n=last_n, window=window, epochs=epochs, batch=batch, lr=lr, pos_weight=pos_weight, out=out)
    elif model_name == "ls14pp":
        model = train_ls14pp(last_n=last_n, window=window, epochs=epochs, batch=batch, lr=lr, pos_weight=pos_weight, out=out)
    elif model_name == "ls15":
        model = train_ls15(last_n=last_n, window=window, epochs=epochs, batch=batch, lr=lr, out=out)
    elif model_name == "ls15pp":
        model = train_ls15pp(last_n=last_n, window=window, epochs=epochs, batch=batch, lr=lr, out=out)
    else:
        raise ValueError("Modelo inválido")

    # 3. avaliar novo modelo
    new_metrics = evaluate_model(model, model_name, lookback=eval_lookback, window=window)
    logging.info(f"Novo {model_name}: {new_metrics}")

    # 4. decisão de promoção
    promote = False
    if old_metrics is None:
        promote = True
    else:
        if new_metrics["taxa14"] >= old_metrics["taxa14"]:
            promote = True

    if promote:
        logging.info(f"Promovendo novo modelo {model_name}...")
        model.save(chk_path)
    else:
        logging.info(f"Novo modelo {model_name} descartado (pior que o atual).")

    # 5. salvar no banco
    db = Session()
    try:
        q = text("""
        INSERT INTO modelo_status(modelo, taxa13, taxa14, params, caminho)
        VALUES(:modelo, :taxa13, :taxa14, :params, :caminho)
        """)
        db.execute(q, {
            "modelo": model_name,
            "taxa13": new_metrics["taxa13"],
            "taxa14": new_metrics["taxa14"],
            "params": str({"last_n": last_n, "window": window, "epochs": epochs, "batch": batch, "lr": lr}),
            "caminho": chk_path,
        })
        db.commit()
    finally:
        db.close()

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=["ls14", "ls14pp", "ls15", "ls15pp"])
    parser.add_argument("--last_n", type=int, default=2000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--eval_lookback", type=int, default=300)
    parser.add_argument("--out", type=str, default="./models")
    args = parser.parse_args()

    auto_retrain(
        model_name=args.name,
        last_n=args.last_n,
        window=args.window,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        pos_weight=args.pos_weight,
        eval_lookback=args.eval_lookback,
        out=args.out,
    )
