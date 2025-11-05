# auto_retrain.py
"""
Pipeline de re-treino automático para LS14/LS14PP (híbrido) + avaliação contínua.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from contextlib import contextmanager

import numpy as np
from sqlalchemy import text

from db import Session  # seu db.py
from train_ls_models import train_ls14, train_ls14pp  # treinadores consolidados

from tensorflow.keras.models import load_model

# --------------------------------- LOG ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("auto_retrain")

# --------------------------------- UTILS ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def to_binary(jogo15):
    b = np.zeros(25, dtype=np.uint8)
    for n in jogo15:
        idx = int(n) - 1
        if 0 <= idx < 25:
            b[idx] = 1
    return b

@contextmanager
def db_session():
    db = Session()
    try:
        yield db
        db.close()
    except Exception:
        try:
            db.close()
        finally:
            raise

def ensure_status_table():
    """Cria tabela modelo_status se não existir (SQLite/Postgres compatível)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS modelo_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        modelo TEXT NOT NULL,
        versao TEXT NOT NULL,
        treinado_ate_concurso INTEGER,
        window_size  INTEGER,
        last_n INTEGER,
        epochs INTEGER,
        batch INTEGER,
        lr REAL,
        pos_weight REAL,
        eval_lookback INTEGER,
        taxa_13 REAL,
        taxa_14 REAL,
        taxa_13_ou_14 REAL,
        hits_histograma TEXT,
        saved_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        with db_session() as db:
            db.execute(text(ddl))
            db.commit()
    except Exception:
        ddl_pg = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        with db_session() as db:
            db.execute(text(ddl_pg))
            db.commit()

def get_latest_concurso(db):
    row = db.execute(text("SELECT MAX(concurso) FROM resultados_oficiais")).fetchone()
    return int(row[0]) if row and row[0] is not None else 0

def load_repete_map(db):
    rows = db.execute(text("SELECT concurso_atual, qtd_repetidos FROM repete")).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}

def fetch_rows(db, last_n=None, asc=True):
    if last_n:
        q = text("""
            SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT :lim
        """)
        rows = db.execute(q, {"lim": last_n}).fetchall()
        rows = list(reversed(rows))  # garantir ordem ascendente
    else:
        q = text("""
            SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY concurso ASC
        """)
        rows = db.execute(q).fetchall()
    # padroniza para int
    rows = [[int(x) for x in r] for r in rows]
    if not asc:
        rows = list(reversed(rows))
    return rows

def atomic_promote(new_path, final_path, backup_path):
    """Promove new_path -> final_path com backup de final_path."""
    if os.path.exists(final_path):
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.replace(final_path, backup_path)
        except Exception as e:
            logger.warning(f"Falha ao criar backup do modelo: {e}")
    os.replace(new_path, final_path)

def evaluate_ls_model(model_path, window_size, lookback_n):
    """Avalia modelo LS14 ou LS14PP (híbrido)."""
    model = load_model(model_path, compile=False)
    if hasattr(model, "inputs") and len(model.inputs) != 2:
        raise RuntimeError("O modelo carregado não é híbrido (precisa de 2 entradas).")

    with db_session() as db:
        rows = fetch_rows(db, last_n=lookback_n + window_size + 1, asc=True)
        rep_map = load_repete_map(db)

    if len(rows) < window_size + 2:
        raise RuntimeError("Histórico insuficiente para avaliação.")

    hits_list = []
    used_targets = 0

    for i in range(len(rows) - window_size):
        seq_rows = rows[i:i + window_size]
        target_row = rows[i + window_size]

        seq_input = np.array([[to_binary(r[1:]) for r in seq_rows]], dtype=np.float32)
        concurso_target = int(target_row[0])
        repeticoes = rep_map.get(concurso_target, 0) / 15.0
        hist_input = np.array([[repeticoes]], dtype=np.float32)

        pred = model.predict([seq_input, hist_input], verbose=0)[0]
        chosen = np.argsort(pred)[-15:] + 1
        chosen = set(chosen.tolist())
        reais = set(target_row[1:])
        hits = len(chosen & reais)
        hits_list.append(hits)
        used_targets += 1

    hits_arr = np.array(hits_list)
    taxa_13 = float(np.mean(hits_arr >= 13))
    taxa_14 = float(np.mean(hits_arr >= 14))
    taxa_13_14 = float(np.mean((hits_arr >= 13) & (hits_arr <= 14)))
    histograma = {k: int(np.sum(hits_arr == k)) for k in range(0, 16)}

    return {
        "targets": used_targets,
        "taxa_13": taxa_13,
        "taxa_14": taxa_14,
        "taxa_13_ou_14": taxa_13 + taxa_14 - float(np.mean(hits_arr >= 14)),
        "taxa_13_14_puro": taxa_13_14,
        "histograma": histograma,
        "hits_list": hits_list,
    }

def save_status_to_db(payload):
    ensure_status_table()
    with db_session() as db:
        db.execute(text("""
            INSERT INTO modelo_status (
                modelo, versao, treinado_ate_concurso, window_size, last_n, epochs, batch, lr, pos_weight,
                eval_lookback, taxa_13, taxa_14, taxa_13_ou_14, hits_histograma, saved_path
            ) VALUES (
                :modelo, :versao, :treinado_ate_concurso, :window_size, :last_n, :epochs, :batch, :lr, :pos_weight,
                :eval_lookback, :taxa_13, :taxa_14, :taxa_13_ou_14, :hits_histograma, :saved_path
            )
        """), payload)
        db.commit()

def read_last_status(model_name):
    try:
        ensure_status_table()
        with db_session() as db:
            row = db.execute(text("""
                SELECT modelo, versao, treinado_ate_concurso, window_size, last_n, epochs, batch, lr, pos_weight,
                       eval_lookback, taxa_13, taxa_14, taxa_13_ou_14, hits_histograma, saved_path, created_at
                FROM modelo_status
                WHERE modelo = :m
                ORDER BY id DESC
                LIMIT 1
            """), {"m": model_name}).fetchone()
            return row
    except Exception:
        return None

# ------------------------- PIPELINE UNIFICADA -------------------------
def pipeline_ls_model(args):
    """Pipeline unificada para LS14 / LS14PP."""
    model_name = args.name
    models_dir = os.path.abspath(args.models_dir)
    final_path = os.path.join(models_dir, f"modelo_{model_name}.h5")
    tmp_path   = os.path.join(models_dir, f"modelo_{model_name}.tmp.h5")
    bak_path   = os.path.join(models_dir, f"modelo_{model_name}.bak.h5")

    # 1) NOVOS CONCURSOS
    with db_session() as db:
        latest_concurso = get_latest_concurso(db)
    logger.info(f"Último concurso no banco: {latest_concurso}")

    # 2) AVALIA MODELO ATUAL
    modelo_existe = os.path.exists(final_path)
    current_metrics = None
    window = args.window_size

    if modelo_existe:
        try:
            current_metrics = evaluate_ls_model(final_path, window_size=window, lookback_n=args.eval_lookback)
            logger.info(f"[ATUAL] 13+: {current_metrics['taxa_13']:.2%} | 14+: {current_metrics['taxa_14']:.2%} "
                        f"| 13/14 puro: {current_metrics['taxa_13_14_puro']:.2%}")
        except Exception as e:
            logger.warning(f"Falha ao avaliar modelo atual: {e}")

    # 3) DECISÃO DE RE-TREINO
    last_status = read_last_status(model_name)
    last_trained_concurso = int(last_status[2]) if (last_status and last_status[2] is not None) else 0
    novos_concursos = max(0, latest_concurso - last_trained_concurso)

    disparar = False
    motivo = []

    if not modelo_existe:
        disparar = True
        motivo.append("modelo inexistente")
    else:
        if current_metrics:
            if current_metrics["taxa_13"] < args.thresh_13:
                disparar = True
                motivo.append(f"taxa_13<{args.thresh_13}")
            if current_metrics["taxa_14"] < args.thresh_14:
                disparar = True
                motivo.append(f"taxa_14<{args.thresh_14}")
        if novos_concursos >= args.min_new_draws:
            disparar = True
            motivo.append(f"{novos_concursos} concursos novos")

    logger.info(f"Decisão de re-treino: {disparar} ({', '.join(motivo) if motivo else 'sem motivo'})")

    # 4) RE-TREINO
    if disparar:
        logger.info(f"Iniciando re-treino {model_name} (híbrido)...")
        train_func = train_ls14 if model_name == "ls14" else train_ls14pp
        train_func(
            last_n=args.last_n,
            window=window,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            pos_weight=args.pos_weight,
            out=args.models_dir
        )

        # mover para tmp -> reavaliar
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.replace(final_path, tmp_path)

        try:
            new_metrics = evaluate_ls_model(tmp_path, window_size=window, lookback_n=args.eval_lookback)
            logger.info(f"[NOVO]  13+: {new_metrics['taxa_13']:.2%} | 14+: {new_metrics['taxa_14']:.2%} "
                        f"| 13/14 puro: {new_metrics['taxa_13_14_puro']:.2%}")
        except Exception as e:
            logger.error(f"Falha ao avaliar o novo modelo treinado: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return

        # política de promoção
        promove = (not modelo_existe) or (
            current_metrics and (
                new_metrics["taxa_13"] >= current_metrics["taxa_13"] or
                new_metrics["taxa_14"] >= current_metrics["taxa_14"]
            )
        )
        if promove:
            logger.info(f"Promovendo novo modelo {model_name}...")
            atomic_promote(tmp_path, final_path, bak_path)
            met = new_metrics
        else:
            logger.info("Mantendo modelo anterior.")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            met = current_metrics if current_metrics else new_metrics
    else:
        logger.info("Sem re-treino. Reavaliando (ou usando) métricas atuais.")
        met = current_metrics

    # 5) PERSISTE STATUS
    try:
        payload = {
            "modelo": model_name,
            "versao": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            "treinado_ate_concurso": latest_concurso,
            "window_size": args.window_size,
            "last_n": args.last_n,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "eval_lookback": args.eval_lookback,
            "taxa_13": float(met["taxa_13"]) if met else None,
            "taxa_14": float(met["taxa_14"]) if met else None,
            "taxa_13_ou_14": float(met["taxa_13_14_puro"]) if met else None,
            "hits_histograma": json.dumps(met["histograma"]) if met else None,
            "saved_path": os.path.join(args.models_dir, f"modelo_{model_name}.h5")
        }
        save_status_to_db(payload)
        logger.info("Status persistido em 'modelo_status'.")
    except Exception as e:
        logger.warning(f"Falha ao salvar status no banco: {e}")

# --------------------------------- CLI ---------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, choices=["ls14", "ls14pp"], help="Modelo a treinar/avaliar")
    p.add_argument("--models_dir", type=str, default=".", help="Diretório onde salva/usa modelo")
    p.add_argument("--last_n", type=int, default=500)
    p.add_argument("--window_size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_weight", type=float, default=4.0)
    p.add_argument("--eval_lookback", type=int, default=200)
    p.add_argument("--min_new_draws", type=int, default=8)
    p.add_argument("--thresh_13", type=float, default=0.75)
    p.add_argument("--thresh_14", type=float, default=0.20)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        pipeline_ls_model(args)
        logger.info("Pipeline concluído.")
    except Exception as exc:
        logger.exception(f"Pipeline falhou: {exc}")
        raise
