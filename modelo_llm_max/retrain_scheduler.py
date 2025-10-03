# retrain_scheduler.py
#python retrain_scheduler.py --models ls15pp ls14pp --models_dir ./models

import os
import json
import logging
from utils_ls_models import fetch_history
from train_llm_loteria import train_model

logging.basicConfig(level=logging.INFO)

def get_last_concurso_from_db():
    rows = fetch_history(last_n=1)
    return rows[0][0] if rows else 0

def get_last_concurso_from_metadata(model_name, models_dir="./models"):
    meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            return meta.get("last_concurso_trained", 0)
    return 0

def retrain_if_needed(model_name, models_dir="./models", min_new_draws=1):
    last_db = get_last_concurso_from_db()
    last_model = get_last_concurso_from_metadata(model_name, models_dir)
    if last_db > last_model + min_new_draws:
        logging.info(f"Novos sorteios detectados para {model_name} ({last_model} → {last_db}). Re-treinando...")
        train_model(model_name, last_n=1500, window=50, epochs=80, batch=32, out_dir=models_dir)
    else:
        logging.info(f"Nenhum novo sorteio relevante para {model_name}. Último treino: {last_model}, DB: {last_db}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["ls15pp", "ls14pp"])
    parser.add_argument("--models_dir", default="./models")
    args = parser.parse_args()

    for model in args.models:
        retrain_if_needed(model, args.models_dir)