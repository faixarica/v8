# pipeline_data.py
import numpy as np
from train_ls_models import fetch_history, build_dataset_ls14pp, build_dataset_ls15pp

def get_training_data(model_type="both", last_n=500, window=50):
    """
    Retorna X e y prontos para treino dos modelos LS14++ e LS15++
    
    Parâmetros:
        model_type (str): "ls14pp", "ls15pp" ou "both"
        last_n (int): quantidade de concursos mais recentes a usar
        window (int): tamanho da janela de histórico
    
    Retorna:
        dict:
            "LS14++": (X_lista, y) -> multi-input
            "LS15++": (X_array, y) -> single-input
    """
    output = {}

    # LS14++ (multi-input)
    if model_type in ("ls14pp", "both"):
        rows, rep_map = fetch_history(last_n=last_n, include_repeats=True)
        X_seq, X_hist, X_freq, X_atraso, X_global, y_ls14pp = build_dataset_ls14pp(rows, rep_map, window)
        X_ls14pp = [X_seq, X_hist, X_freq, X_atraso, X_global]
        output["LS14++"] = (X_ls14pp, y_ls14pp)

    # LS15++ (single-input)
    if model_type in ("ls15pp", "both"):
        rows = fetch_history(last_n=last_n)
        X_ls15pp, X_freq, X_atraso, X_global, y_ls15pp = build_dataset_ls15pp(rows, window)
        # LS15++ usa apenas X_seq (X_ls15pp)
        output["LS15++"] = (X_ls15pp, y_ls15pp)

    return output
