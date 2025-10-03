# ensemble.py - 
#Carrega os 3 modelos treinados (recent, mid, global).
#Recebe os dados de entrada (últimos concursos).
#Faz a previsão de probabilidades em cada modelo.
#Faz a média das probabilidades (ensemble).
#Seleciona as 15 dezenas mais prováveis (ou outro K configurável).
#Exporta os palpites em um arquivo .txt (um jogo por linha).

import argparse
import numpy as np
import tensorflow as tf
from utils_ls_models import to_binary
from build_datasets import build_dataset_ls15pp

# -------------------------
# Função para carregar modelos
# -------------------------
def load_models():
    models = {}
    models["recent"] = tf.keras.models.load_model("./models/recent/best_hits.keras", compile=False)
    models["mid"]    = tf.keras.models.load_model("./models/mid/best_hits.keras", compile=False)
    models["global"] = tf.keras.models.load_model("./models/global/best_hits.keras", compile=False)
    return models

# -------------------------
# Função para ensemble (média das probabilidades)
# -------------------------
def ensemble_predict(models, X):
    preds = []
    for m in models.values():
        p = m.predict(X, verbose=0)
        if isinstance(p, (list, tuple)):  # caso o modelo retorne múltiplas saídas
            p = p[0]
        preds.append(p)
    return np.mean(preds, axis=0)

# -------------------------
# Função para gerar palpites (top-k)
# -------------------------
def generate_palpites(probs, k=15, n_jogos=5):
    palpites = []
    for _ in range(n_jogos):
        # top-k probabilidades
        idx = np.argsort(-probs[0])[:k]
        jogo = np.sort(idx + 1)  # +1 porque dezenas vão de 1 a 25
        palpites.append(jogo)
    return palpites

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jogos", type=int, default=5, help="Número de palpites a gerar")
    parser.add_argument("--k", type=int, default=15, help="Quantidade de dezenas por jogo")
    parser.add_argument("--out", type=str, default="./palpites_ensemble.txt", help="Arquivo de saída")
    args = parser.parse_args()

    # 1) Carregar modelos
    models = load_models()

    # 2) Preparar entrada (últimos concursos)
    # Aqui pegamos os últimos dados disponíveis para prever
    X, _ = build_dataset_ls15pp(last_n=200)  # usa últimos 200 concursos como base
    X_input = [X[0][-1:]] if isinstance(X, list) else X[-1:].reshape(1, -1)

    # 3) Predição média
    probs = ensemble_predict(models, X_input)

    # 4) Gerar palpites
    palpites = generate_palpites(probs, k=args.k, n_jogos=args.n_jogos)

    # 5) Salvar palpites em arquivo
    with open(args.out, "w") as f:
        for jogo in palpites:
            f.write(",".join(map(str, jogo)) + "\n")

    print(f"✅ {args.n_jogos} palpites salvos em {args.out}")
    