# backtest.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train_ls_models_v3 import build_dataset_ls14pp, build_dataset_ls15pp, fetch_history
from palpites_optimizados import gerar_palpites_optimizado, avaliar_acertos
    




# ------------------------------
# Função para gerar palpites com exploração
# ------------------------------
def generate_games_from_probs_with_exploration(probs, k=15, n_games=10, temperature=1.5):
    probs = np.array(probs, dtype=np.float32)
    exp_probs = np.exp(np.log(probs + 1e-8) / temperature)
    probs_t = exp_probs / np.sum(exp_probs)
    games = []
    for _ in range(n_games):
        game = np.random.choice(np.arange(25), size=k, replace=False, p=probs_t)
        games.append(np.sort(game + 1))
    return games

# ------------------------------
# Avaliação de acertos
# ------------------------------
def evaluate(model, X_test, y_test, k=15, n_games=100, temperature=1.5):
    hits_counter = {11:0, 12:0, 13:0, 14:0, 15:0}
    hits_distribution = []

    for i in range(len(y_test)):
        X_input = [X_test[j][i:i+1] for j in range(len(X_test))]
        probs = model.predict(X_input, verbose=0)[0]
        games = generate_games_from_probs_with_exploration(probs, k=k, n_games=n_games, temperature=temperature)
        y_true = np.where(y_test[i]==1)[0] + 1
        
        best_hit = 0
        for game in games:
            hit = len(set(game) & set(y_true))
            if hit > best_hit:
                best_hit = hit
        hits_distribution.append(best_hit)
        
        for t in hits_counter:
            if best_hit >= t:
                hits_counter[t] += 1
    
    n_samples = len(y_test)
    return hits_counter, hits_distribution, n_samples

# ------------------------------
# Preparar dados
# ------------------------------
def prepare_data(model_type, last_n=1500, window=50):
    rows = fetch_history(last_n=last_n)
    if model_type == "ls14pp":
        X_seq, X_hist, X_freq, X_atraso, X_global, y = build_dataset_ls14pp(rows, window)
        X_test = [X_seq, X_hist, X_freq, X_atraso, X_global]
    else:
        X_seq, X_freq, X_atraso, X_global, y = build_dataset_ls15pp(rows, window)
        X_test = [X_seq, X_freq, X_atraso, X_global]
    return X_test, y

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["ls14pp", "ls15pp"])
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--last_n", type=int, default=1500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--n_games", type=int, default=100)
    args = parser.parse_args()

    print(f"Carregando modelo de {args.load} ...")
    model = load_model(args.load)
    print(f"Rodando backtest para {args.model.upper()}...")

    X_test, y_test = prepare_data(args.model, last_n=args.last_n, window=args.window)
    hits_counter, hits_distribution, n_samples = evaluate(
        model, X_test, y_test, k=15, n_games=args.n_games, temperature=args.temperature
    )

    print(f"=== RESULTADOS BACKTEST {args.model.upper()} ===")
    print(f"Concursos testados: {n_samples}")
    for t in sorted(hits_counter):
        print(f">= {t} acertos: {hits_counter[t]/n_samples*100:.2f}%")

# Gera palpites otimizados
palpites_otimizados = gerar_palpites_optimizado(model, X_test, n_games=500, top_k=18, select_k=15)

# Avalia acertos
hits, stats = avaliar_acertos(palpites_otimizados, y_test)

# Exibe relatório
print("=== RELATÓRIO DE ACERTOS OTIMIZADOS ===")
for k in range(11, 16):
    s = stats[k]
    print(f">= {k} acertos: {s['perc']:.2f}% ({s['count']}/{len(hits)})")

# Gráfico de distribuição de acertos
plt.hist(hits, bins=range(0, 16), edgecolor="black")
plt.title("Distribuição de Acertos - Palpites Otimizados")
plt.xlabel("Número de acertos")
plt.ylabel("Quantidade de palpites")
plt.show()