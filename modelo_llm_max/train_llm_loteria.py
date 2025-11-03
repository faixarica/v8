# =======================================================================================
#  train_llm_loteria.py  (versão 2.3.2 - 28/10/2025)
# ---------------------------------------------------------------------------------------
#  Autor:  FaixaBet (Carlos)
#  Objetivo: Treinar modelos LS14 (estatístico+neural) e LS15 (neural puro)
#  Estrutura: gera recent/mid/global com arquitetura Bidirectional LSTM
#  Ajuste: nomeia arquivos conforme subset detectado pelo --out (recent/mid/global)
# =======================================================================================

import os, argparse, numpy as np, tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from build_datasets import build_dataset_ls14pp, build_dataset_ls15pp

# ---------------------------------------------------------------------
# Modelo com maior capacidade
# ---------------------------------------------------------------------
def build_lstm_model(window=50):
    model = tf.keras.Sequential([
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.3), input_shape=(window, 25)),
        LSTM(128, dropout=0.2),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(25, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------------------------------
# Parser de argumentos
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Treinador LSTM para LS14 e LS15 (FaixaBet)")
    parser.add_argument("--model", type=str, default="both",
                        help="Modelo a treinar: ls14, ls15 ou both")
    parser.add_argument("--epochs", type=int, default=64,
                        help="Número de épocas de treinamento")
    parser.add_argument("--batch", type=int, default=32,
                        help="Tamanho do batch")
    parser.add_argument("--window", type=int, default=50,
                        help="Janela temporal (n concursos)")
    parser.add_argument("--out", type=str, default="./models/prod",
                        help="Diretório de saída para os .keras")
    parser.add_argument("--last_n", type=int, default=None,
                        help="Limitar aos últimos N concursos (ex: 200, 500, 1000)")
    return parser.parse_args()

# ---------------------------------------------------------------------
# Util: detecta subset pelo caminho de saída
# ---------------------------------------------------------------------
def detect_subset_from_out(out_dir: str) -> str:
    """
    Deduz o subset pelo nome da pasta final do --out: recent/mid/global.
    Fallback: 'global' (mantém compatibilidade com fluxos antigos).
    """
    leaf = os.path.basename(os.path.normpath(out_dir)).lower()
    if leaf in {"recent", "mid", "global"}:
        return leaf
    return "global"

# ---------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    subset = detect_subset_from_out(args.out)  # <-- chave: recent/mid/global
    print("===================================================")
    print(f"🚀 Iniciando geração de modelos LS14 e LS15...")
    print(f"Parâmetros: epochs={args.epochs}, batch={args.batch}, window={args.window}, last_n={args.last_n}")
    print(f"📂 Out: {args.out}  |  Subset detectado: {subset}")
    print("===================================================\n")

    # ---------------- LS14 ----------------
    if args.model in ["ls14", "both"]:
        print("🎯 [LS14] Gerando dataset...")
        X14, y14 = build_dataset_ls14pp(last_n=args.last_n, window=args.window)

        model14 = build_lstm_model(args.window)
        print("🧠 [LS14] Treinando modelo...")
        model14.fit(X14, y14, epochs=args.epochs, batch_size=args.batch, verbose=1)

        out_path14 = os.path.join(args.out, f"{subset}_ls14pp_final.keras")
        model14.save(out_path14)
        print(f"✅ [LS14] Modelo salvo em: {out_path14}\n")

    # ---------------- LS15 ----------------
    if args.model in ["ls15", "both"]:
        print("🎯 [LS15] Gerando dataset...")
        X15, y15 = build_dataset_ls15pp(last_n=args.last_n, window=args.window)

        model15 = build_lstm_model(args.window)
        print("🧠 [LS15] Treinando modelo...")
        model15.fit(X15, y15, epochs=args.epochs, batch_size=args.batch, verbose=1)

        out_path15 = os.path.join(args.out, f"{subset}_ls15pp_final.keras")
        model15.save(out_path15)
        print(f"✅ [LS15] Modelo salvo em: {out_path15}\n")

    print("🏁 Todos os treinamentos concluídos com sucesso!\n")

if __name__ == "__main__":
    main()
