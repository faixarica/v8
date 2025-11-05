# =======================================================================================
#  train_llm_loteria.py  (versão 2.4.0 - 04/11/2025)
#    (versão 2.4.1 - 05/11/2025)

# ---------------------------------------------------------------------------------------
#  Autor:  FaixaBet (franciscof)
#  Objetivo: Treinar modelos LS14 (estatístico+neural) e LS15 (neural puro)
#  Estrutura: gera recent/mid/global com arquitetura Bidirectional LSTM
#  Ajuste: cria fallback automático em .h5 (compatível com TensorFlow 2.12/3.11)
# =======================================================================================

import os,sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from build_datasets import build_dataset_ls14pp, build_dataset_ls15pp

if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.append(os.getcwd())


# ---------------------------------------------------------------------
# Modelo LSTM principal
# ---------------------------------------------------------------------
def build_lstm_model(window=50):
    model = Sequential([
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
                        help="Diretório de saída para os modelos")
    parser.add_argument("--last_n", type=int, default=None,
                        help="Limitar aos últimos N concursos (ex: 200, 500, 1000)")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Util: detecta subset pelo caminho de saída
# ---------------------------------------------------------------------
def detect_subset_from_out(out_dir: str) -> str:
    leaf = os.path.basename(os.path.normpath(out_dir)).lower()
    if leaf in {"recent", "mid", "global"}:
        return leaf
    return "global"


# ---------------------------------------------------------------------
# Util: salva modelo compatível (.keras + .h5)
# ---------------------------------------------------------------------
def save_model_compatible(model, out_path):
    """
    Salva o modelo em .keras (moderno) e .h5 (fallback compatível).
    """
    try:
        model.save(out_path)
        print(f"✅ Modelo salvo em: {out_path}")
    except Exception as e:
        print(f"[!] Falha ao salvar .keras: {e}")

    # cria fallback .h5
    alt_path = out_path.replace(".keras", ".h5")
    try:
        model.save(alt_path)
        print(f"💾 Fallback salvo em: {alt_path}")
    except Exception as e:
        print(f"[!] Falha ao salvar fallback .h5: {e}")


# ---------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    subset = detect_subset_from_out(args.out)
    print("===================================================")
    print("🚀 Iniciando geração de modelos LS14 e LS15...")
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
        save_model_compatible(model14, out_path14)
        print(f"✅ [LS14] Concluído!\n")

    # ---------------- LS15 ----------------
    if args.model in ["ls15", "both"]:
        print("🎯 [LS15] Gerando dataset...")
        X15, y15 = build_dataset_ls15pp(last_n=args.last_n, window=args.window)
        model15 = build_lstm_model(args.window)

        print("🧠 [LS15] Treinando modelo...")
        model15.fit(X15, y15, epochs=args.epochs, batch_size=args.batch, verbose=1)

        out_path15 = os.path.join(args.out, f"{subset}_ls15pp_final.keras")
        save_model_compatible(model15, out_path15)
        print(f"✅ [LS15] Concluído!\n")

    print("🏁 Todos os treinamentos concluídos com sucesso!\n")


# ---------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
