# build_datasets.py
import numpy as np
#from modelo_llm_max.utils_ls_models import to_binary
from utils_ls_models import to_binary

def build_dataset_ls15pp(last_n=None, window=50):
    """
    Constrói o dataset LS15++ (Gold) — modelo puramente neural baseado em sequência.
    Aceita filtro 'last_n' para usar apenas os últimos N concursos.

    Parâmetros:
        last_n (int): Quantidade de concursos recentes a usar (ex: 500, 1000, 1500).
        window (int): Tamanho da janela temporal para o modelo LSTM.

    Retorna:
        X (np.ndarray): Dados de entrada (sequências históricas).
        y (np.ndarray): Labels binários (dezenas sorteadas no próximo concurso).
    """

    print(f"[build_dataset_ls15pp] Construindo dataset (last_n={last_n}, window={window})...")

    # =========================================================
    # 🔹 Carrega dados reais
    # =========================================================
    try:
        rows = np.load("./dados/rows.npy", allow_pickle=True)
        print(f"[build_dataset_ls15pp] Dados carregados de rows.npy ({len(rows)} registros).")
    except Exception as e:
        print(f"[build_dataset_ls15pp] Falha ao carregar rows.npy: {e}")
        try:
            df = pd.read_csv("loteria.csv", sep=",")
            rows = df[[f"n{i}" for i in range(1, 16)]].values
            print(f"[build_dataset_ls15pp] Dados carregados de loteria.csv ({len(rows)} registros).")
        except Exception as e2:
            raise RuntimeError(f"Erro ao carregar dados base: {e2}")

    # =========================================================
    # 🔹 Filtro last_n (se especificado)
    # =========================================================
    if last_n is not None and last_n < len(rows):
        rows = rows[-last_n:]
        print(f"[build_dataset_ls15pp] Filtrando últimos {last_n} concursos (total {len(rows)}).")

    # =========================================================
    # 🔹 Conversão para vetores binários (25 dezenas)
    # =========================================================
    def to_binary(draw):
        v = np.zeros(25, dtype=int)
        for d in draw:
            if 1 <= int(d) <= 25:
                v[int(d) - 1] = 1
        return v

    bin_rows = np.array([to_binary(r) for r in rows])

    # =========================================================
    # 🔹 Criação das janelas (input X e target y)
    # =========================================================
    X, y = [], []
    for i in range(len(bin_rows) - window):
        X.append(bin_rows[i:i + window])
        y.append(bin_rows[i + window])

    X = np.array(X)
    y = np.array(y)

    # =========================================================
    # 🔹 Escalonamento (0–1)
    # =========================================================
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, 25)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    print(f"[build_dataset_ls15pp] Dataset pronto: X={X_scaled.shape}, y={y.shape}")
    return X_scaled, y

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def build_dataset_ls14pp(last_n=None, window=50):
    """
    Constrói o dataset LS14++ (Silver) com base nos dados reais de sorteios.
    Aceita filtro 'last_n' para usar apenas os últimos N concursos.

    Parâmetros:
        last_n (int): Quantidade de concursos recentes a usar (ex: 500, 1000, 1500).
        window (int): Tamanho da janela temporal para o modelo LSTM.

    Retorna:
        X (np.ndarray): Dados de entrada (janelas de sorteios).
        y (np.ndarray): Labels binários (presença de dezenas no próximo concurso).
    """

    print(f"[build_dataset_ls14pp] Construindo dataset (last_n={last_n}, window={window})...")

    # =========================================================
    # 🔹 Carrega dados reais (rows.npy ou CSV de backup)
    # =========================================================
    try:
        rows = np.load("./dados/rows.npy", allow_pickle=True)
        print(f"[build_dataset_ls14pp] Dados carregados de rows.npy ({len(rows)} registros).")
    except Exception as e:
        print(f"[build_dataset_ls14pp] Falha ao carregar rows.npy: {e}")
        try:
            df = pd.read_csv("loteria.csv", sep=",")
            rows = df[[f"n{i}" for i in range(1, 16)]].values
            print(f"[build_dataset_ls14pp] Dados carregados de loteria.csv ({len(rows)} registros).")
        except Exception as e2:
            raise RuntimeError(f"Erro ao carregar dados base: {e2}")

    # =========================================================
    # 🔹 Aplica filtro last_n se fornecido
    # =========================================================
    if last_n is not None and last_n < len(rows):
        rows = rows[-last_n:]
        print(f"[build_dataset_ls14pp] Filtrando últimos {last_n} concursos (total {len(rows)}).")

    # =========================================================
    # 🔹 Normalização binária (dezenas 1–25)
    # =========================================================
    def to_binary(draw):
        v = np.zeros(25, dtype=int)
        for d in draw:
            if 1 <= int(d) <= 25:
                v[int(d) - 1] = 1
        return v

    bin_rows = np.array([to_binary(r) for r in rows])

    # =========================================================
    # 🔹 Gera janelas (window) para treinamento
    # =========================================================
    X, y = [], []
    for i in range(len(bin_rows) - window):
        X.append(bin_rows[i:i + window])
        y.append(bin_rows[i + window])

    X = np.array(X)
    y = np.array(y)

    # =========================================================
    # 🔹 Escalonamento (ajuste opcional de amplitude)
    # =========================================================
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, 25)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    print(f"[build_dataset_ls14pp] Dataset pronto: X={X_scaled.shape}, y={y.shape}")
    return X_scaled, y
