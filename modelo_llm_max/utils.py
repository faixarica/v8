# -*- coding: utf-8 -*-
"""
utils.py  |  Funções auxiliares do projeto FaixaBet
Autor: Carlos (FaixaBet)
Data: 2025-10-28

Contém rotinas genéricas usadas pelos scripts de avaliação e geração,
incluindo a simulação de acertos (simulate_hits) para medir eficiência
dos modelos LSTM/LLM (LS14, LS15, LS16, etc.).
"""

import numpy as np

def simulate_hits(model, rows, n_sim=500):
    """
    Simula palpites e mede desempenho do modelo.
    
    Parâmetros:
        model : keras.Model
            Modelo carregado via load_model()
        rows : np.ndarray
            Dados reais (historico de concursos ou features normalizadas)
        n_sim : int
            Quantas simulações realizar (default=500)
    
    Retorna:
        mean_hits : float  → média de acertos
        acc_13 : float     → % de palpites com >=13 acertos
        acc_14 : float     → % de palpites com >=14 acertos
    """
    hits = []

    for _ in range(n_sim):
        # Entrada aleatória do conjunto de dados
        x = np.random.choice(len(rows) - 1)
        sample = rows[x].reshape(1, -1)

        # Predição do modelo
        pred = model.predict(sample, verbose=0)[0]

        # Seleciona as 15 dezenas mais prováveis
        predicted = np.argsort(pred)[-15:]

        # Define o alvo (números reais do próximo concurso)
        target = np.where(rows[x + 1] == 1)[0]

        # Conta acertos
        n_hits = len(set(predicted) & set(target))
        hits.append(n_hits)

    hits = np.array(hits)

    # Calcula métricas
    mean_hits = np.mean(hits)
    acc_13 = np.mean(hits >= 13) * 100
    acc_14 = np.mean(hits >= 14) * 100

    return mean_hits, acc_13, acc_14


def format_metrics(mean_hits, acc_13, acc_14):
    """Formata métricas em string amigável para logs ou print."""
    return (
        f"🎯 Média de acertos: {mean_hits:.2f}\n"
        f"✅ >=13 acertos: {acc_13:.1f}%\n"
        f"⭐ >=14 acertos: {acc_14:.1f}%"
    )
