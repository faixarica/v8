import numpy as np
import itertools
import random

def gerar_palpites_optimizado(model, X_test, n_games=500, top_k=18, select_k=15):
    """
    Gera palpites diversificados usando top_k maiores probabilidades do modelo,
    selecionando select_k dezenas para cada palpite de forma aleatória,
    mas favorecendo dezenas com maiores chances.
    """
    probs = model.predict(X_test, verbose=0)  # shape (N, 25)
    palpites = []

    for p in probs:
        # Pegando indices das top_k maiores probabilidades
        top_indices = np.argsort(p)[-top_k:][::-1] + 1  # converte para dezenas 1-25
        
        # Gerar variações para diversificação
        if len(top_indices) > select_k:
            # Seleciona 15 dezenas aleatoriamente dentro do top_k
            palpite = sorted(np.random.choice(top_indices, size=select_k, replace=False))
        else:
            palpite = sorted(top_indices[:select_k])
        palpites.append(palpite)

    # Completar até n_games com combinações diversas
    while len(palpites) < n_games:
        sample = random.choice(palpites)
        shuffled = sample.copy()
        random.shuffle(shuffled)
        palpites.append(sorted(shuffled))

    return palpites[:n_games]


def avaliar_acertos(palpites, y_test):
    """
    Compara lista de palpites (listas de 15 dezenas) com y_test (one-hot 25 posições).
    Retorna lista de hits e relatório de >=11..15 acertos.
    """
    hits_list = []
    for i, palpite in enumerate(palpites):
        y_true = np.where(y_test[i] == 1)[0] + 1  # converte para dezenas 1-25
        hits = len(set(palpite) & set(y_true))
        hits_list.append(hits)

    # Estatísticas
    stats = {}
    for k in range(11, 16):
        count = sum(h >= k for h in hits_list)
        perc = count / len(hits_list) * 100
        stats[k] = {"count": count, "perc": perc}

    return hits_list, stats
    