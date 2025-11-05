# sampler.py — Diversificador de palpites (v1.0)
# Gera palpites a partir de probabilidades por dezena, aplicando:
# - Diversidade entre palpites
# - Restrições de soma, pares e faixa
# - Controle de temperatura

import numpy as np

def _ajustar_temperatura(p, temperatura: float):
    """Aplica softmax com temperatura."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p = np.log(p)
    p = p / temperatura
    p = np.exp(p - np.max(p))
    return p / p.sum()

def _validar_palpite(palpite):
    """Restrições básicas."""
    soma = np.sum(palpite)
    pares = np.sum(palpite % 2 == 0)
    return (150 <= soma <= 230) and (5 <= pares <= 10)

def gerar_palpites(probabilidades, n_palpites=10, temperatura=1.0, diversify_strength=0.85):
    """
    Gera K palpites de 15 dezenas com diversidade e restrições.
    probabilidades: vetor (25,) com p(d=1..25)
    """
    palpites = []
    uso = np.zeros(25, dtype=int)  # frequência de uso

    for k in range(n_palpites):
        # penaliza dezenas já usadas
        penalizacao = diversify_strength ** uso
        p_corrigida = probabilidades * penalizacao
        p_corrigida = _ajustar_temperatura(p_corrigida, temperatura)

        # amostra até encontrar um válido
        for _ in range(100):
            cand = np.random.choice(np.arange(1, 26), size=15, replace=False, p=p_corrigida)
            cand.sort()
            if _validar_palpite(cand):
                palpites.append(cand)
                uso[cand - 1] += 1
                break
        else:
            # se não achou válido
            cand = np.random.choice(np.arange(1, 26), size=15, replace=False, p=p_corrigida)
            cand.sort()
            palpites.append(cand)
            uso[cand - 1] += 1

    return np.array(palpites, dtype=int)
