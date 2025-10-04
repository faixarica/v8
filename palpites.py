# palpites.py
# Versão consolidada com Ensemble (mantém compatibilidade com chamadas existentes)
# Substitua seu palpites.py atual por este arquivo (faça backup antes).

# Configure no servidor de produção:
#  export DEBUG_MODE=0 -  para na mostrar warnings na UI (padrão)
# -------------------- [1] IMPORTS --------------------
import streamlit as st
import random
from datetime import datetime
import pyperclip
import os
import glob
import numpy as np
from sqlalchemy import text
from db import Session
from tensorflow.keras.models import load_model as tf_load_model
from modelo_llm_max.utils_ls_models import to_binary
import sys
import logging
import re
import tensorflow as tf
from tensorflow import keras as tf_keras
# módulos do projeto
try:
    from modelo_llm_max.build_datasets import build_dataset_ls15pp, build_dataset_ls14pp
except Exception:
    build_dataset_ls15pp = None
    build_dataset_ls14pp = None
    st.warning("Aviso: módulo modelo_llm_max.build_datasets não disponível — imports build_dataset_* definidos como None.")


# -------------------- [2] CONFIGS --------------------
# tente inferir corretamente a base do projeto mesmo quando __file__ não existir
try:
    _base_file = __file__
except NameError:
    _base_file = sys.argv[0] if sys.argv and sys.argv[0] else os.getcwd()

BASE_DIR = os.environ.get("FAIXABET_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(_base_file))))
# Permite sobrescrever a pasta de modelos via variável de ambiente caso queira apontar manualmente
MODELS_DIR = os.environ.get("FAIXABET_MODELS_DIR", os.path.join(BASE_DIR, "models"))

# tenta importar TFSMLayer com fallback
try:
    from keras.layers import TFSMLayer  # preferível quando keras separado está instalado
except Exception:
    try:
        from tensorflow.keras.layers import TFSMLayer  # fallback
    except Exception:
        TFSMLayer = None  # se não existir, teremos warning e pediremos conversão

# --------- MODO ADMIN / DEBUG ---------
# Defina a variável de ambiente DEBUG_MODE=1 para ver warnings na UI
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

def _log_warn(msg):
    """Mostra warning na UI apenas se DEBUG_MODE estiver ativo, senão manda para log/console."""
    if DEBUG_MODE:
        st.warning(msg)
    else:
        logging.warning(msg)

def _log_info(msg):
    """Mostra info na UI apenas se DEBUG_MODE estiver ativo, senão manda para log/console."""
    if DEBUG_MODE:
        st.info(msg)
    else:
        logging.info(msg)

def _log_info(msg):
    """Mostra info na UI apenas se DEBUG_MODE estiver ativo, senão manda para log/console."""
    if DEBUG_MODE:
        st.info(msg)
    else:
        logging.info(msg)

# ---------------- ENSEMBLE: carregamento e predição ----------------
# ---------- INÍCIO DO BLOCO A COLAR / SUBSTITUIR ----------

# ---------------- AJUSTE PRODUÇÃO ----------------

def _model_group_search_roots():
    """
    Produção: considera apenas ./models/prod/
    """
    roots = []
    prod_path = os.path.join(MODELS_DIR, "prod")
    if os.path.isdir(prod_path):
        # cria subpastas virtuais "recent/mid/global" só para manter compatibilidade
        for sub in ("recent", "mid", "global"):
            roots.append(prod_path)
    return roots

def _detect_group_and_expected_from_path(p):
    """
    Detecta grupo a partir do prefixo do nome no arquivo em ./models/prod/
    Exemplo: recent_ls15pp_final.keras → group=recent
    """
    low = os.path.basename(p).lower()
    if low.startswith("recent_"):
        return "recent", 500
    if low.startswith("mid_"):
        return "mid", 1000
    if low.startswith("global_"):
        return "global", 1550
    return "unknown", None

def _ensure_window_list(ultimos_full, expected):
    """
    Garante que retorne uma lista com exatamente 'expected' jogos (cada jogo lista com 15 números).
    Se len(ultimos_full) >= expected -> retorna últimos expected
    Se len(ultimos_full) < expected  -> preenche repetindo o jogo mais antigo disponível (fallback)
    """
    cur = len(ultimos_full)
    if cur >= expected:
        return ultimos_full[-expected:]
    if cur == 0:
        # fallback: cria expected jogos vazios (não ideal, mas evita crash)
        pad = [[0]*15 for _ in range(expected)]
        return pad
    pad_count = expected - cur
    pad = [ultimos_full[0]] * pad_count
    return pad + ultimos_full

def infer_expected_seq_from_loaded_model(loaded):
    """
    Tenta inferir o tamanho da janela (seq_len) a partir de model.inputs.
    Retorna int ou None.
    """
    try:
        ins = getattr(loaded, "inputs", None)
        if not ins:
            return None
        for inp in ins:
            try:
                shape = getattr(inp, "shape", None)
                if shape is None:
                    continue
                # transform shape em lista de ints/None
                dims = []
                for d in shape:
                    try:
                        dims.append(int(d))
                    except Exception:
                        dims.append(None)
                # busca um input do tipo (None, TIME, 25)
                if len(dims) == 3 and dims[-1] == 25:
                    return dims[1]  # pode ser None
            except Exception:
                continue
    except Exception:
        pass
    return None

# Mapas de acesso por plano (ajuste conforme regras do seu negócio)
_PLAN_TO_GROUPS = {
    "Free": ["recent"],
    "Silver": ["recent", "mid"],
    "Gold": ["recent", "mid", "global"],
    # alias que você mencionou:
    "Plano Pago X": ["recent", "mid", "global"]
}

def _groups_allowed_for_plan(nome_plano):
    return _PLAN_TO_GROUPS.get(nome_plano, ["recent", "mid", "global"])

def _prepare_inputs_for_model_meta(meta, ultimos_full):
    """
    Dado meta (com expected_seq_len, n_inputs, model), gera seq_bin/freq/atraso/global
    apropriados para esse modelo (cortando/paddando ultimos_full conforme necessário)
    e retorna X_inputs prontos (array ou lista de arrays).
    """
    expected = meta.get("expected_seq_len", None)
    if expected is None:
        # fallback: use full history
        window_list = ultimos_full
    else:
        window_list = _ensure_window_list(ultimos_full, expected)

    seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(window_list)
    # seq_bin shape (window,25) -> precisamos (1,window,25)
    if meta.get("n_inputs", 1) == 1:
        X_inputs = seq_bin[None, ...].astype(np.float32)
    else:
        X_inputs = [
            seq_bin[None, ...].astype(np.float32),
            freq_vec[None, :].astype(np.float32),
            atraso_vec[None, :].astype(np.float32),
            global_vec[None, :].astype(np.float32)
        ]
    return X_inputs

def _model_paths_for(model_name):
    """
    Busca recursivamente por candidatos de modelos para 'model_name'.
    Retorna lista de paths (arquivos .keras/.h5/.hdf5 ou diretórios SavedModel).
    A busca é case-insensitive e tolerante a variações de nome.
    """
    candidates = []
    model_name_l = model_name.lower()

    # Se MODELS_DIR não existe, tenta a pasta atual também
    search_base = MODELS_DIR if os.path.isdir(MODELS_DIR) else os.getcwd()

    for root, dirs, files in os.walk(search_base):
        # checar diretórios que contenham saved_model.pb (SavedModel) ou cujo nome contenha model_name
        if "saved_model.pb" in files:
            # se a pasta inclui o nome do modelo, adiciona; senão, adiciona como fallback se model_name em path
            if model_name_l in os.path.basename(root).lower() or model_name_l in root.lower():
                if root not in candidates:
                    candidates.append(root)
        for d in dirs:
            dn = d.lower()
            if model_name_l in dn or f"modelo_{model_name_l}" in dn or f"{model_name_l}pp" in dn:
                p = os.path.join(root, d)
                if p not in candidates:
                    candidates.append(p)
        # checar arquivos com extensões comuns
        for f in files:
            fn = f.lower()
            if model_name_l in fn and fn.endswith((".keras", ".h5", ".hdf5", ".zip")):
                p = os.path.join(root, f)
                if p not in candidates:
                    candidates.append(p)

    # Também tente alguns padrões históricos comuns (não necessariamente recursivos)
    fallback_patterns = [
        f"modelo_{model_name}pp_saved",
        f"{model_name}_saved",
        f"modelo_{model_name}pp",
        f"{model_name}pp",
        model_name
    ]
    for pat in fallback_patterns:
        p = os.path.join(search_base, pat)
        if os.path.isdir(p) and p not in candidates:
            candidates.append(p)
        # procurar arquivo .keras direto no root
        for ext in (".keras", ".h5", ".hdf5"):
            fpath = os.path.join(search_base, pat + ext)
            if os.path.isfile(fpath) and fpath not in candidates:
                candidates.append(fpath)

    return candidates

@st.cache_resource
def _cached_load_ensemble_models(model_name):
    """
    Versão pura e cacheada que carrega modelos (sem usar st.*).
    Retorna lista de metas (dicts) com: model, path, group, expected_seq_len, n_inputs
    NOTA: não deve usar st.* (somente logging).
    """
    metas = []
    candidates = _model_paths_for(model_name)

    # varredura adicional em subpastas 'recent/mid/global'
    for root in _model_group_search_roots():
        for r, dirs, files in os.walk(root):
            for f in files:
                lf = f.lower()
                if model_name.lower() in lf and lf.endswith((".keras", ".h5", ".hdf5")):
                    p = os.path.join(r, f)
                    if p not in candidates:
                        candidates.append(p)
            for d in dirs:
                if model_name.lower() in d.lower():
                    p = os.path.join(r, d)
                    if p not in candidates:
                        candidates.append(p)

    # último fallback: procurar qualquer arquivo dentro de MODELS_DIR que contenha model_name
    if not candidates and os.path.isdir(MODELS_DIR):
        for r, dirs, files in os.walk(MODELS_DIR):
            for f in files:
                if model_name.lower() in f.lower() and f.lower().endswith((".keras", ".h5", ".hdf5")):
                    p = os.path.join(r, f)
                    if p not in candidates:
                        candidates.append(p)

    logging.info(f"[load_models] candidatos encontrados para '{model_name}': {len(candidates)} (base={MODELS_DIR})")

    for p in candidates:
        group, expected_from_path = _detect_group_and_expected_from_path(p)
        loaded = None
        try:
            # tenta carregar .keras/.h5
            loaded = tf_load_model(p, compile=False)
        except Exception as e_load:
            # Se for diretório SavedModel (contendo saved_model.pb) e tivermos TFSMLayer disponível,
            # tentamos montar um wrapper (apenas quando expected_from_path estiver disponível).
            sm_pb = os.path.join(p, "saved_model.pb")
            if os.path.isdir(p) and os.path.isfile(sm_pb):
                if TFSMLayer is not None and expected_from_path is not None:
                    try:
                        in_seq = tf_keras.Input(shape=(expected_from_path, 25), name="seq")
                        tsl = TFSMLayer(p, call_endpoint="serving_default")
                        out = tsl(in_seq)
                        loaded = tf_keras.Model(inputs=in_seq, outputs=out)
                    except Exception as e_tfs:
                        logging.warning(f"[load_models] Falha ao envolver SavedModel em {p} com TFSMLayer: {e_tfs}")
                        loaded = None
                else:
                    logging.warning(
                        f"[load_models] SavedModel detectado em {p} mas não foi possível criar wrapper automático."
                        " (Converta para .keras/.h5 ou ative TFSMLayer.)"
                    )
                    loaded = None
            else:
                logging.warning(f"[load_models] Falha ao carregar {p}: {e_load}")

        if loaded is not None:
            n_inputs = len(getattr(loaded, "inputs", [])) or 1
            expected_from_model = infer_expected_seq_from_loaded_model(loaded)
            expected = expected_from_model or expected_from_path
            metas.append({
                "model": loaded,
                "path": p,
                "group": group,
                "expected_seq_len": expected,
                "n_inputs": n_inputs
            })
        else:
            logging.warning(f"[load_models] Modelo não carregado (ignorando): {p}")

    if not metas:
        logging.warning(f"[load_models] Nenhum modelo válido carregado para '{model_name}'.")
    else:
        logging.info(f"[load_models] {len(metas)} modelo(s) carregado(s) para '{model_name}'.")

    return metas

def carregar_ensemble_models(model_name):
    """
    Wrapper de UI: mostra spinner durante o carregamento e depois reporta via _log_info/_log_warn
    (que exibe na UI apenas se DEBUG_MODE=True).
    """
    # mostra quantos candidatos foram detectados (apenas para o texto do spinner)
    try:
        cand_count = len(_model_paths_for(model_name))
    except Exception:
        cand_count = 0

    with st.spinner(f"🚴 Carregando modelos '{model_name}' ({cand_count} candidatos encontrados)..."):
        metas = _cached_load_ensemble_models(model_name)

    if not metas:
        _log_warn(f"Nenhum modelo válido carregado para '{model_name}'. Revise formatos e nomes.")
    else:
        _log_info(f"{len(metas)} modelo(s) carregado(s) para '{model_name}'.")

    return metas

@st.cache_resource
def carregar_modelo_ls15_ensemble():
    models = carregar_ensemble_models("ls15")
    if models:
        st.info(f"Ensemble LS15 carregado com {len(models)} modelo(s).")
    else:
        st.error("Nenhum modelo LS15 encontrado no ensemble. Use verificar_modelos() para diagnosticar.")
    return models

@st.cache_resource
def carregar_modelo_ls14_ensemble():
    models = carregar_ensemble_models("ls14")
    if models:
        st.info(f"Ensemble LS14 carregado com {len(models)} modelo(s).")
    else:
        st.error("Nenhum modelo LS14 encontrado no ensemble. Use verificar_modelos() para diagnosticar.")
    return models

def listar_candidatos_modelo(model_name):
    """
    Função de debug: retorna e escreve (no Streamlit) os paths candidatos encontrados.
    """
    cand = _model_paths_for(model_name)
    st.write(f"🔎 Candidatos encontrados para '{model_name}' (base={MODELS_DIR}):")
    if cand:
        for c in cand:
            st.write(" - ", c)
    else:
        st.write(" (nenhum candidato encontrado)")
    return cand

# <<< Fim do bloco de substituição >>>

def ensemble_predict(models_list, X_inputs):
    """
    Recebe lista de modelos (keras) ou lista de metas (dict com chave 'model').
    Retorna média das probabilidades (shape (batch,25)).
    Compatível com ambos os formatos para evitar AttributeError.
    """
    if not models_list:
        raise ValueError("Nenhum modelo carregado para ensemble.")

    preds = []
    for entry in models_list:
        # entry pode ser um modelo Keras ou um dict meta
        model_obj = entry
        if isinstance(entry, dict):
            model_obj = entry.get("model")

        if model_obj is None:
            _log_warn(f"Ignorando entrada inválida no ensemble: {entry}")
            continue

        try:
            p = model_obj.predict(X_inputs, verbose=0)
            if isinstance(p, (list, tuple)):
                p = p[0]
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            preds.append(p)
        except Exception as e:
            # mensagem mais informativa
            name = getattr(model_obj, "name", getattr(model_obj, "__class__", None))
            st.warning(f"Predição falhou para um dos modelos do ensemble ({name}): {e}")

    if not preds:
        raise ValueError("Nenhuma predição válida obtida dos modelos.")

    mean_pred = np.mean(preds, axis=0)
    return mean_pred

# -------------------- [3] UTILITÁRIOS E FEATURES --------------------
def montar_entrada_binaria(ultimos_concursos):
    """
    Transforma lista de últimos concursos (lista de listas com 15 números)
    em array binário shape (window,25) dtype float32.
    """
    arr = np.array([[1.0 if (i+1) in jogo else 0.0 for i in range(25)] for jogo in ultimos_concursos], dtype=np.float32)
    return arr

def apply_temperature(p, T=1.0):
    """
    temperature scaling para vetor p (25,)
    """
    p = np.clip(p, 1e-12, 1.0)
    logits = np.log(p)
    scaled = np.exp(logits / float(T))
    return scaled / scaled.sum()

def gerar_palpite_from_probs(probs, limite=15, reinforce_threshold=0.06, boost_factor=2.0, temperature=1.0, deterministic=False):
    """
    Dado um vetor de probabilidades (25,), aplica temperatura, boost e retorna
    'limite' dezenas. Pode rodar de forma determinística (top-k) ou amostral.
    """
    p = apply_temperature(probs, temperature)
    mask = p > reinforce_threshold
    if mask.any():
        p[mask] = p[mask] * boost_factor
        p = p / p.sum()
    if deterministic:
        idxs = np.argsort(-p)[:limite]
        chosen = np.sort(idxs + 1).tolist()
        return chosen
    else:
        chosen_idxs = np.random.choice(np.arange(25), size=limite, replace=False, p=p)
        return np.sort(chosen_idxs + 1).tolist()

def _calc_features_from_window(ultimos):
    """
    Calcula seq_bin, freq_vec, atraso_vec, global_vec a partir da janela 'ultimos'
    onde ultimos é lista de jogos (cada um com 15 dezenas), do mais antigo para o mais recente.
    Retorna seq_bin (window,25), freq_vec(25,), atraso_vec(25,), global_vec(2,)
    """
    seq_bin = np.array([to_binary(j) for j in ultimos], dtype=np.float32)  # (window,25)
    window = len(ultimos)

    freq_vec = seq_bin.sum(axis=0) / float(window)

    atraso_vec = np.zeros(25, dtype=np.float32)
    for d in range(1, 26):
        atraso = 0
        for jogo in reversed(ultimos):
            atraso += 1
            if d in jogo:
                break
        atraso_vec[d-1] = min(atraso, window) / float(window)

    last = ultimos[-1]
    soma = sum(last) / (25.0 * 15.0)
    pares = sum(1 for x in last if x % 2 == 0) / 15.0
    global_vec = np.array([soma, pares], dtype=np.float32)

    return seq_bin, freq_vec.astype(np.float32), atraso_vec.astype(np.float32), global_vec

# -------------------- [4] FUNÇÕES DE PLANOS / DB --------------------
def verificar_limite_palpites(id_usuario):
    db = Session()
    try:
        resultado = db.execute(text(""" 
            SELECT p.palpites_dia, p.palpites_max, p.nome
            FROM usuarios u
            JOIN planos p ON u.id_plano = p.id
            WHERE u.id = :id
        """), {"id": id_usuario}).fetchone()

        if not resultado:
            return False, "Plano não encontrado", 0

        palpites_dia, limite_mes, nome_plano = resultado

        usados_dia = db.execute(text("""
            SELECT COUNT(*) FROM palpites
            WHERE id_usuario = :id AND DATE(data) = CURRENT_DATE
        """), {"id": id_usuario}).scalar()

        usados_mes = db.execute(text("""
            SELECT COUNT(*) FROM palpites
            WHERE id_usuario = :id AND TO_CHAR(data, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM')
        """), {"id": id_usuario}).scalar()

        if usados_dia >= palpites_dia or usados_mes >= limite_mes:
            return False, nome_plano, 0

        palpites_restantes_mes = limite_mes - usados_mes
        return True, nome_plano, palpites_restantes_mes

    except Exception as e:
        st.error(f"Erro ao verificar limite de palpites: {e}")
        return False, "Erro", 0
    finally:
        db.close()

def obter_limite_dezenas_por_plano(tipo_plano):
    db = Session()
    try:
        resultado = db.execute(text("""
            SELECT palpites_max FROM planos WHERE nome = :nome
        """), {"nome": tipo_plano}).fetchone()
        return resultado[0] if resultado else 15
    except Exception as e:
        st.error(f"Erro ao obter limite de dezenas: {e}")
        return 15
    finally:
        db.close()

def atualizar_contador_palpites(id_usuario):
    db = Session()
    try:
        db.execute(text("""
            UPDATE client_plans
            SET palpites_dia_usado = COALESCE(palpites_dia_usado, 0) + 1
            WHERE id_client = :id
              AND DATE(data_expira_plan) >= CURRENT_DATE
              AND ativo = true
        """), {"id": id_usuario})
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao atualizar contador de palpites: {e}")
    finally:
        db.close()

def salvar_palpite(palpite, modelo):
    db = Session()
    try:
        id_usuario = st.session_state.usuario["id"]
        numeros_str = ','.join(map(str, palpite)) if isinstance(palpite, list) else str(palpite)
        db.execute(text("""
            INSERT INTO palpites (id_usuario, numeros, modelo, data, status)
            VALUES (:id_usuario, :numeros, :modelo, NOW(), 'N')
        """), {
            "id_usuario": id_usuario,
            "numeros": numeros_str,
            "modelo": modelo
        })
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao salvar palpite: {e}")
    finally:
        db.close()

# -------------------- [5] GERADORES SIMPLES --------------------
def gerar_palpite_pares_impares(limite=15):
    num_pares = limite // 2
    num_impares = limite - num_pares
    pares = random.sample(range(2, 26, 2), num_pares)
    impares = random.sample(range(1, 26, 2), num_impares)
    return sorted(pares + impares)

def gerar_palpite_aleatorio(limite=15):
    return sorted(random.sample(range(1, 26), limite))

def gerar_palpite_estatistico(limite=15):
    db = Session()
    try:
        resultados = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
            FROM resultados_oficiais
        """)).fetchall()
        if not resultados:
            return gerar_palpite_aleatorio(limite)

        todos_numeros = [num for row in resultados for num in row]
        frequencia = {num: todos_numeros.count(num) for num in range(1, 26)}
        numeros_ordenados = sorted(frequencia.items(), key=lambda x: x[1], reverse=True)

        top_10 = [num for num, _ in numeros_ordenados[:10]]
        outros = [num for num, _ in numeros_ordenados[10:20]]
        baixa_freq = [num for num, _ in numeros_ordenados[20:]]

        palpite = (
            random.sample(top_10, min(7, len(top_10))) +
            random.sample(outros, min(5, len(outros))) +
            random.sample(baixa_freq, min(3, len(baixa_freq)))
        )
        return sorted(palpite)[:limite]

    except Exception:
        return gerar_palpite_aleatorio(limite)
    finally:
        db.close()

# -------------------- [6] GERADORES ML (LS15 / LS14) - compatíveis --------------------
def gerar_palpite_ls15(limite=15):
    """
    Gera palpite LS15 usando ensemble adaptado aos vários modelos (recent/mid/global).
    Respeita o plano do usuário (se disponível em st.session_state.usuario ou busca no DB).
    """
    # carrega modelos meta
    metas = carregar_ensemble_models("ls15")
    if not metas:
        st.error("Nenhum modelo LS15 encontrado/carregado.")
        return []

    # descobrir o plano do usuário (tenta sessão primeiro)
    nome_plano = None
    try:
        nome_plano = st.session_state.usuario.get("nome_plano") or None
    except Exception:
        nome_plano = None

    if not nome_plano:
        # tenta obter via id_plano no banco se existir
        try:
            id_plano = st.session_state.usuario.get("id_plano")
            if id_plano:
                db = Session()
                r = db.execute(text("SELECT nome FROM planos WHERE id = :id"), {"id": id_plano}).fetchone()
                db.close()
                if r:
                    nome_plano = r[0]
        except Exception:
            nome_plano = None

    allowed_groups = _groups_allowed_for_plan(nome_plano)
    metas = [m for m in metas if m.get("group") in allowed_groups]
    if not metas:
        st.error(f"Nenhum modelo LS15 disponível para seu plano ({nome_plano}).")
        return []

    # busca a maior janela necessária entre os modelos disponíveis
    max_window = max([m.get("expected_seq_len") or 0 for m in metas])
    if not max_window or max_window <= 0:
        # fallback razoável (reduz prob de crash)
        max_window = 1550

    # buscar histórico suficiente no DB
    db = Session()
    try:
        rows = db.execute(text(f"""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY data DESC
            LIMIT :lim
        """), {"lim": int(max_window)}).fetchall()
    finally:
        db.close()

    if not rows:
        st.error("Não há histórico suficiente para gerar palpites LS15.")
        return []

    # ultimos_full: do mais antigo pro mais recente (para _calc_features_from_window)
    ultimos_full = [list(r) for r in reversed(rows)]

    preds = []
    for meta in metas:
        try:
            X = _prepare_inputs_for_model_meta(meta, ultimos_full)
            # chamar predict
            m = meta["model"]
            p = m.predict(X, verbose=0)
            if isinstance(p, (list, tuple)):
                p = p[0]
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            preds.append(p)
        except Exception as e:
            _log_warn(f"Predição falhou para um dos modelos do ensemble (path={meta.get('path')}): {e}")
            

    if not preds:
        st.error("Nenhuma predição válida obtida dos modelos LS15.")
        return []

    mean_pred = np.mean(preds, axis=0)[0]  # shape (25,)
    idx = np.argsort(-mean_pred)[:limite]
    return np.sort(idx + 1).tolist()

def gerar_palpite_ls14(limite=14):
    """
    Igual ao LS15, mas para LS14.
    """
    metas = carregar_ensemble_models("ls14")
    if not metas:
        st.error("Nenhum modelo LS14 encontrado/carregado.")
        return []

    # plano do usuário
    nome_plano = None
    try:
        nome_plano = st.session_state.usuario.get("nome_plano") or None
    except Exception:
        nome_plano = None

    if not nome_plano:
        try:
            id_plano = st.session_state.usuario.get("id_plano")
            if id_plano:
                db = Session()
                r = db.execute(text("SELECT nome FROM planos WHERE id = :id"), {"id": id_plano}).fetchone()
                db.close()
                if r:
                    nome_plano = r[0]
        except Exception:
            nome_plano = None

    allowed_groups = _groups_allowed_for_plan(nome_plano)
    metas = [m for m in metas if m.get("group") in allowed_groups]
    if not metas:
        st.error(f"Nenhum modelo LS14 disponível para seu plano ({nome_plano}).")
        return []

    max_window = max([m.get("expected_seq_len") or 0 for m in metas])
    if not max_window or max_window <= 0:
        max_window = 1550

    db = Session()
    try:
        rows = db.execute(text(f"""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY data DESC
            LIMIT :lim
        """), {"lim": int(max_window)}).fetchall()
    finally:
        db.close()

    if not rows:
        st.error("Não há histórico suficiente para gerar palpites LS14.")
        return []

    ultimos_full = [list(r) for r in reversed(rows)]

    preds = []
    for meta in metas:
        try:
            X = _prepare_inputs_for_model_meta(meta, ultimos_full)
            m = meta["model"]
            p = m.predict(X, verbose=0)
            if isinstance(p, (list, tuple)):
                p = p[0]
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            preds.append(p)
        except Exception as e:
            _log_warn(f"Predição falhou para um dos modelos do ensemble (path={meta.get('path')}): {e}")

    if not preds:
        st.error("Nenhuma predição válida obtida dos modelos LS14.")
        return []

    mean_pred = np.mean(preds, axis=0)[0]
    idx = np.argsort(-mean_pred)[:limite]
    return np.sort(idx + 1).tolist()

# -------------------- [7] UI / CORE - GENERATION --------------------
def gerar_palpite():
    """
    Função principal chamada pela streamlit app. Mantive a estrutura original
    com controles para plano, quantidade, tipo de modelo, etc.
    """
    st.title("Gerar Bets")

    # Verificar se usuário está logado
    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado para Gerar Bets.")
        return

    id_usuario = st.session_state.usuario["id"]
    id_plano = st.session_state.usuario.get("id_plano", 0)

    # Recuperar nome do plano do banco
    db = Session()
    try:
        result = db.execute(text("SELECT nome FROM planos WHERE id = :id_plano"), {"id_plano": id_plano})
        row = result.fetchone()
        nome_plano = row[0] if row else "Desconhecido"
    except Exception as e:
        st.error(f"Erro ao obter nome do plano: {e}")
        nome_plano = "Desconhecido"
    finally:
        db.close()

    # Mostrar o plano atual do usuário
    st.markdown(f"""
        <div style='font-family: "Poppins", sans-serif; font-size:16px; color:#0b450b; margin-bottom: 20px;'>
            Plano atual: <strong>{nome_plano}</strong>
        </div>
    """, unsafe_allow_html=True)

    try:
        permitido, nome_plano_verif, palpites_restantes = verificar_limite_palpites(id_usuario)
        if not permitido:
            if nome_plano_verif in ["Plano não encontrado", "Erro DB", "Erro"]:
                st.error(f"Erro ao verificar seu plano: {nome_plano_verif}")
            else:
                st.error(f"Você atingiu o limite de palpites do Plano {nome_plano_verif} para este mês.")
            return

        limite_dezenas = obter_limite_dezenas_por_plano(nome_plano)

        # Modelos disponíveis
        modelos_disponiveis = ["Aleatório", "Estatístico", "Pares/Ímpares"]
        if nome_plano in ["Silver", "Gold", "Plano Pago X"]:
            modelos_disponiveis += ["LS15", "LS14"]

        modelo = st.selectbox("Modelo de Geração:", modelos_disponiveis, key="select_modelo")

        num_palpites = st.number_input(
            "Quantos Palpites Deseja Gerar?",
            min_value=1,
            max_value=max(1, palpites_restantes),
            value=1,
            step=1,
            key="num_palpites"
        )

        # Limites de dezenas por plano
        min_dezenas = 15
        max_dezenas = 15  # padrão para Free
        if nome_plano == "Silver":
            max_dezenas = 17
        elif nome_plano == "Gold":
            max_dezenas = 20

        qtde_dezenas = st.number_input(
            "Quantas dezenas por palpite?",
            min_value=min_dezenas,
            max_value=max_dezenas,
            value=min_dezenas,
            step=1,
            key="qtde_dezenas"
        )

        if st.button("Gerar Palpites", key="btn_gerar_palpites"):
            palpites_gerados = []

            for _ in range(num_palpites):
                try:
                    if modelo == "Aleatório":
                        palpite = gerar_palpite_aleatorio(limite=qtde_dezenas)
                    elif modelo == "Estatístico":
                        palpite = gerar_palpite_estatistico(limite=qtde_dezenas)
                    elif modelo == "Pares/Ímpares":
                        palpite = gerar_palpite_pares_impares(limite=qtde_dezenas)
                    elif modelo == "LS15":
                        palpite = gerar_palpite_ls15(limite=qtde_dezenas)
                    elif modelo == "LS14":
                        palpite = gerar_palpite_ls14(limite=qtde_dezenas)
                    else:
                        st.warning("Modelo inválido.")
                        continue

                    if palpite:
                        salvar_palpite(palpite, modelo)
                        atualizar_contador_palpites(id_usuario)
                        palpites_gerados.append(palpite)

                except ValueError as e:
                    st.error(f"Erro ao gerar palpite: {e}")
                except Exception as e:
                    st.error(f"Erro inesperado ao gerar palpite: {e}")

            if palpites_gerados:
                st.success(f"{len(palpites_gerados)} Palpite(s) Gerado(s) com Sucesso:")

                for i in range(0, len(palpites_gerados), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(palpites_gerados):
                            palpite = palpites_gerados[i + j]
                            numeros_str = ', '.join(map(str, palpite)) if isinstance(palpite, list) else str(palpite)

                            with cols[j]:
                                st.markdown(f"""
                                    <div style='background:#fdfdfd; padding:10px 12px; border-radius:8px; border:1px solid #ccc; margin-bottom:12px'>
                                        <div style='font-size:13px; color:#1f77b4; font-weight:bold;'>Novo Palpite Gerado</div>
                                        <div style='font-size:11px; color:gray;'>{modelo} | {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
                                        <div style='font-family: monospace; font-size:14px; margin-top:6px;'>{numeros_str}</div>
                                        <button onclick="navigator.clipboard.writeText('{numeros_str}')" 
                                                style="margin-top:8px; padding:4px 8px; font-size:11px; border:none; background-color:#1f77b4; color:white; border-radius:5px; cursor:pointer;">
                                            Copiar
                                        </button>
                                    </div>
                                """, unsafe_allow_html=True)

                with st.expander("ℹ️ Aviso Sobre Cópia"):
                    st.markdown("Em alguns navegadores o botão de cópia pode não funcionar. Use o modo tradicional.")
            else:
                st.warning("Nenhum palpite foi gerado.")

    except Exception as e:
        st.error(f"Erro geral ao preparar o gerador de palpites: {e}")

    # -------------------- [8] HISTÓRICO / VALIDAÇÃO --------------------
def historico_palpites():
    if "usuario" not in st.session_state or not st.session_state.usuario:
        st.warning("Você precisa estar logado para acessar o histórico.")
        return

    st.markdown("### 📜 Histórico de Palpites")

    opcoes_modelo = ["Todos", "Aleatório", "Estatístico", "Ímpares-Pares", "LS15", "LS14"]
    filtro_modelo = st.selectbox("Filtrar por modelo:", opcoes_modelo)

    db = Session()
    try:
        query = """
            SELECT id, numeros, modelo, data, status 
            FROM palpites 
            WHERE id_usuario = :id
        """
        params = {"id": st.session_state.usuario["id"]}

        if filtro_modelo != "Todos":
            query += " AND modelo = :modelo"
            params["modelo"] = filtro_modelo

        query += " ORDER BY data DESC"
        result = db.execute(text(query), params)
        palpites = result.fetchall()

        if palpites:
            for i in range(0, len(palpites), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(palpites):
                        pid, numeros, modelo, data, status = palpites[i + j]
                        status_str = "✅ Válido" if status == "S" else "⏳ Não usado"

                        with cols[j]:
                            st.markdown(f"""
                                <div style='background:#fdfdfd; padding:8px 12px; border-radius:8px; border:1px solid #ccc; margin-bottom:10px; min-height:80px'>
                                    <div style='font-size:13px; color:#555; font-weight:bold;'>Palpite #{pid} | {modelo} | {status_str}</div>
                                    <div style='font-size:11px; color:gray;'>{data.strftime('%d/%m/%Y %H:%M')}</div>
                                    <div style='font-family: monospace; font-size:14px; margin-top:4px;'>{numeros}</div>
                                    <button onclick="navigator.clipboard.writeText('{numeros}')" 
                                            style="margin-top:6px; padding:3px 6px; font-size:11px; border:none; background-color:#1f77b4; color:white; border-radius:5px; cursor:pointer;">
                                        Copiar
                                    </button>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Nenhum palpite encontrado.")
    except Exception as e:
        st.error(f"Erro inesperado em historico_palpites: {e}")
    finally:
        db.close()
def validar_palpite():
    if "usuario" not in st.session_state or not st.session_state.usuario:
        st.warning("Você precisa estar logado para validar seus Palpites.")
        return
    st.markdown("### 📤 Validar Palpite")

    db = Session()
    try:
        palpites = db.execute(text("""
            SELECT id, numeros, modelo, data, status
            FROM palpites
            WHERE id_usuario = :uid AND status != 'S'
            ORDER BY data DESC
            LIMIT 10
        """), {"uid": st.session_state.usuario["id"]}).fetchall()

    except Exception as e:
        st.error(f"Erro ao buscar Palpites: {e}")
        return
    finally:
        db.close()

    if not palpites:
        st.info("Você ainda não gerou nenhum Palpite.")
        return

    st.markdown("#### Escolha o Palpite para validar:")
    opcoes = {
        f"#{pid} | {modelo} | {data.strftime('%d/%m %H:%M')}": pid
        for pid, _, modelo, data, _ in palpites
    }
    selecao = st.selectbox("Palpites disponíveis:", list(opcoes.keys()))

    if st.button("✅ Validar este Palpite"):
        palpite_id = opcoes[selecao]
        db = Session()
        try:
            db.execute(text("""
                UPDATE palpites
                SET status = 'S'
                WHERE id = :pid AND id_usuario = :uid
            """), {
                "pid": palpite_id,
                "uid": st.session_state.usuario["id"]
            })
            db.commit()
            st.success(f"Palpite #{palpite_id} marcado como validado com sucesso! Agora ele será destacado como oficial.")
            st.experimental_rerun()
        except Exception as e:
            db.rollback()
            st.error(f"Erro ao validar palpite: {e}")
        finally:
            db.close()

# -------------------- [9] DEBUG / UTIL --------------------
def verificar_modelos():
    """Função de debug: lista conteúdo de modelos conhecidos (não executada automaticamente)."""
    ls14_path = os.path.join(MODELS_DIR, "modelo_ls14pp_saved")
    ls15_path = os.path.join(MODELS_DIR, "modelo_ls15pp_saved")

    st.write(f"🔍 Procurando modelos em: {MODELS_DIR}")
    st.write(f"📁 LS14++ existe: {os.path.isdir(ls14_path)}")
    st.write(f"📁 LS15++ existe: {os.path.isdir(ls15_path)}")

    if os.path.isdir(ls14_path):
        try:
            conteudo = os.listdir(ls14_path)
            st.write(f"📂 Conteúdo LS14++: {conteudo[:10]}")
        except Exception as e:
            st.write(f"❌ Erro ao ler LS14++: {e}")
    if os.path.isdir(ls15_path):
        try:
            conteudo = os.listdir(ls15_path)
            st.write(f"📂 Conteúdo LS15++: {conteudo[:10]}")
        except Exception as e:
            st.write(f"❌ Erro ao ler LS15++: {e}")

# FIM DO ARQUIVO
