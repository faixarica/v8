# palpites_v2.py
# Versão revisada do palpites.py com carregamento robusto de modelos por grupo
# - Busca modelos contendo várias variações de nomes (ls15, ls15pp, etc.)
# - Detecta grupo (recent/mid/global) em qualquer parte do nome/path
# - Normaliza nomes de plano (case-insensitive)
# - Cache do carregamento leva em conta a pasta de modelos (MODELS_DIR) para evitar staleness
# Substitua seu palpites.py atual por este arquivo (faça backup antes).

import streamlit as st
import random
from datetime import datetime
import os
import sys
import logging
import numpy as np
from sqlalchemy import text
from db import Session
from tensorflow.keras.models import load_model as tf_load_model
from modelo_llm_max.utils_ls_models import to_binary
import tensorflow as tf
from tensorflow import keras as tf_keras

# -------------------- CONFIGURAÇÃO BÁSICA --------------------
try:
    _base_file = __file__
except NameError:
    _base_file = sys.argv[0] if sys.argv and sys.argv[0] else os.getcwd()

BASE_DIR = os.environ.get("FAIXABET_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(_base_file))))
MODELS_DIR = os.environ.get("FAIXABET_MODELS_DIR", os.path.join(BASE_DIR, "models"))

# DEBUG
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"


def _log_warn(msg):
    if DEBUG_MODE:
        st.warning(msg)
    else:
        logging.warning(msg)


def _log_info(msg):
    if DEBUG_MODE:
        st.info(msg)
    else:
        logging.info(msg)


# TFSMLayer fallback (opcional)
try:
    from keras.layers import TFSMLayer
except Exception:
    try:
        from tensorflow.keras.layers import TFSMLayer
    except Exception:
        TFSMLayer = None


# -------------------- REGRAS DE PLANOS (normalize keys to lower-case) --------------------
_PLAN_TO_GROUPS = {
    "free": ["recent"],
    "silver": ["recent", "mid"],
    "gold": ["recent", "mid", "global"],
    "plano pago x": ["recent", "mid", "global"]
}


def _groups_allowed_for_plan(nome_plano):
    """Retorna lista de grupos permitidos para um nome de plano (case-insensitive).
    Se nome_plano for None/empty ou não estiver mapeado, assume comportamento conservador (Free).
    """
    if not nome_plano:
        return _PLAN_TO_GROUPS["free"]
    key = str(nome_plano).strip().lower()
    return _PLAN_TO_GROUPS.get(key, _PLAN_TO_GROUPS["free"])


# -------------------- DETECÇÃO E BUSCA DE MODELOS --------------------

def _detect_group_and_expected_from_path(p):
    """Detecta grupo (recent/mid/global) a partir do path/filename (procura em qualquer parte do nome).
    Também tenta inferir um expected window a partir do grupo (mantendo convenção anterior).
    """
    low = os.path.basename(p).lower() + " " + p.lower()
    if "recent" in low:
        return "recent", 500
    if "mid" in low:
        return "mid", 1000
    if "global" in low:
        return "global", 1550

    # tentativas por prefixo 'recent_' 'mid_' 'global_' já cobertas, mas deixamos fallback
    # Se não identificar, devolve 'unknown'
    return "unknown", None


def _model_name_variants(model_name):
    """Gera variantes comuns para facilitar matching (ls15 -> ls15, ls15pp, ls_15, etc.)."""
    m = model_name.lower()
    variants = {m}
    variants.add(m + "pp")
    variants.add(m.replace("ls", "ls_"))
    variants.add(m + "_pp")
    return list(variants)

def _model_paths_for(tipo: str):
    """
    Retorna apenas os modelos oficiais de produção (recent, mid, global).
    """
    base = os.path.join(os.getcwd(), "modelo_llm_max", "models", "prod")
    padrões = [
        f"recent_{tipo}pp_final.keras",
        f"mid_{tipo}pp_final.keras",
        f"global_{tipo}pp_final.keras"
    ]
    encontrados = []
    for p in padrões:
        caminho = os.path.join(base, p)
        if os.path.exists(caminho):
            encontrados.append(caminho)
    return encontrados


@st.cache_resource
def _cached_load_ensemble_models(model_name, models_dir=None):
    """Carrega (e cacheia) modelos para model_name dentro de models_dir.
    A presença de models_dir no argumento garante que mudanças na pasta invalidam o cache.
    Retorna lista de metas: {model, path, group, expected_seq_len, n_inputs}
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    metas = []
    candidates = _model_paths_for(model_name, models_dir=models_dir)

    logging.info(f"[load_models] candidatos encontrados para '{model_name}': {len(candidates)} (base={models_dir})")

    for p in candidates:
        group, expected_from_path = _detect_group_and_expected_from_path(p)
        loaded = None
        try:
            # tentar carregar arquivos .keras/.h5 direto
            if os.path.isfile(p) and p.lower().endswith((".keras", ".h5", ".hdf5", ".zip")):
                loaded = tf_load_model(p, compile=False)
            elif os.path.isdir(p):
                # diretório: pode ser SavedModel
                sm_pb = os.path.join(p, "saved_model.pb")
                if os.path.isfile(sm_pb):
                    # se for possível montar wrapper com TFSMLayer
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
                        logging.warning(f"[load_models] SavedModel detectado em {p} mas wrapper não disponível.")
                        loaded = None
                else:
                    # talvez seja um diretório com arquivos .keras dentro
                    for root, _, files in os.walk(p):
                        for f in files:
                            if f.lower().endswith((".keras", ".h5", ".hdf5")) and any(v in f.lower() for v in _model_name_variants(model_name)):
                                try:
                                    loaded = tf_load_model(os.path.join(root, f), compile=False)
                                    p = os.path.join(root, f)
                                    break
                                except Exception as e_lf:
                                    logging.warning(f"[load_models] falha ao carregar {os.path.join(root,f)}: {e_lf}")
                        if loaded is not None:
                            break

        except Exception as e_load:
            logging.warning(f"[load_models] Falha ao carregar {p}: {e_load}")
            loaded = None

        if loaded is not None:
            n_inputs = len(getattr(loaded, "inputs", [])) or 1
            expected_from_model = None
            try:
                expected_from_model = None
                ins = getattr(loaded, 'inputs', None)
                if ins:
                    for inp in ins:
                        shp = getattr(inp, 'shape', None)
                        if shp is None:
                            continue
                        # procurar input do tipo (None, TIME, 25)
                        if len(shp) >= 2 and int(shp[-1]) == 25:
                            # time dimension é shp[-2] (pode ser None)
                            expected_from_model = shp[-2]
                            break
            except Exception:
                expected_from_model = None

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


def carregar_ensemble_models(model_name, models_dir=None):
    """Wrapper que mostra spinner e retorna metas carregados. models_dir pode ser passado
    para forçar recarregamento quando a pasta mudou.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    cand_count = 0
    try:
        cand_count = len(_model_paths_for(model_name, models_dir=models_dir))
    except Exception:
        cand_count = 0

    with st.spinner(f"🚴 Carregando modelos '{model_name}' ({cand_count} candidatos encontrados)..."):
        metas = _cached_load_ensemble_models(model_name, models_dir=models_dir)

    if not metas:
        _log_warn(f"Nenhum modelo válido carregado para '{model_name}' (base={models_dir}). Use verificar_modelos() para diagnosticar.")
    else:
        _log_info(f"{len(metas)} modelo(s) carregado(s) para '{model_name}'.")

    return metas


# -------------------- UTILITÁRIOS --------------------

def _ensure_window_list(ultimos_full, expected):
    cur = len(ultimos_full)
    if cur >= expected:
        return ultimos_full[-expected:]
    if cur == 0:
        return [[0]*15 for _ in range(expected)]
    pad_count = expected - cur
    pad = [ultimos_full[0]] * pad_count
    return pad + ultimos_full


def infer_expected_seq_from_loaded_model(loaded):
    try:
        ins = getattr(loaded, 'inputs', None)
        if not ins:
            return None
        for inp in ins:
            try:
                shape = getattr(inp, 'shape', None)
                if shape is None:
                    continue
                dims = []
                for d in shape:
                    try:
                        dims.append(int(d))
                    except Exception:
                        dims.append(None)
                if len(dims) >= 3 and dims[-1] == 25:
                    return dims[-2]
            except Exception:
                continue
    except Exception:
        pass
    return None


def _prepare_inputs_for_model_meta(meta, ultimos_full):
    expected = meta.get("expected_seq_len", None)
    if expected is None:
        window_list = ultimos_full
    else:
        window_list = _ensure_window_list(ultimos_full, expected)

    seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(window_list)
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


def listar_candidatos_modelo(model_name, models_dir=None):
    """Função de debug/diagnóstico que escreve os caminhos candidatos encontrados."""
    if models_dir is None:
        models_dir = MODELS_DIR
    cand = _model_paths_for(model_name, models_dir=models_dir)
    st.write(f"🔎 Candidatos encontrados para '{model_name}' (base={models_dir}):")
    if cand:
        for c in cand:
            st.write(" - ", c)
    else:
        st.write(" (nenhum candidato encontrado)")
    return cand


# -------------------- PREDIÇÃO EM ENSEMBLE --------------------

def ensemble_predict(models_list, X_inputs):
    if not models_list:
        raise ValueError("Nenhum modelo carregado para ensemble.")

    preds = []
    for entry in models_list:
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
            name = getattr(model_obj, "name", getattr(model_obj, "__class__", None))
            _log_warn(f"Predição falhou para um dos modelos do ensemble ({name}): {e}")

    if not preds:
        raise ValueError("Nenhuma predição válida obtida dos modelos.")

    mean_pred = np.mean(preds, axis=0)
    return mean_pred


# -------------------- FEATURES / GERADORES SIMPLES --------------------

def montar_entrada_binaria(ultimos_concursos):
    arr = np.array([[1.0 if (i+1) in jogo else 0.0 for i in range(25)] for jogo in ultimos_concursos], dtype=np.float32)
    return arr


def apply_temperature(p, T=1.0):
    p = np.clip(p, 1e-12, 1.0)
    logits = np.log(p)
    scaled = np.exp(logits / float(T))
    return scaled / scaled.sum()


def gerar_palpite_from_probs(probs, limite=15, reinforce_threshold=0.06, boost_factor=2.0, temperature=1.0, deterministic=False):
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
    seq_bin = np.array([to_binary(j) for j in ultimos], dtype=np.float32)
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


# -------------------- DB / PLANOS --------------------

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


# -------------------- GERADORES SIMPLES --------------------

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
        """))
        resultados = resultados.fetchall()
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


# -------------------- GERADORES ML (LS15 / LS14) --------------------

def _load_and_filter_metas_for_plan(model_name, nome_plano, models_dir=None):
    metas = carregar_ensemble_models(model_name, models_dir=models_dir)
    if not metas:
        return []
    allowed_groups = _groups_allowed_for_plan(nome_plano)
    # filtra
    filtered = [m for m in metas if m.get("group") in allowed_groups]
    return filtered


def gerar_palpite_ls15(limite=15, models_dir=None):
    metas = _load_and_filter_metas_for_plan("ls15", None if 'usuario' not in st.session_state else st.session_state.usuario.get('nome_plano'), models_dir=models_dir)
    if not metas:
        st.error("Nenhum modelo LS15 disponível para seu plano. Use verificar_modelos() para diagnosticar onde estão os arquivos.")
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
        st.error("Não há histórico suficiente para gerar palpites LS15.")
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
        st.error("Nenhuma predição válida obtida dos modelos LS15.")
        return []

    mean_pred = np.mean(preds, axis=0)[0]
    idx = np.argsort(-mean_pred)[:limite]
    return np.sort(idx + 1).tolist()


def gerar_palpite_ls14(limite=14, models_dir=None):
    metas = _load_and_filter_metas_for_plan("ls14", None if 'usuario' not in st.session_state else st.session_state.usuario.get('nome_plano'), models_dir=models_dir)
    if not metas:
        st.error("Nenhum modelo LS14 disponível para seu plano. Use verificar_modelos() para diagnosticar onde estão os arquivos.")
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


# -------------------- UI / CORE - GENERATION (mantido parecido com original) --------------------
def gerar_palpite():
    st.title("Gerar Bets")

    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado para Gerar Bets.")
        return

    id_usuario = st.session_state.usuario["id"]
    id_plano = st.session_state.usuario.get("id_plano", 0)

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

        modelos_disponiveis = ["Aleatório", "Estatístico", "Pares/Ímpares"]
        if str(nome_plano).strip().lower() in ["silver", "gold", "plano pago x"]:
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

        min_dezenas = 15
        max_dezenas = 15
        if str(nome_plano).strip().lower() == "silver":
            max_dezenas = 17
        elif str(nome_plano).strip().lower() == "gold":
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
                        palpite = gerar_palpite_ls15(limite=qtde_dezenas, models_dir=MODELS_DIR)
                    elif modelo == "LS14":
                        palpite = gerar_palpite_ls14(limite=qtde_dezenas, models_dir=MODELS_DIR)
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


# -------------------- HISTÓRICO / VALIDAÇÃO / DEBUG --------------------
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
    opcoes = {f"#{pid} | {modelo} | {data.strftime('%d/%m %H:%M')}": pid for pid, _, modelo, data, _ in palpites}
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


# -------------------- DEBUG: verificar_modelos --------------------

def verificar_modelos(models_dir=None):
    """Diagnóstico rápido para checar como o sistema vê os modelos (use via UI)."""
    if models_dir is None:
        models_dir = MODELS_DIR
    st.write(f"🔍 Procurando modelos em: {models_dir}")

    for key in ("ls14", "ls15"):
        st.write(f"\n--- Candidatos para {key} ---")
        listar_candidatos_modelo(key, models_dir=models_dir)

    st.write('\n--- Lista de arquivos diretos na raiz models/ ---')
    try:
        arquivos = os.listdir(models_dir)
        st.write(arquivos[:50])
    except Exception as e:
        st.write(f"Erro ao listar pasta de modelos: {e}")


# FIM
