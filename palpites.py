# -*- coding: utf-8 -*-
"""
palpites_fixed.py  -- substitua seu palpites.py por este (backup antes).
Correções:
- busca recursiva flexível de modelos (arquivos .keras/.h5/dir SavedModel)
- aceita parâmetro models_dir em todas as funções relevantes
- cache que invalida quando a pasta de modelos muda (usa mtime)
- mensagens de diagnóstico claras (verificar_modelos)
- correções de assinaturas inconsistentes
"""
# ADIE imports pesados
import os, sys, logging, random
import streamlit as st
from datetime import datetime
from sqlalchemy import text
from db import Session

# Import leve
import numpy as np

import streamlit as st
from sqlalchemy import text

# tensorflow (carregamento)
import time
_t0 = time.time()

from tensorflow.keras.models import load_model as tf_load_model
import tensorflow as tf
from tensorflow import keras as tf_keras
print("Tempo para importar TF:", time.time() - _t0)

# util do projeto (se existir)
from modelo_llm_max.utils_ls_models import to_binary


# adia tensorflow/keras para dentro da função
_tf = None
_tf_load_model = None
_to_binary = None

def _lazy_imports():
    global _tf, _tf_load_model, _to_binary
    if _tf is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as tf_load_model
        from modelo_llm_max.utils_ls_models import to_binary
        _tf = tf
        _tf_load_model = tf_load_model
        _to_binary = to_binary

# -------------------- CONFIGURAÇÃO BÁSICA --------------------
try:
    _base_file = __file__
except NameError:
    _base_file = sys.argv[0] if sys.argv and sys.argv[0] else os.getcwd()

BASE_DIR = os.environ.get("FAIXABET_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(_base_file))))
# por padrão tenta o caminho relativo usado em deploys: modelo_llm_max/models/prod, e também ./models
DEFAULT_MODELS_DIRS = [
    os.path.join(BASE_DIR, "modelo_llm_max", "models", "prod"),
    os.path.join(BASE_DIR, "modelo_llm_max", "models"),
    os.path.join(BASE_DIR, "models"),
    os.environ.get("FAIXABET_MODELS_DIR", "")
]
# escolhe primeiro que existir
MODELS_DIR = next((p for p in DEFAULT_MODELS_DIRS if p and os.path.isdir(p)), DEFAULT_MODELS_DIRS[0])

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


def _log_warn(msg):
    if DEBUG_MODE:
        st.warning(msg)
    logging.warning(msg)


def _log_info(msg):
    if DEBUG_MODE:
        st.info(msg)
    logging.info(msg)


# TFSMLayer fallback (opcional - usado apenas como último recurso)
try:
    from keras.layers import TFSMLayer  # type: ignore
except Exception:
    try:
        from tensorflow.keras.layers import TFSMLayer  # type: ignore
    except Exception:
        TFSMLayer = None


# -------------------- REGRAS DE PLANOS --------------------
_PLAN_TO_GROUPS = {
    "free": ["recent"],
    "silver": ["recent", "mid"],
    "gold": ["recent", "mid", "global"],
    "plano pago x": ["recent", "mid", "global"]
}

def _groups_allowed_for_plan(nome_plano):
    if not nome_plano:
        return _PLAN_TO_GROUPS["free"]
    key = str(nome_plano).strip().lower()
    return _PLAN_TO_GROUPS.get(key, _PLAN_TO_GROUPS["free"])

# -------------------- UTILITÁRIOS DE BUSCA DE MODELO --------------------

def _model_name_variants(model_name):
    """Cria variações para facilitar matching."""
    m = model_name.lower()
    variants = set([m])
    variants.add(m + "pp")
    variants.add(m.replace("ls", "ls_"))
    variants.add(m + "_pp")
    # sem underscore
    variants.add(m.replace("_", ""))
    return list(variants)


def _detect_group_and_expected_from_path(p):
    """Detecta grupo (recent/mid/global) a partir do path/filename e sugere expected_seq_len."""
    low = p.lower()
    if "recent" in low:
        return "recent", 500
    if "mid" in low:
        return "mid", 1000
    if "global" in low:
        return "global", 1550
    # fallback: tenta por prefixos
    bv = os.path.basename(low)
    if bv.startswith("recent_"):
        return "recent", 500
    if bv.startswith("mid_"):
        return "mid", 1000
    if bv.startswith("global_"):
        return "global", 1550
    return "unknown", None

def _dir_mtime(models_dir):
    """Retorna o maior mtime dentre os arquivos em models_dir (0 se inexistente)."""
    try:
        max_m = 0.0
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                try:
                    p = os.path.join(root, f)
                    t = os.path.getmtime(p)
                    if t > max_m:
                        max_m = t
                except Exception:
                    continue
        return max_m
    except Exception:
        return 0.0

def _model_paths_for(model_name: str, models_dir: str = None):
    """
    Retorna os caminhos dos modelos (recent, mid, global) de forma
    compatível com DEV e PROD (detecta caminho automaticamente).
    """
    # 1️⃣ Prioriza o models_dir recebido
    if not models_dir or not os.path.exists(models_dir):
        # 2️⃣ Caso não seja válido, tenta detectar caminhos padrão
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "modelo_llm_max", "models", "prod"),
            os.path.join(base_dir, "modelo_llm_max", "models"),
            os.path.join(base_dir, "models", "prod"),
            os.path.join(base_dir, "models")
        ]
        models_dir = next((p for p in candidates if os.path.isdir(p)), candidates[-1])

    if not os.path.exists(models_dir):
        logging.warning(f"[model_paths_for] models_dir inválido: {models_dir}")
        return []

    padrões = [
        f"recent_{model_name}pp_final.keras",
        f"mid_{model_name}pp_final.keras",
        f"global_{model_name}pp_final.keras",
        f"{model_name}_recent.keras",
        f"{model_name}_mid.keras",
        f"{model_name}_global.keras",
    ]

    encontrados = []
    for padrao in padrões:
        # busca direta
        caminho = os.path.join(models_dir, padrao)
        if os.path.exists(caminho):
            encontrados.append(caminho)
        else:
            # busca recursiva
            for root, _, files in os.walk(models_dir):
                for f in files:
                    if f.lower() == padrao.lower():
                        encontrados.append(os.path.join(root, f))

    return encontrados


@st.cache_resource
def _cached_load_ensemble_models(model_name: str, models_dir: str, dir_mtime: float):
    """
    Carrega e retorna lista de metadados para os modelos do ensemble.
    `dir_mtime` é usado apenas para invalidar cache quando a pasta muda.
    """
    _lazy_imports()
    metas = []

    candidates = _model_paths_for(model_name, models_dir=models_dir)
    logging.info(f"[load_models] candidatos encontrados para '{model_name}': {len(candidates)} (base={models_dir})")

    for p in candidates:
        group, expected_from_path = _detect_group_and_expected_from_path(p)
        loaded = None
        try:
            try:
                loaded = _tf_load_model(p, compile=False)
            except Exception as e_load:
                logging.debug(f"Falha load_model {p}: {e_load}")
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "saved_model.pb")):
                    if TFSMLayer is not None and expected_from_path is not None:
                        try:
                            in_seq = _tf.keras.Input(shape=(expected_from_path, 25), name="seq")
                            tsl = TFSMLayer(p, call_endpoint="serving_default")
                            out = tsl(in_seq)
                            loaded = _tf.keras.Model(inputs=in_seq, outputs=out)
                        except Exception as e_tfs:
                            logging.warning(f"TFSMLayer falhou ({p}): {e_tfs}")
        except Exception as e_outer:
            logging.warning(f"Erro inesperado carregando {p}: {e_outer}")
            loaded = None

        if loaded is not None:
            expected_from_model = infer_expected_seq_from_loaded_model(loaded)
            expected = expected_from_model or expected_from_path
            input_shape = infer_model_input_shape(loaded)
            # n_inputs: número de entradas do modelo (1 ou >1)
            try:
                n_inputs = len(getattr(loaded, "inputs", []) or [1])
            except Exception:
                n_inputs = 1

            metas.append({
                "model": loaded,
                "path": p,
                "group": group,
                "expected_seq_len": expected,
                "n_inputs": n_inputs,
                "input_shape": input_shape
            })
            logging.info(f"✅ Modelo carregado: {p} (group={group}, expected={expected}, input_shape={input_shape})")

        else:
            logging.warning(f"⚠️ Modelo não carregado: {p}")

    if not metas:
        logging.warning(f"Nenhum modelo válido encontrado para '{model_name}' (dir={models_dir})")

    return metas

# ======================================================
# PATCH DE COMPATIBILIDADE - aceitar models_dir
# ======================================================
def carregar_ensemble_models(model_name, models_dir=None):
    """
    Wrapper compatível com versões antigas de palpites.py.
    Agora aceita models_dir explicitamente (opcional).
    """
    import streamlit as st
    import os
    import logging
    from datetime import datetime

    # tenta localizar o diretório de modelos
    if models_dir is None:
        base_dir = os.environ.get("FAIXABET_BASE_DIR", os.getcwd())
        possiveis = [
            os.path.join(base_dir, "modelo_llm_max", "models", "prod"),
            os.path.join(base_dir, "modelo_llm_max", "models"),
            os.path.join(base_dir, "models"),
        ]
        for p in possiveis:
            if os.path.isdir(p):
                models_dir = p
                break
        else:
            models_dir = possiveis[0]

    # chama função original (sem parâmetro extra) se existir
    try:
        from tensorflow.keras.models import load_model as tf_load_model
    except Exception as e:
        logging.error(f"TensorFlow indisponível: {e}")
        return []

    # --- cache simples para invalidar se a pasta mudar ---
    def _dir_mtime(models_dir):
        import os
        try:
            return max(
                os.path.getmtime(os.path.join(root, f))
                for root, _, files in os.walk(models_dir)
                for f in files
            )
        except Exception:
            return 0.0

    dir_mtime = _dir_mtime(models_dir)
    logging.info(f"[carregar_ensemble_models] usando base {models_dir} (mtime={dir_mtime})")

    # tenta usar versão nova (com cache)
    try:
        from palpites import _cached_load_ensemble_models
        metas = _cached_load_ensemble_models(model_name, models_dir=models_dir, dir_mtime=dir_mtime)
        return metas
    except TypeError:
        logging.warning("Versão antiga detectada, usando fallback simples.")
        # fallback para versões antigas que não aceitavam models_dir
        try:
            from palpites import _cached_load_ensemble_models
            metas = _cached_load_ensemble_models(model_name, dir_mtime=dir_mtime)
            return metas
        except Exception as e:
            logging.error(f"Falha ao carregar modelos: {e}")
            st.error(f"Erro ao carregar modelos ({model_name}): {e}")
            return []

# -------------------- DEBUG / DIAGNÓSTICO --------------------

def listar_candidatos_modelo(model_name, models_dir=None):
    """Retorna e escreve candidatos (para debug)."""
    if models_dir is None:
        models_dir = MODELS_DIR
    cand = _model_paths_for(model_name, models_dir=models_dir)
    st.write(f"🔎 Candidatos encontrados para '{model_name}' (base={models_dir}):")
    if cand:
        for c in cand:
            group, expected = _detect_group_and_expected_from_path(c)
            st.write(" - ", c, f" (group={group}, expected={expected})")
    else:
        st.write(" (nenhum candidato encontrado)")
    return cand


def verificar_modelos(models_dir=None):
    """Chamada via UI para diagnosticar pasta e arquivos de modelos."""
    _lazy_imports()
    if models_dir is None:
        models_dir = MODELS_DIR
    st.write(f"🔍 Procurando modelos em: {models_dir}")
    for key in ("ls14", "ls15"):
        st.write(f"\n--- Candidatos para {key} ---")
        listar_candidatos_modelo(key, models_dir=models_dir)

    st.write('\n--- Conteúdo raiz da pasta models/ (até 200 itens) ---')
    try:
        arquivos = os.listdir(models_dir)
        st.write(arquivos[:200])
    except Exception as e:
        st.write(f"Erro ao listar pasta de modelos: {e}")

# -------------------- UTILITÁRIOS DE PREPARO / PREDIÇÃO --------------------

def _ensure_window_list(ultimos_full, expected):
    cur = len(ultimos_full)
    if expected is None or cur >= expected:
        return ultimos_full[-expected:] if expected else ultimos_full
    if cur == 0:
        return [[0]*15 for _ in range(expected)]
    pad_count = expected - cur
    pad = [ultimos_full[0]] * pad_count
    return pad + ultimos_full


def infer_model_input_shape(loaded):
    """
    Retorna tupla (time_steps, features) extraída de loaded.inputs quando possível.
    Se não puder inferir, retorna None.
    """
    try:
        ins = getattr(loaded, 'inputs', None)
        if not ins:
            return None
        # normalize to list
        if not isinstance(ins, (list, tuple)):
            ins = [ins]
        for inp in ins:
            shape = getattr(inp, 'shape', None)
            if not shape:
                continue
            dims = []
            for d in shape:
                try:
                    dims.append(int(d))
                except Exception:
                    dims.append(None)
            # queremos pelo menos (batch, time_steps, features)
            if len(dims) >= 3:
                time_dim = dims[-2]
                feat_dim = dims[-1]
                return (time_dim, feat_dim)
        return None
    except Exception:
        return None


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
    """
    Prepara os dados de entrada (X) para o modelo conforme a meta.
    Faz a adequação automática do número de features (ex: LS14 → 14 colunas).
    """
    import numpy as np
    try:
        input_shape = meta.get("input_shape") or ()
        expected_seq_len = meta.get("expected_seq_len") or 50
        model_path = meta.get("path", "")
        model_name = os.path.basename(model_path).lower()

        # 🔍 Detecta o tipo de modelo (ls14, ls15, etc.)
        if "ls14" in model_name:
            n_features = 14
        elif "ls15" in model_name:
            n_features = 15
        elif "ls17" in model_name:
            n_features = 17
        elif "ls20" in model_name:
            n_features = 20
        else:
            n_features = input_shape[-1] if len(input_shape) > 1 else 15  # fallback

        # 🔧 Corta ou ajusta o número de features conforme o modelo
        seq = np.array(ultimos_full, dtype=np.float32)
        if seq.shape[1] > n_features:
            seq = seq[:, :n_features]
        elif seq.shape[1] < n_features:
            # padding com zeros se faltar coluna (evita erro de shape)
            pad = np.zeros((seq.shape[0], n_features - seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)

        # 🔁 recorta para o expected_seq_len (últimas N amostras)
        if seq.shape[0] > expected_seq_len:
            seq = seq[-expected_seq_len:]

        # reshape para (1, timesteps, features)
        X = np.expand_dims(seq, axis=0)

        return X

    except Exception as e:
        import logging
        logging.warning(f"[prepare_inputs] falha ao preparar dados p/ {meta.get('path')}: {e}")
        # fallback: tenta mínimo compatível
        arr = np.asarray(ultimos_full, dtype=np.float32)
        arr = arr[:, :14] if arr.shape[1] >= 14 else arr
        return np.expand_dims(arr[-50:], axis=0)


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
        atraso_vec[d - 1] = min(atraso, window) / float(window)
    last = ultimos[-1]
    soma = sum(last) / (25.0 * 15.0)
    pares = sum(1 for x in last if x % 2 == 0) / 15.0
    global_vec = np.array([soma, pares], dtype=np.float32)
    return seq_bin, freq_vec.astype(np.float32), atraso_vec.astype(np.float32), global_vec

# -------------------- DB / PLANOS (mantive a lógica original) --------------------

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

def gerar_palpite_pares_impares(limite=15):
    num_pares = limite // 2
    num_impares = limite - num_pares
    pares = random.sample(range(2, 26, 2), num_pares)
    impares = random.sample(range(1, 26, 2), num_impares)
    return sorted(pares + impares)

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
    filtered = [m for m in metas if m.get("group") in allowed_groups or m.get("group") == "unknown"]
    return filtered


def gerar_palpite_ls15(limite=15, models_dir=None):
    nome_plano = st.session_state.usuario.get('nome_plano') if 'usuario' in st.session_state and st.session_state.usuario else None
    res = gerar_palpite_ls("ls15", limite=limite, n_palites=1, nome_plano=nome_plano, models_dir=models_dir)
    return res[0] if res else []
    _lazy_imports()
    if 'usuario' in st.session_state and st.session_state.usuario:
        nome_plano = st.session_state.usuario.get('nome_plano')
    metas = _load_and_filter_metas_for_plan("ls15", nome_plano, models_dir=models_dir)
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
    """
    Gera palpites usando o modelo LS14.
    Disponível para usuários Silver e Gold.
    Usa o motor unificado gerar_palpite_ls().
    """
    _lazy_imports()
    nome_plano = None
    if 'usuario' in st.session_state and st.session_state.usuario:
        nome_plano = st.session_state.usuario.get('nome_plano')

    palpites = gerar_palpite_ls(
        tipo_modelo="ls14",
        limite=limite,
        n_palites=1,
        nome_plano=nome_plano,
        models_dir=models_dir
    )

    # Garante que sempre retorna uma lista ordenada
    if palpites and isinstance(palpites, list) and len(palpites) > 0:
        return sorted(palpites[0])
    else:
        st.warning("Nenhum palpite LS14 foi gerado.")
        return []


def historico_palpites():
    import datetime as dt

    if "usuario" not in st.session_state or not st.session_state.usuario:
        st.warning("Você precisa estar logado para acessar o histórico.")
        return

    st.markdown("### 📜 Histórico de Palpites")

    # ==============================
    # 🔹 Filtros Superiores
    # ==============================
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        opcoes_modelo = ["Todos", "Aleatório", "Estatístico", "Ímpares-Pares", "LS15", "LS14"]
        filtro_modelo = st.selectbox("🔢 Modelo de Palpite:", opcoes_modelo)

    with col2:
        filtro_status = st.selectbox(
            "📊 Status:",
            ["Todos", "✅ Válido", "⏳ Não usado"]
        )

    with col3:
        data_inicio = st.date_input("📅 Início:", value=dt.date.today() - dt.timedelta(days=30))
        data_fim = st.date_input("📅 Fim:", value=dt.date.today())

    # ==============================
    # 🔹 Consulta SQL dinâmica
    # ==============================
    db = Session()
    try:
        query = """
            SELECT id, numeros, modelo, data, status 
            FROM palpites 
            WHERE id_usuario = :id
              AND DATE(data) BETWEEN :data_inicio AND :data_fim
        """
        params = {
            "id": st.session_state.usuario["id"],
            "data_inicio": data_inicio,
            "data_fim": data_fim
        }

        if filtro_modelo != "Todos":
            query += " AND modelo = :modelo"
            params["modelo"] = filtro_modelo

        if filtro_status == "✅ Válido":
            query += " AND status = 'S'"
        elif filtro_status == "⏳ Não usado":
            query += " AND (status IS NULL OR status <> 'S')"

        query += " ORDER BY data DESC"
        result = db.execute(text(query), params)
        palpites = result.fetchall()

        # ==============================
        # 🔹 Contadores Dinâmicos
        # ==============================
        total = len(palpites)
        validos = sum(1 for p in palpites if p.status == "S")
        pendentes = total - validos

        st.markdown("""
            <div style='display:flex; gap:20px; margin-top:10px; margin-bottom:12px;'>
                <div style='flex:1; background:#f8f9fa; padding:10px; border-radius:8px; border:1px solid #ccc; text-align:center;'>
                    <div style='font-size:13px; color:#444;'>Total</div>
                    <div style='font-size:18px; font-weight:bold; color:#007bff;'>{total}</div>
                </div>
                <div style='flex:1; background:#f8fff8; padding:10px; border-radius:8px; border:1px solid #c7e6ca; text-align:center;'>
                    <div style='font-size:13px; color:#444;'>✅ Válidos</div>
                    <div style='font-size:18px; font-weight:bold; color:#28a745;'>{validos}</div>
                </div>
                <div style='flex:1; background:#fffaf8; padding:10px; border-radius:8px; border:1px solid #f2c2b0; text-align:center;'>
                    <div style='font-size:13px; color:#444;'>⏳ Não usados</div>
                    <div style='font-size:18px; font-weight:bold; color:#e67e22;'>{pendentes}</div>
                </div>
            </div>
        """.format(total=total, validos=validos, pendentes=pendentes), unsafe_allow_html=True)

        # ==============================
        # 🔹 Exibição dos resultados
        # ==============================
        if palpites:
            for i in range(0, len(palpites), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(palpites):
                        pid, numeros, modelo, data, status = palpites[i + j]
                        eh_valido = status == "S"
                        status_str = "✅ Válido" if eh_valido else "⏳ Não usado"
                        cor = "#28a745" if eh_valido else "#999"

                        with cols[j]:
                            st.markdown(f"""
                                <div style='background:#fff; padding:10px 14px; border-radius:10px; 
                                            border:1px solid #ccc; margin-bottom:12px; min-height:90px;
                                            box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                                    <div style='font-size:13px; color:#444; font-weight:bold;'>
                                        Palpite #{pid} | <span style='color:#007bff'>{modelo}</span>
                                    </div>
                                    <div style='font-size:12px; color:{cor}; margin-bottom:4px;'>{status_str}</div>
                                    <div style='font-size:11px; color:gray;'>{data.strftime('%d/%m/%Y %H:%M')}</div>
                                    <div style='font-family:monospace; font-size:14px; margin-top:6px;'>{numeros}</div>
                                    <button onclick="navigator.clipboard.writeText('{numeros}')" 
                                            style="margin-top:6px; padding:3px 8px; font-size:11px; border:none; 
                                                   background-color:#1f77b4; color:white; border-radius:6px; cursor:pointer;">
                                        Copiar
                                    </button>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Nenhum palpite encontrado com os filtros selecionados.")
    except Exception as e:
        st.error(f"Erro inesperado em histórico de palpites: {e}")
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

def carregar_modelo_ls(modelo: str, nome_plano: str, models_dir=None):
    """
    Carrega os modelos LS14 ou LS15 disponíveis para o plano do usuário.
    """
    if not nome_plano:
        return []

    planos_acesso = {
        "ls14": ["silver", "gold"],
        "ls15": ["gold"]
    }

    if nome_plano.strip().lower() not in planos_acesso.get(modelo.lower(), []):
        return []

    metas = _load_and_filter_metas_for_plan(modelo.lower(), nome_plano, models_dir=models_dir)
    return metas

def gerar_palpite_ls(modelo: str, limite=15, n_palites=1, nome_plano=None, models_dir=None):
    """
    Gera n_palites para LS14 ou LS15.
    - aceita apenas modelos que retornem vetores de predição com 25 elementos (1..25)
    - se nenhum modelo compatível, faz fallback estatístico (frequência histórica)
    - amostragem probabilística via gerar_palpite_from_probs para diversity
    """
    _lazy_imports()

    metas = carregar_modelo_ls(modelo, nome_plano, models_dir=models_dir)
    if not metas:
        _log_warn(f"Nenhum modelo disponível/permitido para {modelo} no plano {nome_plano}.")
        return []

    # janela máxima requisitada pelos modelos
    max_window = max([m.get("expected_seq_len") or 0 for m in metas]) or 1550

    db = Session()
    try:
        rows = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY data DESC
            LIMIT :lim
        """), {"lim": int(max_window)}).fetchall()
    finally:
        db.close()

    if not rows:
        _log_warn("Nenhum histórico encontrado em resultados_oficiais.")
        return []

    ultimos_full = [list(r) for r in reversed(rows)]

    preds = []
    skipped_models = []
    for meta in metas:
        try:
            X = _prepare_inputs_for_model_meta(meta, ultimos_full)
            model_obj = meta.get("model")
            if model_obj is None:
                _log_warn(f"Modelo inválido em meta ({meta.get('path')}). Ignorando.")
                skipped_models.append(meta.get('path'))
                continue

            p = model_obj.predict(X, verbose=0)
            if isinstance(p, (list, tuple)):
                p = p[0]
            p = np.asarray(p, dtype=np.float32)

            # normaliza shape para vetor 1D
            if p.ndim > 1:
                p = p.reshape(-1)

            # **CRITÉRIO DE COMPATIBILIDADE**: só aceitamos vetores com 25 probabilidades
            if p.size != 25:
                _log_warn(f"Modelo {meta.get('path')} retornou vetor de tamanho {p.size} (esperado 25). Ignorando este modelo.")
                skipped_models.append(meta.get('path'))
                continue

            preds.append(p)
        except Exception as e:
            _log_warn(f"Predição falhou para {modelo.upper()} (path={meta.get('path')}): {e}")
            skipped_models.append(meta.get('path'))

    if not preds:
        _log_warn(f"Nenhuma predição compatível (25 probs) para {modelo.upper()}. Fazendo fallback estatístico.")
        # fallback: construir probabilidades a partir do histórico
        freq = np.zeros(25, dtype=np.float32)
        for r in rows:
            for num in r:
                if 1 <= int(num) <= 25:
                    freq[int(num) - 1] += 1.0
        if freq.sum() <= 0:
            probs = np.ones(25, dtype=np.float32) / 25.0
        else:
            probs = freq / freq.sum()

        palpites = []
        tentativas = set()
        for _ in range(n_palites):
            for _t in range(60):
                pal = gerar_palpite_from_probs(probs, limite=limite, deterministic=False)
                tpl = tuple(pal)
                if tpl not in tentativas:
                    tentativas.add(tpl)
                    palpites.append(pal)
                    break
            else:
                palpites.append(gerar_palpite_from_probs(probs, limite=limite, deterministic=True))
        return palpites

    # agregação do ensemble (média)
    try:
        stacked = np.vstack([np.asarray(x).reshape(1, -1) for x in preds])
        mean_pred = stacked.mean(axis=0)
    except Exception as e:
        _log_warn(f"Erro ao agregar preds do ensemble ({e}). Usando média simples.")
        mean_pred = np.mean(preds, axis=0)
        if mean_pred.ndim > 1:
            mean_pred = mean_pred.reshape(-1)

    # garante que mean_pred seja não-negativo e normalize para soma 1 (probabilidades)
    mean_pred = np.clip(mean_pred, 1e-12, None)
    if mean_pred.sum() <= 0:
        mean_pred = np.ones(25, dtype=np.float32) / 25.0
    else:
        mean_pred = mean_pred / mean_pred.sum()

    # Geração de palpites usando a função de amostragem (vai respeitar probabilidades e variar)
    palpites = []
    tentativas = set()
    for _ in range(n_palites):
        for _t in range(60):
            pal = gerar_palpite_from_probs(mean_pred, limite=limite, deterministic=False)
            tpl = tuple(pal)
            if tpl not in tentativas:
                tentativas.add(tpl)
                palpites.append(pal)
                break
        else:
            palpites.append(gerar_palpite_from_probs(mean_pred, limite=limite, deterministic=True))

    # log útil para debug (mostra quantos modelos foram usados vs ignorados)
    _log_info(f"[gerar_palpite_ls] modelo={modelo} metas_total={len(metas)} metas_usadas={len(preds)} ignorados={len(skipped_models)}")

    return palpites


# -------------------- UI / CORE - GENERATION --------------------
def gerar_palpite_ui():
    import random
    st.title("Gerar Bets")
    _lazy_imports()

    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado para gerar Bets.")
        return

    id_usuario = st.session_state.usuario["id"]
    # Antes de gerar palpites
    nome_plano = st.session_state.usuario.get("nome_plano")
    if not nome_plano:
        # tenta buscar do banco
        db = Session()
        try:
            result = db.execute(text("SELECT nome FROM planos WHERE id = :id_plano"), 
                                {"id_plano": st.session_state.usuario.get("id_plano", 0)})
            row = result.fetchone()
            nome_plano = row[0] if row else None
        except Exception as e:
            st.error(f"Erro ao obter nome do plano: {e}")
            return
        finally:
            db.close()

    if not nome_plano:
        st.error("Plano do usuário não encontrado. Contate o suporte.")
        return

    st.markdown(f"<div>Plano atual: <strong>{nome_plano}</strong></div>", unsafe_allow_html=True)
    permitido, nome_plano_verif, palpites_restantes = verificar_limite_palpites(id_usuario)
    if not permitido:
        st.error(f"Você atingiu o limite de palpites do Plano {nome_plano_verif} para este mês.")
        return

    # Define modelos por plano
    plano_clean = nome_plano.lower()
    if plano_clean == "free":
        modelos_disponiveis = ["Aleatório"]
        min_dezenas, max_dezenas = 15, 15
    elif plano_clean == "silver":
        modelos_disponiveis = ["Estatístico", "Pares/Ímpares", "LS14"]
        min_dezenas, max_dezenas = 15, 17
    elif plano_clean == "gold":
        modelos_disponiveis = ["Estatístico", "Pares/Ímpares", "LS14", "LS15"]
        min_dezenas, max_dezenas = 15, 20
    else:
        modelos_disponiveis = ["Aleatório"]
        min_dezenas, max_dezenas = 15, 15

    modelo = st.selectbox("Modelo de Geração:", modelos_disponiveis, key="select_modelo")
    qtde_dezenas = st.number_input("Quantas dezenas por palpite?", min_value=min_dezenas, max_value=max_dezenas,
                                   value=min_dezenas, step=1, key="qtde_dezenas")
    num_palpites = st.number_input("Quantos palpites deseja gerar?", min_value=1,
                                   max_value=max(1, palpites_restantes),
                                   value=1, step=1, key="num_palpites")

    if st.button("Gerar Palpites", key="btn_gerar_palpites_ui"):
        palpites_gerados = []
        tentativas = set()

        # LS14 predição para priorizar LS15
        ls14_prior = []
        if plano_clean == "gold" and modelo == "LS15":
            ls14_prior = gerar_palpite_ls("ls14", limite=15, n_palites=1, nome_plano=nome_plano, models_dir=MODELS_DIR)
            ls14_prior = ls14_prior[0] if ls14_prior else []

        for _ in range(num_palpites):
            try:
                if modelo == "Aleatório":
                    palpite = sorted(random.sample(range(1, 26), qtde_dezenas))
                elif modelo == "Estatístico":
                    palpite = gerar_palpite_estatistico(limite=qtde_dezenas)
                elif modelo == "Pares/Ímpares":
                    palpite = gerar_palpite_pares_impares(limite=qtde_dezenas)
                elif modelo == "LS14":
                    palpite = gerar_palpite_ls("ls14", limite=qtde_dezenas, n_palites=1,
                                               nome_plano=nome_plano, models_dir=MODELS_DIR)[0]
                elif modelo == "LS15":
                    palpite = gerar_palpite_ls("ls15", limite=qtde_dezenas, n_palites=1,
                                               nome_plano=nome_plano, models_dir=MODELS_DIR)[0]
                    # prioriza LS14 para Gold
                    if ls14_prior:
                        intersec = list(set(palpite) & set(ls14_prior))
                        extras = [x for x in ls14_prior if x not in intersec]
                        for x in extras:
                            if len(intersec) < qtde_dezenas:
                                intersec.append(x)
                        for x in range(1, 26):
                            if len(intersec) < qtde_dezenas and x not in intersec:
                                intersec.append(x)
                        palpite = sorted(intersec[:qtde_dezenas])

                # adiciona palpite único
                tpl = tuple(palpite)
                if tpl not in tentativas:
                    tentativas.add(tpl)
                    salvar_palpite(palpite, modelo)
                    atualizar_contador_palpites(id_usuario)
                    palpites_gerados.append(palpite)

            except Exception as e:
                st.error(f"Erro ao gerar palpite: {e}")

        if palpites_gerados:
            st.success(f"{len(palpites_gerados)} palpite(s) gerado(s) com sucesso:")
            for p in palpites_gerados:
                st.write(", ".join(map(str, p)))
        else:
            st.warning("Nenhum palpite foi gerado.")
