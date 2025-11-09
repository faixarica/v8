# -*- coding: utf-8 -*-
"""
palpites_v8.10.py
Consolidado e funcional — compatível com Streamlit e PostgreSQL.
Inclui:
- geração de palpites (estatístico, pares/ímpares, LS14, LS15)
- controle de limites por plano (via tabela planos)
- gravação dinâmica no banco (detecta colunas reais)
- histórico e validação independentes
"""
import streamlit.components.v1 as components
import random
import pandas as pd
import os, random, logging, json
import numpy as np
import streamlit as st
from sqlalchemy import text
from datetime import datetime, date, timedelta
from db import Session
import math
from functools import lru_cache
import re
# --- LS16: ensemble inteligente (tenta usar modelo_llm_loteria/ensemble.py)
try:
    from modelo_llm_max.ensemble import gerar_palpite_ensemble as _fxb_ls16_ensemble
except Exception:
    _fxb_ls16_ensemble = None

# =========================
# CONFIG LOG
# =========================

# --- LS16: ensemble inteligente (tenta usar modelo_llm_max/ensemble.py)
try:
    from modelo_llm_max.ensemble import gerar_palpite_ensemble as _fxb_ls16_ensemble
except Exception:
    _fxb_ls16_ensemble = None

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",)

def _safe_rerun():
    """Rerun compatível com versões novas e antigas do Streamlit."""
    try:
        st.rerun()  # Streamlit ≥ 1.27
    except AttributeError:
        # Streamlit mais antigo
        _safe_rerun()
        
def _log_info(msg):
    logging.info(msg)
    if DEBUG_MODE:
        st.info(msg)

def _log_warn(msg):
    logging.warning(msg)
    if DEBUG_MODE:
        st.warning(msg)

def get_conn():
    """
    Retorna conexão SQLAlchemy com o banco PostgreSQL (Neon.tech).
    Usa variáveis de ambiente: DATABASE_URL, POSTGRES_URL ou PG_URI.
    """
    db_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("PG_URI")
    )

    if not db_url:
        st.error("❌ Variável de conexão DATABASE_URL não encontrada no ambiente.")
        return None

    try:
        engine = create_engine(db_url)
        return engine.connect()
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# =========================
# FUNÇÕES AUXILIARES
# =========================
def _existing_cols(table_name: str) -> set:
    """Retorna o conjunto de nomes de colunas existentes."""
    db = Session()
    try:
        rows = db.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :t
        """), {"t": table_name}).fetchall()
        return {r[0] for r in rows} if rows else set()
    except Exception as e:
        _log_warn(f"Erro ao buscar colunas da tabela {table_name}: {e}")
        return set()
    finally:
        db.close()

# ================== [MODEL BRIDGE] LS14 / LS15 / LS16 ==================

# --- caminhos / arquivos
BASE_DIR = os.getcwd()
CSV_LOTOFACIL = os.path.join(BASE_DIR, "loteria.csv")  # ajuste se precisar

# -------- Estatístico (freq + recência + correlação) --------

def _carregar_df_lotofacil():
    if not os.path.exists(CSV_LOTOFACIL):
        raise FileNotFoundError(f"CSV não encontrado: {CSV_LOTOFACIL}")
    df = pd.read_csv(CSV_LOTOFACIL, encoding="utf-8")
    bolas = [f"Bola{i}" for i in range(1, 16)]
    for b in bolas:
        df[b] = df[b].astype(str).str.zfill(2)
    if "Concurso" in df.columns:
        df = df.sort_values("Concurso", ascending=False)
    return df, bolas

@lru_cache(maxsize=1)
def _stats_pack(recencia=100):
    df, bolas = _carregar_df_lotofacil()

    # Frequência global
    todas = df[bolas].values.flatten()
    freq_global = pd.Series(todas).value_counts().sort_index()
    freq_global = {int(k): int(v) for k, v in freq_global.items()}

    # Frequência recente
    subset = df.head(recencia) if len(df) > recencia else df
    todas_rec = subset[bolas].values.flatten()
    freq_rec = pd.Series(todas_rec).value_counts().sort_index()
    freq_rec = {int(k): int(v) for k, v in freq_rec.items()}

    # Correlação (coocorrência par-a-par)
    nums_df = df[bolas].apply(pd.to_numeric)
    corr = pd.DataFrame(0, index=range(1, 26), columns=range(1, 26))
    for _, row in nums_df.iterrows():
        arr = row.values
        for i in range(15):
            for j in range(i+1, 15):
                a, b = int(arr[i]), int(arr[j])
                corr.loc[a, b] += 1
                corr.loc[b, a] += 1

    return freq_global, freq_rec, corr, df, bolas

def scores_estatisticos(alpha_hist=0.7, alpha_recent=0.3):
    """
    Frequência histórica + recência (vetor 25 normalizado).
    """
    freq_global, freq_rec, _, _, _ = _stats_pack()
    v = np.zeros(25, dtype=float)
    for d in range(1, 26):
        g = float(freq_global.get(d, 0))
        r = float(freq_rec.get(d, 0))
        v[d-1] = alpha_hist * g + alpha_recent * r
    return _normalize_probs(v)

def scores_correlacao(boost_from=set()):
    """
    Coocorrência: soma linhas da matriz de coocorrência para dezenas 'boost_from'.
    """
    _, _, corr, _, _ = _stats_pack()
    if not boost_from:
        return np.ones(25, dtype=float) / 25.0
    s = np.zeros(25, dtype=float)
    for b in boost_from:
        try:
            bi = int(b)
            if 1 <= bi <= 25:
                s += np.asarray(corr.loc[bi].values, dtype=float)
        except Exception:
            continue
    s = np.maximum(s, 0.0)
    return _normalize_probs(s)

# cache global para evitar loop recursivo
# --- Anti-loop / caches globais ---
_IN_SCORES_LS16 = False         # flag de reentrância para LS16
_LS15_CACHE = None              # cache de vetor (25,) vindo do LS15
_LOADED_METAS = {}              # cache de modelos carregados por (model_name, models_dir, mtime)

@lru_cache(maxsize=1)
def combinacoes_ja_sorteadas():
    """Set de tuplas(15) com todas as combinações já sorteadas (para veto)."""
    _, _, _, df, bolas = _stats_pack()
    usados = set(tuple(sorted(map(int, r[bolas].values))) for _, r in df.iterrows())
    return usados

# -------- Adapters LS14 / LS15 (não quebram se modelo não estiver pronto) --------
def _score_from_ls_via_amostragem(func_gerador, amostras=200):
    """
    Se o gerador é LS14/LS15, reduz amostras para 1 (evita loops + reloads).
    Caso contrário, mantém o comportamento original.
    """
    name = getattr(func_gerador, "__name__", "").lower()
    if "ls14" in name or "ls15" in name or "gerar_palpite_ls" in name:
        amostras = 1  # 🔑 evita disparar load/predict repetidas

    cont = np.zeros(25, dtype=float)
    ok = 0
    for _ in range(max(1, amostras)):
        try:
            bet = func_gerador()  # pode devolver list ou [list]
            if isinstance(bet, (list, tuple)) and len(bet) == 1 and isinstance(bet[0], (list, tuple)):
                bet = bet[0]
            if not bet:
                continue
            for d in bet:
                di = int(d)
                if 1 <= di <= 25:
                    cont[di - 1] += 1.0
            ok += 1
        except Exception:
            continue

    if ok == 0 or cont.sum() <= 0:
        return np.zeros(25, dtype=float)
    return _normalize_probs(cont)

@lru_cache(maxsize=4)
def _score_from_ls(model_name: str):
    """
    Obtém 25 scores do LS14/LS15 via amostragem de geradores locais.
    Se falhar, retorna zeros (o chamador faz fallback para estatístico).
    """
    try:
        if model_name.upper() == "LS14" and 'gerar_palpite_ls14' in globals():
            return _score_from_ls_via_amostragem(globals()['gerar_palpite_ls14'], amostras=200)
        if model_name.upper() == "LS15" and 'gerar_palpite_ls15' in globals():
            return _score_from_ls_via_amostragem(globals()['gerar_palpite_ls15'], amostras=200)
    except Exception as e:
        print(f"⚠️ Falha ao amostrar {model_name}: {e}")

    return np.zeros(25, dtype=float)

def scores_neurais(model_name: str):
    """
    Tenta obter scores (25,) por amostragem do gerador LS.
    Não recarrega modelos N vezes (vide sampler acima).
    """
    v = _score_from_ls(model_name)
    if v.sum() <= 0 or not np.all(np.isfinite(v)):
        return np.zeros(25, dtype=float)
    return _normalize_probs(v)

def scores_ls16(model_name_for_ensemble="LS15", w_neural=0.6, w_stats=0.4):
    """
    LS16 híbrido com cache de LS15 e anti-reentrância.
    - Se já está computando LS16, cai para estatístico (evita recursão indireta).
    - Reusa _LS15_CACHE quando existir; não chama LS15 repetidas vezes.
    """
    global _IN_SCORES_LS16, _LS15_CACHE
    if _IN_SCORES_LS16:
        return scores_estatisticos()

    _IN_SCORES_LS16 = True
    try:
        # usa cache LS15 se disponível
        if _LS15_CACHE is not None and np.isfinite(_LS15_CACHE).all() and _LS15_CACHE.sum() > 0:
            s_n = _LS15_CACHE
        else:
            s_n = _score_from_ls(model_name_for_ensemble)
            if s_n is not None and np.isfinite(s_n).all() and s_n.sum() > 0:
                _LS15_CACHE = _normalize_probs(s_n)
            else:
                _LS15_CACHE = None

        s_e = scores_estatisticos()

        if _LS15_CACHE is None:
            return s_e

        s = w_neural * _LS15_CACHE + w_stats * s_e
        return _normalize_probs(s)
    except Exception as e:
        logging.error(f"[scores_ls16] Falhou: {e}")
        return scores_estatisticos()
    finally:
        _IN_SCORES_LS16 = False

# =========================
# GERAÇÃO ESTATÍSTICA
# =========================
def gerar_palpite_estatistico_paridade(limite=15):
    """Gera palpites equilibrando pares e ímpares."""
    pares = [n for n in range(1, 26) if n % 2 == 0]
    impares = [n for n in range(1, 26) if n % 2 == 1]
    random.shuffle(pares)
    random.shuffle(impares)
    qtd_pares = limite // 2
    qtd_impares = limite - qtd_pares
    dezenas = sorted(pares[:qtd_pares] + impares[:qtd_impares])
    return dezenas

def gerar_palpite_aleatorio(limite=15):
    dezenas = random.sample(range(1, 26), limite)
    return sorted(dezenas)

# ================================================================
# BLOCO FINAL: SALVAR PALPITE COM TELEMETRIA (meta JSON)
# ================================================================
# ================================================================
# SALVAR PALPITE — versão final (coluna 'numeros' padrão oficial)
# ================================================================

def salvar_palpite(palpite, modelo, extras_meta=None):
    # ✅ Validação: LS16 nunca deve salvar palpite vazio
    if not palpite or len(palpite) < 15:
        _log_warn("⚠️ Palpite LS16 inválido — não será salvo no banco.")
        return None

    """
    Salva palpite no banco (PostgreSQL/Neon), compatível com campo data_norm.
    Retorna o ID do palpite inserido.
    Corrigido para evitar erro 'sql not defined' em exceções.
    """
    usuario = st.session_state.get("usuario", {}) if 'usuario' in st.session_state else {}
    id_usuario = usuario.get("id")

    if not id_usuario:
        _log_warn("Usuário não logado — não é possível salvar palpite.")
        return None

    # 🔹 Força nome correto de modelo híbrido (Platinum → LS16)
    if extras_meta and isinstance(extras_meta, dict):
        plano = str(extras_meta.get("plano", "")).lower()
        if plano == "platinum" and not modelo.upper().startswith("LS16"):
            _log_info(f"Forçando modelo LS16 (plano={plano})")
            modelo = "LS16"

    dezenas_txt = ",".join(f"{int(d):02d}" for d in palpite)
    cols = _existing_cols("palpites")
    if not cols:
        _log_warn("Tabela palpites não encontrada.")
        return None

    # --- [1] Coluna principal ---
    if "numeros" in cols:
        num_col = "numeros"
    elif "dezenas" in cols:
        num_col = "dezenas"
    elif "palpite" in cols:
        num_col = "palpite"
    elif "jogada" in cols:
        num_col = "jogada"
    else:
        _log_warn("Nenhuma coluna de dezenas encontrada.")
        return None

    ts_col = next((c for c in ("created_at", "criado_em", "data") if c in cols), None)
    defaults_map = {"status": "N", "ativo": True, "validado": False}
    extras = {c: defaults_map[c] for c in cols if c in defaults_map}

    # --- [2] Monta meta JSON ---
    meta_payload = {}
    if "meta" in cols:
        try:
            meta_payload = {
                "modelo": modelo,
                "usuario_id": id_usuario,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "temperature": os.getenv("ENSEMBLE_TEMPERATURE", "1.0"),
                    "ensemble_weights": os.getenv("ENSEMBLE_WEIGHTS", "recent:0.5,mid:0.3,global:0.2"),
                    "use_gumbel": os.getenv("USE_GUMBEL", "0"),
                    "use_soft_constraints": os.getenv("USE_SOFT_CONSTRAINTS", "0"),
                },
                "palpite": palpite,
            }
            if extras_meta and isinstance(extras_meta, dict):
                meta_payload.update(extras_meta)
        except Exception as e:
            _log_warn(f"Erro ao montar meta JSON: {e}")

    # --- [3] SQL dinâmico ---
    db = Session()
    new_id = None
    sql = ""
    params = {}
    try:
        base_cols = ["id_usuario", "modelo", num_col]
        params = {"id_usuario": id_usuario, "modelo": modelo, num_col: dezenas_txt}

        if ts_col:
            base_cols.append(ts_col)
        for k, v in extras.items():
            if k not in base_cols:
                base_cols.append(k)
                params[k] = v

        if "meta" in cols:
            base_cols.append("meta")
            params["meta"] = json.dumps(meta_payload, ensure_ascii=False)

        # ✅ adiciona data_norm se existir
        if "data_norm" in cols:
            base_cols.append("data_norm")
            params["data_norm"] = date.today().isoformat()  # agora 'date' está importado

        placeholders = [("NOW()" if c == ts_col else f":{c}") for c in base_cols]
        sql = f"""
            INSERT INTO palpites ({', '.join(base_cols)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """
        result = db.execute(text(sql), params)
        new_id = result.scalar()
        db.commit()

        _log_info(f"✅ Palpite salvo com sucesso! ID={new_id} (modelo={modelo}, usuario={id_usuario})")
        return new_id

    except Exception as e:
        db.rollback()
        safe_sql = sql or "<não definido>"
        safe_params = params or {}
        _log_warn(f"❌ Falha ao salvar palpite: {e}\nSQL: {safe_sql}\nparams: {safe_params}")
        return None
    finally:
        db.close()

# 🔹 Normalização dos nomes de modelo para exibição amigável

# =========================
# LIMITE POR PLANO

def verificar_limite_palpites(id_usuario: int) -> tuple[bool, str]:
    """
    Verifica se o usuário ainda pode gerar palpites no mês atual,
    de acordo com seu plano (Free, Silver ou Gold).

    Retorna:
        (permitido: bool, msg: str)
    """

    db = Session()
    try:
        # --- 1️⃣ Obter plano e nome ---
        result = db.execute(text("""
            SELECT u.id, p.nome AS nome_plano
            FROM usuarios u
            LEFT JOIN planos p ON p.id = u.id_plano
            WHERE u.id = :id
        """), {"id": id_usuario}).fetchone()

        if not result:
            return False, "Usuário não encontrado."

        nome_plano = (result.nome_plano or "").lower()

        # --- 2️⃣ Definir limites mensais ---
        limites = {
            "free": 30,     # 30 palpites/mês
            "silver": 150,  # 150 palpites/mês
            "gold": 500,    # 500 palpites/mês
        }
        limite = limites.get(nome_plano, 30)

        # --- 3️⃣ Calcular período do mês atual ---
        hoje = datetime.now()
        inicio_mes = hoje.replace(day=1)
        fim_mes = hoje.replace(day=28, hour=23, minute=59, second=59)
        # OBS: pode-se usar calendar.monthrange se quiser precisão de 30/31

        # --- 4️⃣ Contar palpites já feitos ---
        q = text("""
            SELECT COUNT(*) AS total
            FROM palpites
            WHERE id_usuario = :id_usuario
              AND dt_criacao >= :inicio_mes
              AND dt_criacao <= :fim_mes
        """)
        total = db.execute(q, {
            "id_usuario": id_usuario,
            "inicio_mes": inicio_mes,
            "fim_mes": fim_mes
        }).scalar() or 0

        # --- 5️⃣ Validação ---
        if total >= limite:
            msg = f"🚫 Limite mensal atingido ({total}/{limite}) para o plano **{nome_plano.title()}**."
            return False, msg
        else:
            restante = limite - total
            msg = f"✅ Você ainda pode gerar **{restante}** palpites neste mês ({total}/{limite} usados)."
            return True, msg

    except Exception as e:
        return False, f"Erro ao validar limite: {e}"

    finally:
        db.close()

# INTERFACE (UI)
# ================================================================
# 🔧 CONFIGURAÇÃO BASE + IMPORT LAZY (TensorFlow sob demanda)

_tf = None
_tf_load_model = None
_to_binary = None

def _lazy_imports():
    """
    Importa TensorFlow e utilitários pesados apenas quando necessário.
    Evita overhead e falhas em ambientes sem TF instalado (ex: Render lite).
    """
    global _tf, _tf_load_model, _to_binary
    if _tf is not None:
        return  # já importado

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as tf_load_model
        from modelo_llm_max.utils_ls_models import to_binary

        _tf = tf
        _tf_load_model = tf_load_model
        _to_binary = to_binary

        logging.info("✅ TensorFlow importado com sucesso via _lazy_imports().")

    except Exception as e:
        logging.error(f"⚠️ TensorFlow indisponível: {e}")
        _tf = None
        _tf_load_model = None
        _to_binary = None
        # Não levanta exceção — o app continua, apenas desativa recursos ML.
        #st.warning("TensorFlow não encontrado — modelos LS14/LS15 estarão desativados.")

# ================================================================
# 📁 CONFIGURAÇÃO DE DIRETÓRIOS E LOGS

try:
    _base_file = __file__
except NameError:
    _base_file = sys.argv[0] if sys.argv and sys.argv[0] else os.getcwd()

BASE_DIR = os.environ.get(
    "FAIXABET_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(_base_file))))

DEFAULT_MODELS_DIRS = [
    os.path.join(BASE_DIR, "modelo_llm_max", "models", "prod"),
    os.path.join(BASE_DIR, "modelo_llm_max", "models"),
    os.path.join(BASE_DIR, "models"),
    os.environ.get("FAIXABET_MODELS_DIR", "")]

# Usa o primeiro caminho existente como diretório de modelos
MODELS_DIR = next((p for p in DEFAULT_MODELS_DIRS if p and os.path.isdir(p)), DEFAULT_MODELS_DIRS[0])

# Modo debug controlado via variável de ambiente DEBUG_MODE=1
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s")

logging.info(f"🧩 MODELS_DIR definido como: {MODELS_DIR}")
logging.info(f"🔧 Base dir: {BASE_DIR}")
logging.info(f"🪶 DEBUG_MODE: {DEBUG_MODE}")

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
    "plano pago x": ["recent", "mid", "global"]}

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
    Prepara X para o modelo como sequência de one-hot (25 features).
    Garante shape: (1, expected_seq_len, 25)
    - expected_seq_len: meta.get('expected_seq_len') OU inferido OU 50
    - features: SEMPRE 25 (one-hot das dezenas 1..25)
    """
    import numpy as np

    # helper local (fallback caso _to_binary não esteja disponível)
    def _one_hot_25(jogo):
        v = np.zeros(25, dtype=np.float32)
        for d in jogo:
            try:
                di = int(d)
                if 1 <= di <= 25:
                    v[di - 1] = 1.0
            except Exception:
                continue
        return v

    # 1) expected_seq_len
    expected_seq_len = meta.get("expected_seq_len")
    if not expected_seq_len:
        try:
            inferred = infer_expected_seq_from_loaded_model(meta.get("model"))
            expected_seq_len = inferred or 50
        except Exception:
            expected_seq_len = 50

    # 2) converte últimos concursos para one-hot (25)
    seq = []
    for jogo in ultimos_full:
        if "_to_binary" in globals() and _to_binary is not None:
            try:
                seq.append(np.array(_to_binary(jogo), dtype=np.float32))
                continue
            except Exception:
                pass
        # fallback
        seq.append(_one_hot_25(jogo))

    seq = np.asarray(seq, dtype=np.float32)  # shape (n, 25)

    # 3) ajusta tamanho temporal (pad/truncate)
    n = seq.shape[0]
    if n > expected_seq_len:
        seq = seq[-expected_seq_len:]  # mantém as últimas N
    elif n < expected_seq_len:
        pad = np.zeros((expected_seq_len - n, 25), dtype=np.float32)
        seq = np.vstack([pad, seq])

    # 4) retorna com batch dimension
    X = np.expand_dims(seq, axis=0)  # (1, T, 25)
    return X

def _prepare_inputs_for_model_meta(meta, ultimos_full):
    """
    Prepara X com one-hot de 25 features: (1, expected_seq_len, 25)
    """
    import numpy as np

    def _one_hot_25(jogo):
        v = np.zeros(25, dtype=np.float32)
        for d in jogo:
            try:
                di = int(d)
                if 1 <= di <= 25:
                    v[di - 1] = 1.0
            except Exception:
                continue
        return v

    expected_seq_len = meta.get("expected_seq_len") or 50

    seq = []
    for jogo in ultimos_full:
        try:
            if "_to_binary" in globals() and _to_binary is not None:
                seq.append(np.array(_to_binary(jogo), dtype=np.float32))
            else:
                seq.append(_one_hot_25(jogo))
        except Exception:
            seq.append(_one_hot_25(jogo))

    seq = np.asarray(seq, dtype=np.float32)  # (n, 25)

    if seq.shape[0] > expected_seq_len:
        seq = seq[-expected_seq_len:]
    elif seq.shape[0] < expected_seq_len:
        pad = np.zeros((expected_seq_len - seq.shape[0], 25), dtype=np.float32)
        seq = np.vstack([pad, seq])

    return np.expand_dims(seq, axis=0)  # (1, T, 25)

def _ultimos_sorteios_para_modelo(limit=50):
    try:
        db = Session()
        rows = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY data DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()
        db.close()
        if not rows:
            return []
        jogos = [[int(x) for x in row if x is not None] for row in rows]
        return list(reversed([sorted(j) for j in jogos]))  # cronológico
    except Exception as e:
        logging.warning(f"[ultimos_sorteios_para_modelo] Falhou: {e}")
        return []

# ================================================================
# BLOCO NOVO: CALIBRAÇÃO & PESOS (SOFTMAX + ENSEMBLE)
# ================================================================

def _softmax(x, temperature=1.0):
    """Aplica softmax com temperatura para calibrar probabilidades."""
    x = np.array(x, dtype=np.float32)
    x = x / float(temperature)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def _parse_weights_from_env(default="recent:0.5,mid:0.3,global:0.2"):
    """
    Lê os pesos de ensemble a partir da variável de ambiente ENSEMBLE_WEIGHTS.
    Exemplo: ENSEMBLE_WEIGHTS="recent:0.6,mid:0.3,global:0.1"
    """
    raw = os.getenv("ENSEMBLE_WEIGHTS", default)
    parts = [p.strip() for p in raw.split(",") if ":" in p]
    weights = {}
    for p in parts:
        k, v = p.split(":")
        try:
            weights[k.strip()] = float(v)
        except ValueError:
            continue

    total = sum(weights.values())
    if total <= 0:
        return {"recent": 0.5, "mid": 0.3, "global": 0.2}

    return {k: v / total for k, v in weights.items()}

def combinar_modelos_com_pesos(predicoes_por_grupo, temperature=1.0):
    """
    Recebe um dicionário {grupo: vetor_predição(25,)} e aplica:
    - Softmax (calibração)
    - Média ponderada conforme ENSEMBLE_WEIGHTS
    Retorna vetor final (25,)
    """
    weights = _parse_weights_from_env()
    grupos_validos = [g for g in predicoes_por_grupo.keys() if g in weights]
    if not grupos_validos:
        grupos_validos = list(predicoes_por_grupo.keys())
        weights = {g: 1 / len(grupos_validos) for g in grupos_validos}

    soft_preds = {}
    for g in grupos_validos:
        soft_preds[g] = _softmax(predicoes_por_grupo[g], temperature=temperature)

    # Média ponderada
    final = np.zeros_like(list(soft_preds.values())[0])
    for g in grupos_validos:
        final += weights[g] * soft_preds[g]

    # Normaliza novamente
    final /= final.sum()

    _log_info(f"[Ensemble] Grupos={grupos_validos}, Pesos={weights}, Temp={temperature:.2f}")
    return final

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

    # --- Novo ensemble calibrado ---
    temperature = float(os.getenv("ENSEMBLE_TEMPERATURE", "1.0"))

    # Determina grupo de cada modelo, se possível
    preds_por_grupo = {}
    for i, entry in enumerate(models_list):
        group = "recent"
        if isinstance(entry, dict):
            group = entry.get("group", f"m{i}")
        preds_por_grupo[group] = preds[i][0] if preds[i].ndim > 1 else preds[i]

    mean_pred = combinar_modelos_com_pesos(preds_por_grupo, temperature=temperature)
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

# ================================================================
# Utilitários para diversidade (Gumbel-top-k)

def _sample_gumbel(shape, eps=1e-20):
    """Ruído Gumbel i.i.d para amostragem Top-K."""
    U = np.random.uniform(0.0, 1.0, shape)
    return -np.log(-np.log(U + eps) + eps)

def _gumbel_top_k_sampling(probs, k=15, temperature=1.0):
    """
    Seleciona Top-K aplicando ruído Gumbel nos logits.
    - probs: vetor (25,) normalizado
    - k: número de dezenas
    - temperature: >1 aumenta diversidade; <1 deixa mais “focado”
    """
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    g = _sample_gumbel(logits.shape)
    noisy = (logits + g) / float(temperature)
    topk = np.argsort(-noisy)[:k]
    return np.sort(topk + 1).tolist()

def gerar_palpite_from_probs(
    probs,
    limite=15,
    reinforce_threshold=0.06,
    boost_factor=2.0,
    temperature=1.0,
    deterministic=False):
    """
    Gera palpites a partir de probabilidades.
    Agora com diversidade opcional (Gumbel-top-k) controlada por env USE_GUMBEL.
    Mantém compatibilidade total com chamadas existentes.
    """
    use_gumbel = os.getenv("USE_GUMBEL", "0") == "1"

    # 1) Aplica temperatura (mesma lógica existente)
    p = apply_temperature(probs, temperature)

    # 2) Reforço leve em probabilidades altas (mesma lógica existente)
    mask = p > reinforce_threshold
    if mask.any():
        p[mask] = p[mask] * boost_factor
        p = p / p.sum()

    # 3) Caminho determinístico (inalterado)
    if deterministic:
        idxs = np.argsort(-p)[:limite]
        chosen = np.sort(idxs + 1).tolist()
        return chosen

    # 4) Caminho estocástico:
    #    - se USE_GUMBEL=1 => usa Gumbel-top-k (mais diversidade)
    #    - caso contrário => amostragem atual por np.random.choice (comportamento igual ao antigo)
    if use_gumbel:
        return _gumbel_top_k_sampling(p, k=limite, temperature=temperature)
    else:
        chosen_idxs = np.random.choice(np.arange(25), size=limite, replace=False, p=p)
        return np.sort(chosen_idxs + 1).tolist()

# ================================================================
# BLOCO NOVO: DIVERSIDADE — GUMBEL-TOP-K SAMPLER (OPCIONAL)

def _sample_gumbel(shape, eps=1e-20):
    """Gera ruído Gumbel i.i.d para amostragem Top-K."""
    U = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(U + eps) + eps)

def gumbel_top_k_sampling(probs, k=15, temperature=1.0):
    """
    Retorna índices Top-K com ruído Gumbel (para mais diversidade).
    - probs: vetor (25,) normalizado
    - k: número de dezenas
    """
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    gumbels = _sample_gumbel(logits.shape)
    noisy_logits = (logits + gumbels) / float(temperature)
    topk = np.argsort(-noisy_logits)[:k]
    return np.sort(topk + 1).tolist()

def gerar_palpite_from_probs_diverso(probs, limite=15, reinforce_threshold=0.06,
                                     boost_factor=2.0, temperature=1.0,
                                     deterministic=False):
    """
    Wrapper compatível com gerar_palpite_from_probs, adicionando diversidade.
    Ativado se USE_GUMBEL=1 no ambiente.
    """
    use_gumbel = os.getenv("USE_GUMBEL", "0") == "1"

    # Reforço leve, igual ao método original
    p = np.array(probs, dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    mask = p > reinforce_threshold
    if mask.any():
        p[mask] *= boost_factor
        p /= p.sum()

    if deterministic:
        idxs = np.argsort(-p)[:limite]
        return np.sort(idxs + 1).tolist()

    if use_gumbel:
        return gumbel_top_k_sampling(p, k=limite, temperature=temperature)
    else:
        chosen_idxs = np.random.choice(np.arange(25), size=limite, replace=False, p=p)
        return np.sort(chosen_idxs + 1).tolist()

# ================================================================
# BLOCO NOVO: CONSTRAINTS SUAVES (paridade, consecutivos, distribuição)

def _score_palpite_regras(palpite):
    """
    Calcula um score baseado em regras da Lotofácil:
    - Paridade: ideal ~7 ou 8 pares
    - Consecutivos: penaliza mais de 3 seguidos
    - Distribuição 5x5: idealmente equilibrado entre 5 linhas e 5 colunas
    Retorna score ∈ [0,1], quanto maior melhor.
    """
    palpite = np.array(sorted(palpite))
    pares = np.sum(palpite % 2 == 0)
    impares = len(palpite) - pares

    # ---- regra 1: paridade ----
    par_score = 1.0 - abs(pares - 7.5) / 7.5  # máximo quando 7 ou 8 pares

    # ---- regra 2: consecutivos ----
    diffs = np.diff(palpite)
    consecutivos = np.sum(diffs == 1)
    consecutivo_score = 1.0 - min(consecutivos, 5) / 5.0  # penaliza mais de 3-4 consecutivos

    # ---- regra 3: distribuição 5x5 (linhas e colunas de 1 a 25) ----
    matriz = np.arange(1, 26).reshape(5, 5)
    lin_counts = [np.sum(np.isin(row, palpite)) for row in matriz]
    col_counts = [np.sum(np.isin(matriz[:, c], palpite)) for c in range(5)]
    dist_score = 1.0 - (
        (np.std(lin_counts) + np.std(col_counts)) / (2 * len(palpite))
    )

    # Score final (média ponderada)
    final_score = 0.4 * par_score + 0.3 * consecutivo_score + 0.3 * dist_score
    return max(0.0, min(1.0, final_score))

def ajustar_palpite_por_regras(palpite, max_tentativas=50):
    """
    Tenta melhorar o palpite com pequenas trocas até atingir score aceitável.
    Mantém a mesma quantidade de dezenas.
    """
    alvo_minimo = 0.75  # limiar de qualidade
    limite = len(palpite)
    melhor = sorted(palpite)
    melhor_score = _score_palpite_regras(melhor)

    for _ in range(max_tentativas):
        candidato = melhor.copy()
        trocas = np.random.randint(1, 4)  # até 3 trocas aleatórias
        for _ in range(trocas):
            rem = np.random.choice(candidato)
            add = np.random.choice([x for x in range(1, 26) if x not in candidato])
            candidato.remove(rem)
            candidato.append(add)
        candidato = sorted(candidato)
        sc = _score_palpite_regras(candidato)
        if sc > melhor_score:
            melhor, melhor_score = candidato, sc
        if melhor_score >= alvo_minimo:
            break

    return melhor

def aplicar_constraints_suaves(palpite):
    """
    Wrapper para aplicar constraints apenas se USE_SOFT_CONSTRAINTS=1
    """
    use_constraints = os.getenv("USE_SOFT_CONSTRAINTS", "0") == "1"
    if not use_constraints:
        return palpite
    return ajustar_palpite_por_regras(palpite)

# ============ Bloco filha ===========================================

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

# -------------------- GERADORES ML (LS16 / LS15 / LS14) --------------------

def _model_paths_for(model_name: str, models_dir: str = None):
    """
    Retorna lista de caminhos de modelos (regular, mid, global) para LS14/LS15.
    Funciona tanto em DEV quanto em PROD.
    """
    if models_dir and os.path.isdir(models_dir):
        base_dir = models_dir
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "modelo_llm_max", "models", "prod"),
            os.path.join(base_dir, "modelo_llm_max", "models"),
            os.path.join(base_dir, "models", "prod"),
            os.path.join(base_dir, "models")
        ]
        base_dir = next((p for p in candidates if os.path.isdir(p)), None)

    if not base_dir:
        logging.warning(f"[_model_paths_for] models_dir inválido: {models_dir}")
        return []

    tipos = ['recent', 'mid', 'global', 'regular']  # recent primeiro
    encontrados = []

    for tipo in tipos:
        padrao = f"{tipo}_{model_name}pp_final.keras"
        # busca direta
        caminho = os.path.join(base_dir, padrao)
        if os.path.exists(caminho):
            encontrados.append(caminho)
        else:
            # busca recursiva
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if f.lower() == padrao.lower():
                        encontrados.append(os.path.join(root, f))
                        break
                if encontrados and encontrados[-1].lower().endswith(padrao.lower()):
                    break

    logging.info(f"[_model_paths_for] Modelos encontrados para {model_name}: {encontrados}")
    return encontrados

def _safe_dir_mtime(path):
    try:
        max_m = 0.0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    t = os.path.getmtime(os.path.join(root, f))
                    if t > max_m:
                        max_m = t
                except Exception:
                    pass
        return max_m
    except Exception:
        return 0.0

def carregar_modelo_ls(model_name: str, nome_plano=None, models_dir=None):
    """
    Carrega modelos LS14/LS15 com cache por mtime da pasta.
    """
    _lazy_imports()
    if _tf_load_model is None:
        logging.warning("[carregar_modelo_ls] TensorFlow indisponível; retornando vazio.")
        return []

    # resolve base
    base = models_dir or MODELS_DIR
    mtime = _safe_dir_mtime(base)
    key = (model_name.lower(), os.path.abspath(base), mtime)

    if key in _LOADED_METAS:
        return _LOADED_METAS[key]

    caminhos = _model_paths_for(model_name, models_dir=base)
    metas = []

    for path in caminhos:
        try:
            model_obj = _tf_load_model(path, compile=False)
            # inferir expected_seq_len
            expected_seq_len = None
            try:
                if getattr(model_obj, "inputs", None):
                    shp = model_obj.inputs[0].shape  # (None, T, F)
                    if len(shp) >= 3 and int(shp[-1]) == 25:
                        expected_seq_len = int(shp[-2])
            except Exception:
                pass
            tipo = os.path.basename(path).split("_")[0]  # recent/mid/global
            metas.append({
                "model": model_obj,
                "path": path,
                "tipo": tipo,
                "expected_seq_len": expected_seq_len or 50
            })
            logging.info(f"[carregar_modelo_ls] Carregado: {path} (expected_seq_len={expected_seq_len or 50})")
        except Exception as e:
            logging.warning(f"[carregar_modelo_ls] Falha ao carregar {path}: {e}")

    _LOADED_METAS[key] = metas
    return metas

def _load_and_filter_metas_for_plan(model_name, nome_plano, models_dir=None):
    metas = carregar_ensemble_models(model_name, models_dir=models_dir)
    if not metas:
        return []
    allowed_groups = _groups_allowed_for_plan(nome_plano)
    filtered = [m for m in metas if m.get("group") in allowed_groups or m.get("group") == "unknown"]
    return filtered

def gerar_palpite_ls16_platinum(k=15):
    """
    Tenta primeiro o ensemble inteligente (modelo_llm_max/ensemble.py).
    Se não der, usa o caminho atual baseado em scores_ls16 (LS15 + estatístico).
    Retorna uma lista ordenada de dezenas (int).
    """
    # 1) Tenta o ensemble.py novo
    if _fxb_ls16_ensemble is not None:
        try:
            dezenas = _fxb_ls16_ensemble()  # retorna ['01','07',...]
            if dezenas and isinstance(dezenas, (list, tuple)):
                dz_int = sorted({int(str(d).strip()) for d in dezenas})[:k]
                dz_int = [x for x in dz_int if 1 <= x <= 25]
                if len(dz_int) >= min(15, k):
                    return sorted(dz_int[:k])
        except Exception as e:
            logging.warning(f"[LS16 ensemble] fallback por erro: {e}")

    # 2) Fallback: usar seu pipeline atual (scores_ls16 -> amostragem)
    s = scores_ls16(model_name_for_ensemble="LS15", w_neural=0.6, w_stats=0.4)
    p = _amostrar_dezenas(s, k=k)
    return sorted(p or gerar_palpite_estatistico(limite=k))

def gerar_palpite_ls15(limite=15, models_dir=None):
    """
    Versão segura: 1) usa modelos cacheados, 2) prevê uma única vez,
    3) fallback direto p/ estatístico, sem chamar gerar_palpite_ls() de novo.
    """
    _lazy_imports()
    metas = carregar_modelo_ls("ls15", None, models_dir=models_dir)
    if not metas:
        _log_warn("Nenhum modelo LS15 carregado — fallback estatístico.")
        return gerar_palpite_estatistico(limite=limite)

    # prepara janela única (50 por padrão)
    T = max((m.get("expected_seq_len") or 50) for m in metas) or 50
    ultimos = _ultimos_sorteios_para_modelo(limit=T)

    preds = []
    for meta in metas:
        try:
            X = _prepare_inputs_for_model_meta(meta, ultimos)
            p = meta["model"].predict(X, verbose=0)
            p = np.asarray(p, dtype=np.float32)
            if p.ndim > 1:
                p = p.reshape(-1)
            if p.size == 25:
                preds.append(p)
        except Exception as e:
            _log_warn(f"Predição falhou para LS15 ({meta.get('path')}): {e}")

    if not preds:
        _log_warn("Nenhuma predição válida LS15 — fallback estatístico.")
        return gerar_palpite_estatistico(limite=limite)

    mean_pred = _normalize_probs(np.mean(np.vstack(preds), axis=0))
    palpite = gerar_palpite_from_probs(mean_pred, limite=limite, deterministic=False)
    return sorted(palpite)

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
        opcoes_modelo = ["Todos", "Aleatório", "Estatístico", "Ímpares-Pares", "LS15", "LS14","LS16"]
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

# 🔹 Helper para compatibilidade de rerun
def _safe_rerun():
    """Rerun compatível com versões novas e antigas do Streamlit."""
    try:
        st.rerun()  # Streamlit >= 1.27
    except AttributeError:
        st.experimental_rerun()  # compatibilidade com versões antigas

def _normalize_probs(vec, eps=1e-12):
    """
    Normaliza vetor de pesos garantindo:
      - sem NaN/Inf
      - soma > 0
    Se falhar, retorna uniforme (25) para Lotofácil.
    """
    v = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    s = v.sum()
    if s <= 0:
        # uniforme de 25 posições (Lotofácil)
        return np.ones(25, dtype=float) / 25.0
    v = v / s
    # sanidade final
    if not np.all(np.isfinite(v)):
        return np.ones(25, dtype=float) / 25.0
    return v

# 🔹 Função para atualizar o status do palpite no banco
def _validar_no_banco(pid: int):
    db = Session()
    try:
        db.execute(text("""
            UPDATE palpites SET status = 'S' WHERE id = :pid
        """), {"pid": pid})
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao validar: {e}")
    finally:
        db.close()

# 🔹 Função principal
def validar_palpite():
    """Exibe os palpites em cards com botão de validação individual."""
    if "usuario" not in st.session_state or not st.session_state.usuario:
        st.warning("Você precisa estar logado para validar seus Palpites.")
        return

    st.markdown("### 📤 Validar Palpite")

    db = Session()
    try:
        rows = db.execute(text("""
            SELECT id, numeros, modelo, data, status
            FROM palpites
            WHERE id_usuario = :uid
            ORDER BY data DESC
            LIMIT 50
        """), {"uid": st.session_state.usuario["id"]}).fetchall()
    except Exception as e:
        st.error(f"Erro ao buscar Palpites: {e}")
        return
    finally:
        db.close()

    if not rows:
        st.info("Você ainda não gerou nenhum Palpite.")
        return

    # --- CSS grid responsiva ---
    st.markdown("""
    <style>
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 14px;
        max-height: 550px;
        overflow-y: auto;
        padding: 8px;
    }
    .palpite-card {
        background: #f9fff9;
        border: 2px solid #009900;
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }
    .palpite-header {
        font-weight: bold;
        color: #009900;
        margin-bottom: 4px;
    }
    .palpite-data {
        font-size: 13px;
        color: #555;
        margin-bottom: 6px;
    }
    .palpite-dezenas {
        font-family: monospace;
        font-size: 15px;
        color: #000;
        word-spacing: 5px;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Renderização em grid ---
    st.markdown('<div class="grid">', unsafe_allow_html=True)

    for pid, numeros, modelo, data, status in rows:
        dezenas = " ".join(f"{int(d):02d}" for d in str(numeros).replace(",", " ").split() if d.isdigit())

        # renderiza o card
        st.markdown(
            f"""
            <div class="palpite-card">
                <div class="palpite-header">#{pid} — {modelo}</div>
                <div class="palpite-data">{data.strftime('%d/%m/%Y %H:%M')}</div>
                <div class="palpite-dezenas">{dezenas}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # renderiza botão abaixo do card
        if status != "S":
            if st.button(f"✅ Validar #{pid}", key=f"val_{pid}"):
                _validar_no_banco(pid)
                st.success(f"Palpite #{pid} validado com sucesso! 🎯")
                _safe_rerun()
        else:
            st.caption("✅ Já validado")

    st.markdown('</div>', unsafe_allow_html=True)

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
                pal = aplicar_constraints_suaves(pal)  # ← NOVO
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
    mean_pred = _normalize_probs(np.clip(mean_pred, 1e-12, None))

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

# ================== [PLAN RULES] ==================

def gerar_para_plano(nome_plano: str, qtd: int = 3, k_escolhido: int = None):
    """
    nome_plano: "Free" | "Silver" | "Gold" | "Platinum"
    k_escolhido: número de dezenas solicitado (respeita faixa do plano)
    """
    nome = (nome_plano or "").strip().lower()

    if nome == "free":
        k = 15
        modelo = "ESTAT"

    elif nome == "silver":
        k = k_escolhido or 15
        k = max(15, min(17, k))
        modelo = "LS14"

    elif nome == "gold":
        k = k_escolhido or 15
        k = max(15, min(20, k))
        modelo = "LS15"

    elif nome == "platinum":
        # para você testar internamente agora
        k = k_escolhido or 15
        k = max(15, min(20, k))
        modelo = "LS16"

    else:
        # default seguro
        k = 15
        modelo = "ESTAT"

    palpites = gerar_palpites_por_modelo(modelo=modelo, qtd=qtd, k=k)
    return modelo, k, palpites

# ================== [GENERATOR CORE] ==================
def _amostrar_dezenas(scores_25, k=15, correl_steps=10, pares_range=(6,9)):
    usados = combinacoes_ja_sorteadas()
    dezenas = list(range(1, 26))

    base_weights = _normalize_probs(scores_25)

    for _ in range(5000):
        base = set(random.choices(dezenas, weights=base_weights, k=min(5, k)))
        atual = set(base)

        for _step in range(correl_steps):
            if len(atual) >= k:
                break
            aff = scores_correlacao(atual)
            mixed = _normalize_probs(0.5*aff + 0.5*base_weights)
            prox = random.choices(dezenas, weights=mixed, k=1)[0]
            atual.add(prox)

        while len(atual) < k:
            prox = random.choices(dezenas, weights=base_weights, k=1)[0]
            atual.add(prox)

        comb = sorted(atual)
        if tuple(comb) in usados:
            continue
        pares = sum(1 for d in comb if d % 2 == 0)
        if not (pares_range[0] <= pares <= pares_range[1]):
            continue
        return comb

    return None

def gerar_palpites_por_modelo(modelo: str, qtd: int = 3, k: int = 15):
    """
    Gera palpites conforme o modelo especificado.
    - LS14 / LS15 usam redes neurais locais.
    - LS16 tenta o ensemble inteligente (modelo_llm_max/ensemble.py) e,
      se falhar, faz fallback para scores_ls16.
    - Demais modelos usam o método estatístico.
    """
    modelo = (modelo or "ESTAT").upper()

    # -----------------------------
    # LS14 → Rede neural pura
    # -----------------------------
    if modelo == "LS14":
        s = scores_neurais("LS14")
        if s.sum() == 0:
            s = scores_estatisticos()

    # -----------------------------
    # LS15 → Rede neural pura
    # -----------------------------
    elif modelo == "LS15":
        s = scores_neurais("LS15")
        if s.sum() == 0:
            s = scores_estatisticos()

    # -----------------------------
    # LS16 → Ensemble inteligente (Platinum)
    # -----------------------------
    elif modelo == "LS16":
        try:
            if _fxb_ls16_ensemble is None:
                raise RuntimeError("ensemble LS16 indisponível (import falhou).")

            palpites = _fxb_ls16_ensemble()

            # Exibir no Streamlit (se disponível)
            #if "st" in globals():
            #    st.markdown("### Palpites LS16 (Platinum — Ensemble com Diversidade)")
            #    for i, p in enumerate(palpites, 1):
            #        dezenas_txt = " ".join(f"{x:02d}" for x in p)
            #        st.markdown(f"**Palpite {i:02d}:** {dezenas_txt}")
            #    st.info(f"📦 Total gerado: {len(palpites)} palpites diversificados")
            #
            return [list(map(int, p)) for p in palpites]

        except Exception as e:
            _log_warn(f"[LS16 ensemble] Falhou → fallback p/ LS15: {e}")

            # 🔸 Fallback: mistura LS15 + estatístico
            palpites = []
            vistos = set()
            faltam = qtd
            s = scores_ls16(model_name_for_ensemble="LS15", w_neural=0.6, w_stats=0.4)
            tries = 0
            while len(palpites) < faltam and tries < 8000:
                tries += 1
                p = _amostrar_dezenas(s, k=k)
                if not p:
                    continue
                tp = tuple(sorted(p))
                if tp in vistos:
                    continue
                vistos.add(tp)
                palpites.append(p)

            return palpites

    # -----------------------------
    # Qualquer outro modelo → Estatístico
    # -----------------------------
    else:
        s = scores_estatisticos()

    # -----------------------------
    # Geração de palpites padrão (LS14, LS15 ou Estatístico)
    # -----------------------------
    palpites = []
    tent = 0
    while len(palpites) < qtd and tent < 8000:
        p = _amostrar_dezenas(s, k=k)
        if p:
            palpites.append(p)
        tent += 1

    return palpites

# ---------- [GENERATOR] unificado por plano ----------
#Gerador de palpites com regras por plano (Free/Silver/Gold/Platinum) 23/10/2025

@lru_cache(maxsize=4)
def _score_from_ls(model_name: str):
    """
    Obtém 25 scores do modelo neural via amostragem:
    1) Chama o gerador LS14/LS15 local N vezes e calcula frequência por dezena.
    2) Se falhar, retorna zeros (cai no estatístico/ensemble).
    """
    try:
        # usa as funções locais se já existem neste módulo:
        if model_name.upper() == "LS14" and 'gerar_palpite_ls14' in globals():
            return _score_from_ls_via_amostragem(globals()['gerar_palpite_ls14'], amostras=200)

        if model_name.upper() == "LS15" and 'gerar_palpite_ls15' in globals():
            return _score_from_ls_via_amostragem(globals()['gerar_palpite_ls15'], amostras=200)

    except Exception as e:
        print(f"⚠️ Falha ao amostrar {model_name}: {e}")

    # fallback neutro → força uso do estatístico/ensemble
    return np.zeros(25, dtype=float)

# 23/10 sera q estamos usanso esse def : gerar_palpites_por_modelo - apaguei PRECISO VOLTAR

# ---------- [PLAN RULES] — Free / Silver / Gold / Platinum ----------
def _modelo_e_k_por_plano(nome_plano: str, k_escolhido: int | None):
    nome = (nome_plano or "").strip().lower()
    if nome == "free":
        return "ESTAT", 15
    elif nome == "silver":
        k = k_escolhido or 15
        return "LS14", max(15, min(17, k))
    elif nome == "gold":
        k = k_escolhido or 15
        return "LS15", max(15, min(20, k))
    elif nome == "platinum":
        k = k_escolhido or 15
        return "LS16", max(15, min(20, k))
    else:
        return "ESTAT", 15

def gerar_para_plano(nome_plano: str, qtd: int = 3, k_escolhido: int | None = None):
    modelo, k = _modelo_e_k_por_plano(nome_plano, k_escolhido)
    palpites = gerar_palpites_por_modelo(modelo=modelo, qtd=qtd, k=k)
    return modelo, k, palpites

def _produto_copy_modelo(modelo_key: str) -> tuple[str, str]:
    """
    Retorna (nome_exibicao, copy_curta) para o modelo.
    """
    m = (modelo_key or "").upper()
    if m in ("ALEATÓRIO", "ALEATORIO", "RANDOM"):
        return ("Aleatório", "Combinações aleatórias equilibradas.")
    if m in ("ESTAT", "ESTATÍSTICO", "ESTATISTICO"):
        return ("Estatístico Inteligente", "Frequência histórica + recência + correlação (coocorrência).")
    if m == "LS14":
        return ("IA Neural v14", "Padrões avançados aprendidos com janela reduzida.")
    if m == "LS15":
        return ("IA Neural v15", "Padrões avançados + robustez temporal (ensemble).")
    if m == "LS16":
        return ("IA Híbrida LS16", "Ensemble Neural + Estatística (o melhor dos dois mundos).")
    return (modelo_key, "")

def _render_badge_modelo(modelo_key: str, k_final: int | None = None):
    nome, copy = _produto_copy_modelo(modelo_key)
    k_txt = f" · {k_final} dezenas" if k_final else ""
    st.markdown(f"""
    <div style="margin:8px 0 12px 0; padding:8px 12px; border:1px solid #0a0; border-radius:10px; background:#f5fff5;">
        <div style="font-weight:700; color:#0a0;">{nome}{k_txt}</div>
        <div style="font-size:13px; color:#333;">{copy}</div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# Modal FaixaBet — versão estável e funcional (sem congelar tela)
# ================================================================

def _normalize_numbers(x):
    out = []
    if isinstance(x, (list, tuple)):
        for v in x:
            out.extend(_normalize_numbers(v))
        return out
    if isinstance(x, str):
        return [int(n) for n in re.findall(r"\d+", x)]
    try:
        return [int(x)]
    except Exception:
        return out

def _extract_id_and_numbers(item):
    pid = None
    dezenas = []
    if isinstance(item, dict):
        for k in ("id", "id_palpite", "id_registro", "pk", "id_palpites"):
            if k in item and item[k]:
                pid = str(item[k])
                break
        for k in ("numeros", "dezenas", "palpite", "jogada"):
            if k in item and item[k]:
                dezenas = _normalize_numbers(item[k])
                break
    elif isinstance(item, (list, tuple)):
        for v in item:
            dezenas.extend(_normalize_numbers(v))
        if dezenas and isinstance(item[0], (int, str)) and not isinstance(item[0], list):
            pid = str(item[0])
    elif isinstance(item, str):
        dezenas = _normalize_numbers(item)
    else:
        dezenas = _normalize_numbers(item)
    id_str = f" · id {pid}" if pid else ""
    return id_str, dezenas

def exibir_modal_palpites(palpites_gerados):
    cards = []
    for i, item in enumerate(palpites_gerados or [], start=1):
        id_str, dezenas_list = _extract_id_and_numbers(item)
        dezenas_txt = " ".join(f"{int(d):02d}" for d in dezenas_list)
        cards.append(f"""
        <div class="palpite-card">
            <div class="palpite-num">#{i}{id_str}</div>
            <div class="palpite-dezenas">{dezenas_txt}</div>
        </div>
        """)
    cards_html = "".join(cards) or "<div style='opacity:.7'>Sem palpites.</div>"

    rows = math.ceil(max(1, len(palpites_gerados or [1])) / 3)
    fallback_height = max(400, min(1100, 240 + rows * 100))

    html = f"""


<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <style>
            :root {{
            --brand: #07a507;
            --bg-dim: rgba(0,0,0,.45);
            }}
            html, body {{
            margin:0; padding:0; font-family: system-ui, 'Segoe UI', Roboto, Arial, sans-serif;
            background: transparent;
            }}
            .overlay {{
            position: fixed; inset:0; background: var(--bg-dim);
            display:flex; align-items:center; justify-content:center;
            }}
            .modal {{
            position: relative;
            width: min(940px, 94vw);
            max-height: 88vh; overflow:auto;
            background:#fff; border:2px solid var(--brand);
            border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.25);
            padding:20px 22px;
            }}
            .close {{
            position:absolute; top:10px; right:14px;
            border:0; background:none; color:var(--brand);
            font-size:24px; cursor:pointer;
            }}
            h3 {{ text-align:center; color:var(--brand); margin:6px 0 10px; }}
            .msg {{ text-align:center; margin:4px 0 14px; font-weight:600; }}
            .grid {{
            display:grid; gap:12px;
            grid-template-columns: repeat(auto-fit, minmax(260px,1fr));
            }}
            .palpite-card {{
            background:#eaffe9; border:2px solid var(--brand);
            border-radius:12px; padding:8px 10px;
            word-wrap:break-word; overflow-wrap:break-word;
            }}
            .palpite-num {{ color:var(--brand); font-weight:700; margin-bottom:4px; }}
            .palpite-dezenas {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size:14px; color:#000; line-height:1.4;
            white-space:normal;
            }}
            .tip {{
            margin-top:14px; text-align:center; color:#555; font-size:13px;
            }}
        </style>
    </head>

    <body>
        <div class="overlay" id="fxb-overlay">
        <div class="modal" role="dialog" aria-modal="true">
            <button class="close" id="fxb-close" title="Fechar">✕</button>
            <h3>Palpites Gerados</h3>
            <div class="msg">Hoje é o dia certo para apostar!</div>
            <div class="grid">{cards_html}</div>
            <div class="tip">💡 Selecione e copie os números (Ctrl+C) para colar onde quiser.</div>
        </div>
        </div>
        <script>
            (function(){{
            const iframe = window.frameElement;
            function lockScroll(lock){{
                const pd = window.parent.document;
                const ht = pd.documentElement, bd = pd.body;
                if(lock){{ ht.style.overflow='hidden'; bd.style.overflow='hidden'; }}
                else{{ ht.style.overflow=''; bd.style.overflow=''; }}
            }}
            function closeModal(){{
                if(!iframe) return;
                lockScroll(false);
                iframe.remove();
            }}
            if(iframe){{
                iframe.style.position='fixed';
                iframe.style.top='0'; iframe.style.left='0';
                iframe.style.width='100vw'; iframe.style.height='100vh';
                iframe.style.zIndex='999999'; iframe.style.background='transparent';
                lockScroll(true);
                document.getElementById('fxb-close').onclick=closeModal;
                document.getElementById('fxb-overlay').onclick=(e)=>{{ if(e.target.id==='fxb-overlay') closeModal(); }};
                window.onkeydown=(e)=>{{ if(e.key==='Escape') closeModal(); }};
            }}
            }})();
        </script>
    </body>
</html>
    """
    components.html(html, height=fallback_height, scrolling=False)

# -------------------- UI / CORE - GENERATION --------------------
def gerar_palpite_ui():
    st.title("Gerar Bets - Lotofácil    ")
    _lazy_imports()  # importa módulos pesados sob demanda

    # Verifica login
    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado para gerar Bets.")
        return

    id_usuario = st.session_state.usuario.get("id")
    nome_plano = st.session_state.usuario.get("nome_plano")

    # Busca plano do banco se não estiver no session_state
    if not nome_plano:
        db = Session()
        try:
            result = db.execute(
                text("SELECT nome FROM planos WHERE id = :id_plano"),
                {"id_plano": st.session_state.usuario.get("id_plano", 0)}
            )
            row = result.fetchone()
            nome_plano = row[0] if row else None
        except Exception as e:
            st.error(f"Erro ao obter nome do plano: {e}")
            logging.error(f"[gerar_palpite_ui] Erro ao buscar plano do usuário: {e}")
            return
        finally:
            db.close()

    if not nome_plano:
        st.error("Plano do usuário não encontrado. Contate o suporte.")
        logging.warning("[gerar_palpite_ui] Plano do usuário não encontrado")
        return

    st.markdown(
    f"""
    <div style="font-size:19px; font-weight:bold; color:#222;">
        Plano atual: <span style="color:#009900;">{nome_plano}</span>
    </div>
    """,
    unsafe_allow_html=True)
    # Verifica limite de palpites
    permitido, nome_plano_verif, palpites_restantes = verificar_limite_palpites(id_usuario)
    if not permitido:
        st.error(f"Você atingiu o limite de palpites do Plano {nome_plano_verif} para este mês.")
        logging.info(f"[gerar_palpite_ui] Limite de palpites atingido para usuário {id_usuario}")
        return

    # Define modelos e limites de dezenas por plano
        # --- Definição de planos e modelos ---
    # --- Definição de planos e modelos ---
        # --- Definição de planos e modelos ---
    plano_clean = nome_plano.lower()

    if plano_clean == "free":
        modelos_disponiveis = ["Aleatório", "Estatístico"]
        min_dezenas, max_dezenas = 15, 15

    elif plano_clean == "silver":
        modelos_disponiveis = ["Estatístico", "Pares/Ímpares", "LS14"]
        min_dezenas, max_dezenas = 15, 17

    elif plano_clean == "gold":
        modelos_disponiveis = ["Estatístico", "Pares/Ímpares", "LS14", "LS15"]
        min_dezenas, max_dezenas = 15, 20

    elif plano_clean == "platinum":
        # 🔮 novo plano usando LS16
        modelos_disponiveis = ["LS16 (Híbrido AI)"]
        min_dezenas, max_dezenas = 15, 20

    else:
        modelos_disponiveis = ["Aleatório"]
        min_dezenas, max_dezenas = 15, 15

    # --- Seletores UI ---
    modelo = st.selectbox("Modelos disponíveis:", modelos_disponiveis, key="select_modelo")
    qtde_dezenas = st.number_input(
        "Quantas dezenas por palpite?",
        min_value=min_dezenas, max_value=max_dezenas,
        value=min_dezenas, step=1, key="qtde_dezenas"
    )
    num_palpites = st.number_input(
        "Quantos palpites deseja gerar?",
        min_value=1, max_value=max(1, palpites_restantes),
        value=1, step=1, key="num_palpites"
    )

    clicked = st.button("Gerar Palpites", key="btn_gerar_palpites_ui")
    if not clicked:
        return

    palpites_gerados = []
    tentativas = set()
    try:
        if plano_clean == "platinum" or modelo.startswith("LS16"):
            # 🔥 novo fluxo híbrido: usa nossa API por plano
            modelo_usado, k_final, bets = gerar_para_plano("Platinum", qtd=int(num_palpites), k_escolhido=int(qtde_dezenas))
            _render_badge_modelo(modelo_usado, k_final)

            for palpite in bets:
                if not palpite:
                    continue
                tpl = tuple(palpite)
                if tpl not in tentativas:
                    tentativas.add(tpl)
                    palpite_id = salvar_palpite(palpite, modelo_usado)
                    atualizar_contador_palpites(id_usuario)
                    palpites_gerados.append({"id": palpite_id, "numeros": palpite})

        else:
            # ⚙️ fluxo original preservado para Free/Silver/Gold
            ls14_prior = []
            if plano_clean == "gold" and modelo == "LS15":
                ls14_prior_list = gerar_palpite_ls(
                    "ls14", limite=15, n_palites=1, nome_plano=nome_plano, models_dir=MODELS_DIR
                )
                ls14_prior = ls14_prior_list[0] if ls14_prior_list else []

            # mostra badge coerente com modelo escolhido
            modelo_chave = "ESTAT" if modelo == "Estatístico" else ("LS14" if modelo == "LS14" else ("LS15" if modelo == "LS15" else "Aleatório"))
            _render_badge_modelo(modelo_chave, int(qtde_dezenas))

            for _ in range(int(num_palpites)):
                if modelo == "Aleatório":
                    palpite = sorted(random.sample(range(1, 26), int(qtde_dezenas)))
                elif modelo == "Estatístico":
                    palpite = gerar_palpite_estatistico(limite=int(qtde_dezenas))
                elif modelo == "Pares/Ímpares":
                    palpite = gerar_palpite_pares_impares(limite=int(qtde_dezenas))
                elif modelo == "LS14":
                    palpite_list = gerar_palpite_ls(
                        "ls14", limite=int(qtde_dezenas), n_palites=1,
                        nome_plano=nome_plano, models_dir=MODELS_DIR
                    )
                    palpite = palpite_list[0] if palpite_list else sorted(random.sample(range(1, 26), int(qtde_dezenas)))
                elif modelo == "LS15":
                    palpite_list = gerar_palpite_ls(
                        "ls15", limite=int(qtde_dezenas), n_palites=1,
                        nome_plano=nome_plano, models_dir=MODELS_DIR
                    )
                    palpite = palpite_list[0] if palpite_list else sorted(random.sample(range(1, 26), int(qtde_dezenas)))

                    if ls14_prior:
                        intersec = list(set(palpite) & set(ls14_prior))
                        extras = [x for x in ls14_prior if x not in intersec]
                        for x in extras:
                            if len(intersec) < int(qtde_dezenas):
                                intersec.append(x)
                        for x in range(1, 26):
                            if len(intersec) < int(qtde_dezenas) and x not in intersec:
                                intersec.append(x)
                        palpite = sorted(intersec[:int(qtde_dezenas)])

                tpl = tuple(palpite)
                if tpl not in tentativas:
                    tentativas.add(tpl)
                    palpite_id = salvar_palpite(palpite, modelo if modelo in ("Aleatório", "Pares/Ímpares", "Estatístico") else modelo)
                    atualizar_contador_palpites(id_usuario)
                    palpites_gerados.append({"id": palpite_id, "numeros": palpite})
    except Exception as e:
        st.error(f"Erro ao gerar palpite: {e}")
        logging.error(f"[gerar_palpite_ui] Falha: {e}")
        return
    # --- saída visual (igual) ---
    if palpites_gerados:
        st.success(f"{len(palpites_gerados)} palpites gerados com sucesso!")
        exibir_modal_palpites(palpites_gerados)
    else:
        st.warning("Nenhum palpite foi gerado.")
