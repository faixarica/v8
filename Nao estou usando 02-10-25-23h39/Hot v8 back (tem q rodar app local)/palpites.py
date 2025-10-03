# -------------------- [1] IMPORTS -------------------- code after at 12/08/2025 (new models)
import streamlit as st
import random
from datetime import datetime
import pyperclip 
import os
import numpy as np
from sqlalchemy import text
from db import Session
from sqlalchemy import text
from db import Session
# imports necessários no topo do palpites.py
from tensorflow.keras.models import load_model

# -------------------- [2] CONFIGS --------------------
# -------------------- [3] DEFINIÇÃO DE FUNÇÕES --------------------

# 1. Limites e plano do usuário
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

        if usados_dia >= palpites_dia:
            return False, nome_plano, 0
        if usados_mes >= limite_mes:
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
        usuario = st.session_state.usuario
        id_usuario = usuario["id"]
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

# 2. Modelos tradicionais
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
            raise ValueError("Nenhum resultado encontrado.")

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

    except Exception as e:
        st.error(f"Erro em gerar_palpite_estatistico: {e}")
        return gerar_palpite_aleatorio(limite)
    finally:
        db.close()

# -------------------- [2] CONFIGS ----- 09/09/25---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- [3] FUNÇÕES DE MODELOS --------------------

@st.cache_resource
def carregar_modelo_ls14(path=os.path.join(BASE_DIR, "modelo_ls14pp.keras")):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        alt_saved = path.replace(".keras", "") + "_saved"
        if os.path.isdir(alt_saved):
            return load_model(alt_saved, compile=False)
        st.error(f"Erro ao carregar modelo LS14++: {e}")
        return None

@st.cache_resource
def carregar_modelo_ls15(path=os.path.join(BASE_DIR, "modelo_ls15pp.keras")):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        alt_saved = path.replace(".keras", "") + "_saved"
        if os.path.isdir(alt_saved):
            return load_model(alt_saved, compile=False)
        st.error(f"Erro ao carregar modelo LS15++: {e}")
        return None

# -------------------- [4] RESTANTE DO CÓDIGO ORIGINAL --------------------
# Aqui você mantém todo o restante do seu palpites.py original sem mudanças,
# incluindo funções de limite, salvar palpite, geração aleatória/estatística,
# gerar_palpite_ls15, gerar_palpite_ls14, gerar_palpite, historico_palpites, etc.

# A única modificação necessária foi nos carregadores de modelos LS14 e LS15,
# que agora tentam carregar também a pasta `_saved` caso o `.keras` falhe.













def _calc_features_from_window(ultimos):  # ultimos = lista de jogos (cada um com 15 dezenas)
    # binários por concurso
    seq_bin = np.array([to_binary(j) for j in ultimos], dtype=np.float32)  # (window,25)
    window = len(ultimos)

    # frequência (proporção de aparecimentos na janela)
    freq_vec = seq_bin.sum(axis=0) / float(window)

    # atraso normalizado
    atraso_vec = np.zeros(25, dtype=np.float32)
    for d in range(1, 26):
        atraso = 0
        for jogo in reversed(ultimos):
            atraso += 1
            if d in jogo:
                break
        atraso_vec[d-1] = min(atraso, window) / float(window)

    # global (use o último jogo como proxy)
    last = ultimos[-1]
    soma = sum(last) / (25.0 * 15.0)
    pares = sum(1 for x in last if x % 2 == 0) / 15.0
    global_vec = np.array([soma, pares], dtype=np.float32)

    return seq_bin, freq_vec.astype(np.float32), atraso_vec.astype(np.float32), global_vec

def to_binary(jogo):
    b = [0] * 25
    for n in jogo:
        try:
            idx = int(n) - 1
        except Exception:
            continue  # ignora lixo
        if 0 <= idx < 25:
            b[idx] = 1
        # se quiser, logue os inválidos aqui
    return b

# 4. LS15
def gerar_palpite_ls15(limite=15, window=50):
    db = Session()
    try:
        resultados = db.execute(
            text("""
                SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso DESC
                LIMIT :lim
            """),
            {"lim": window}
        ).fetchall()

        if len(resultados) < window:
            raise ValueError("Histórico insuficiente para LS15.")

        ultimos = [list(map(int, r)) for r in reversed(resultados)]
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)

        modelo = carregar_modelo_ls15()
        if modelo is None:
            raise ValueError("Modelo LS15 não carregado.")

        # Detecta quantas entradas o modelo espera
        n_inputs = len(getattr(modelo, "inputs", [])) or 1

        if n_inputs == 1:  # LS15 "simples"
            pred = modelo.predict(seq_bin[None, ...], verbose=0)[0]
        elif n_inputs == 4:  # LS15++
            pred = modelo.predict(
                [seq_bin[None, ...], freq_vec[None, :], atraso_vec[None, :], global_vec[None, :]],
                verbose=0
            )[0]
        else:
            raise ValueError(f"LS15: modelo com {n_inputs} entradas não suportado.")

        # safety: garantir vetor de 25
        if len(pred) < 25:
            buf = np.zeros(25, dtype=np.float32)
            buf[:len(pred)] = pred
            pred = buf

        chosen = np.argsort(pred)[-limite:] + 1
        return sorted(chosen.tolist())

    finally:
        db.close()


# 5. LS14 híbrido
def gerar_palpite_ls14(limite=15, window=50):
    db = Session()
    try:
        resultados = db.execute(
            text("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso DESC
                LIMIT :lim
            """),
            {"lim": window}
        ).fetchall()

        if len(resultados) < window:
            raise ValueError("Histórico insuficiente para LS14.")

        # ordem cronológica
        resultados = list(reversed(resultados))
        ultimos = [list(map(int, r[1:])) for r in resultados]  # só as dezenas
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)

        # estimativa de 'hist_input' (repetições) para o PRÓXIMO sorteio
        # usa interseção entre os dois últimos como proxy [0..1]
        if len(ultimos) >= 2:
            rep_est = len(set(ultimos[-1]).intersection(set(ultimos[-2]))) / 15.0
        else:
            rep_est = 0.6  # fallback neutro
        hist_input = np.array([[rep_est]], dtype=np.float32)

        modelo = carregar_modelo_ls14()
        if modelo is None:
            raise ValueError("Modelo LS14 não carregado.")

        n_inputs = len(getattr(modelo, "inputs", [])) or 2

        if n_inputs == 2:  # LS14 híbrido (seq + hist)
            pred = modelo.predict([seq_bin[None, ...], hist_input], verbose=0)[0]
        elif n_inputs == 5:  # LS14++ (seq + hist + freq + atraso + global)
            pred = modelo.predict(
                [seq_bin[None, ...], hist_input, freq_vec[None, :], atraso_vec[None, :], global_vec[None, :]],
                verbose=0
            )[0]
        else:
            raise ValueError(f"LS14: modelo com {n_inputs} entradas não suportado.")

        if len(pred) < 25:
            buf = np.zeros(25, dtype=np.float32)
            buf[:len(pred)] = pred
            pred = buf

        chosen = np.argsort(pred)[-limite:] + 1
        return sorted(chosen.tolist())

    except Exception as e:
        st.error(f"Erro ao gerar palpite LS14: {e}")
        return []
    finally:
        db.close()

# FIM NEW MODELS
# util: transforma lista de últimos concursos em array (janela,25)
def montar_entrada_binaria(ultimos_concursos):
    # ultimos_concursos: lista de listas de 15 numeros, já na ordem mais antiga->recente
    arr = np.array([[1 if (i+1) in jogo else 0 for i in range(25)] for jogo in ultimos_concursos], dtype=np.float32)
    return arr  # shape (window,25)

# temperature scaling
def apply_temperature(p, T=1.0):
    p = np.clip(p, 1e-12, 1.0)
    logits = np.log(p)
    scaled = np.exp(logits / T)
    return scaled / scaled.sum()

# função geral de geração (usada por LS15)
def gerar_palpite_from_probs(probs, limite=15, reinforce_threshold=0.06, boost_factor=2.0, temperature=1.0, deterministic=False):
    # probs: vetor (25,) com probabilidades
    p = apply_temperature(probs, temperature)
    # boost números acima do limiar
    mask = p > reinforce_threshold
    if mask.any():
        p[mask] = p[mask] * boost_factor
        p = p / p.sum()
    if deterministic:
        idxs = np.argsort(p)[-limite:]
        chosen = np.sort(idxs + 1).tolist()
        return chosen
    else:
        chosen_idxs = np.random.choice(np.arange(25), size=limite, replace=False, p=p)
        return np.sort(chosen_idxs + 1).tolist()

## --------------------------------------------- CORE -----------------------------------------------------------
def gerar_palpite():
    st.title("Gerar Bets")
    db = Session()
    try:
        id_plano = st.session_state.usuario.get("id_plano", 0)
        result = db.execute(text("SELECT nome FROM planos WHERE id = :id_plano"), {"id_plano": id_plano})
        row = result.fetchone()
        nome_plano = row[0] if row else "Desconhecido"
    finally:
        db.close()
    # Mostrar o plano atual do usuário
    if "usuario" in st.session_state and st.session_state.usuario:
        st.markdown(f"""
        <div style='font-family: "Poppins", sans-serif; font-size:16px; color:#0b450b; margin-bottom: 20px;'>
            Plano atual: <strong>{nome_plano}</strong>
        </div>
    """, unsafe_allow_html=True)

        if 'usuario' not in st.session_state or st.session_state.usuario is None:
            st.error("Você precisa estar logado para gerar palpites.")
            return

        id_usuario = st.session_state.usuario["id"]

        try:
            permitido, nome_plano, palpites_restantes = verificar_limite_palpites(id_usuario)
            if not permitido:
                if nome_plano in ["Plano não encontrado", "Erro DB", "Erro"]:
                    st.error(f"Erro ao verificar seu plano: {nome_plano}")
                else:
                    st.error(f"Você atingiu o Limite de Palpites do Plano {nome_plano} para este mês.")
                return

            limite_dezenas = obter_limite_dezenas_por_plano(nome_plano)

            # Modelos disponíveis
            modelos_disponiveis = ["Aleatório", "Estatístico", "Pares/Ímpares"]
            if nome_plano in ["Silver", "Gold", "Plano Pago X"]:
                modelos_disponiveis += ["LS15", "LS14"]

            modelo = st.selectbox("Modelo de Geração:", modelos_disponiveis)

            num_palpites = st.number_input(
                "Quantos Palpites Deseja Gerar?",
                min_value=1,
                max_value=max(1, palpites_restantes),
                value=1,
                step=1
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
                step=1
            )

            if st.button("Gerar Palpites"):
                palpites_gerados = []

                try:
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
                    st.error(f"Erro crítico ao iniciar a geração de palpites: {str(e)}")

        except Exception as e:
            st.error(f"Erro geral ao preparar o gerador de palpites: {e}")

## ------------------------------------------FIM CORE --------------------------------------------------------
def gerar_palpite():
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
            for i in range(0, len(palpites), 2):  # 2 colunas por linha
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
            st.rerun()
        except Exception as e:
            db.rollback()
            st.error(f"Erro ao validar palpite: {e}")
        finally:
            db.close()

    # Exibir cards com status
    st.markdown("---")
    st.markdown("### Seus últimos Palpites:")

    for i in range(0, len(palpites), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(palpites):
                pid, numeros, modelo, data, status = palpites[i + j]
                status_texto = "✅ Validado" if status == "S" else "⏳ Não validado"
                cor_status = "#28a745" if status == "S" else "#666"
                bg = "#e9f7ef" if status == "S" else "#f8f8f8"

                with cols[j]:
                    data_str = data.strftime('%d/%m/%Y %H:%M') if hasattr(data, "strftime") else str(data)
                    st.markdown(f"""
                        <div style='background:{bg}; padding:10px; border-radius:8px; border:1px solid #ccc; margin-bottom:10px'>
                            <div style='font-size:13px; color:{cor_status}; font-weight:bold;'>{status_texto}</div>
                            <div style='font-size:12px; color:#888; font-weight:bold;'>
                                Palpite <span style="color:#000;">#{pid}</span> | {modelo} | {data_str}
                            </div>
                            <div style='font-family: monospace; font-size:14px; margin-top:4px;'>{numeros}</div>
                        </div>
                    """, unsafe_allow_html=True)
def salvar_palpite(palpite, modelo):
    db = Session()
    try:
        id_usuario = st.session_state.usuario["id"]
        numeros = ",".join(map(str, palpite)) if isinstance(palpite, list) else palpite
        data_hoje = datetime.now()

        db.execute(text("""
            INSERT INTO palpites (id_usuario, numeros, modelo, data, status)
            VALUES (:id_usuario, :numeros, :modelo, :data, :status)
        """), {
            "id_usuario": id_usuario,
            "numeros": numeros,
            "modelo": modelo,
            "data": data_hoje,
            "status": "N"  # N = Não validado
        })

        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao salvar palpite: {e}")
    finally:
        db.close()