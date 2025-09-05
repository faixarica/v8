# -------------------- [1] IMPORTS --------------------
import streamlit as st
import random
from datetime import datetime
import os
import numpy as np
from sqlalchemy import text
from db import Session
from tensorflow.keras.models import load_model

# -------------------- [2] CONFIGS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- [3] FUNÇÕES --------------------

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
            SET palpites_dia_usado = COALESCE(palpites_dia_usado,0)+1
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
        numeros = ",".join(map(str, palpite)) if isinstance(palpite, list) else str(palpite)
        db.execute(text("""
            INSERT INTO palpites (id_usuario, numeros, modelo, data, status)
            VALUES (:id_usuario, :numeros, :modelo, NOW(), 'N')
        """), {"id_usuario": id_usuario, "numeros": numeros, "modelo": modelo})
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao salvar palpite: {e}")
    finally:
        db.close()

# 2. Modelos básicos
def gerar_palpite_aleatorio(limite=15):
    return sorted(random.sample(range(1,26), limite))

def gerar_palpite_pares_impares(limite=15):
    num_pares = limite // 2
    num_impares = limite - num_pares
    pares = random.sample(range(2,26,2), num_pares)
    impares = random.sample(range(1,26,2), num_impares)
    return sorted(pares + impares)

def gerar_palpite_estatistico(limite=15):
    db = Session()
    try:
        resultados = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
            FROM resultados_oficiais
        """)).fetchall()
        if not resultados:
            raise ValueError("Nenhum resultado encontrado")

        todos_numeros = [num for row in resultados for num in row]
        freq = {num: todos_numeros.count(num) for num in range(1,26)}
        numeros_ordenados = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_10 = [num for num,_ in numeros_ordenados[:10]]
        outros = [num for num,_ in numeros_ordenados[10:20]]
        baixa = [num for num,_ in numeros_ordenados[20:]]
        palpite = random.sample(top_10, min(7,len(top_10))) + \
                  random.sample(outros, min(5,len(outros))) + \
                  random.sample(baixa, min(3,len(baixa)))
        return sorted(palpite)[:limite]
    except Exception:
        return gerar_palpite_aleatorio(limite)
    finally:
        db.close()

# 3. Funções auxiliares para LS14/LS15
def to_binary(jogo):
    b = [0]*25
    for n in jogo:
        try:
            idx = int(n)-1
            if 0<=idx<25:
                b[idx]=1
        except: 
            continue
    return b

def _calc_features_from_window(ultimos):
    seq_bin = np.array([to_binary(j) for j in ultimos], dtype=np.float32)
    window = len(ultimos)
    freq_vec = seq_bin.sum(axis=0)/float(window)
    atraso_vec = np.zeros(25, dtype=np.float32)
    for d in range(1,26):
        atraso = 0
        for jogo in reversed(ultimos):
            atraso +=1
            if d in jogo:
                break
        atraso_vec[d-1]=min(atraso, window)/float(window)
    last = ultimos[-1]
    soma = sum(last)/(25*15)
    pares = sum(1 for x in last if x%2==0)/15
    global_vec = np.array([soma, pares], dtype=np.float32)
    return seq_bin, freq_vec, atraso_vec, global_vec

# 4. Carregamento híbrido de modelos
@st.cache_resource
def carregar_modelo(nome_modelo="ls15"):
    possiveis_paths = []
    if nome_modelo.lower()=="ls15":
        possiveis_paths = [
            os.path.join(BASE_DIR,"modelo_ls15pp.keras"),
            os.path.join(BASE_DIR,"modelo_ls15pp.h5"),
            os.path.join(BASE_DIR,"modelos","modelo_ls15pp.keras"),
            os.path.join(BASE_DIR,"modelos","modelo_ls15pp.h5")
        ]
    elif nome_modelo.lower()=="ls14":
        possiveis_paths = [
            os.path.join(BASE_DIR,"modelo_ls14pp.keras"),
            os.path.join(BASE_DIR,"modelos","modelo_ls14pp.keras")
        ]
    for path in possiveis_paths:
        if os.path.exists(path):
            try:
                return load_model(path, compile=False)
            except Exception as e:
                st.error(f"Erro ao carregar {nome_modelo.upper()}: {e}")
                return None
    st.warning(f"Modelo {nome_modelo.upper()} não encontrado. Usar apenas Aleatório/Estatístico.")
    return None

# 5. LS15
def gerar_palpite_ls15(limite=15, window=50):
    db = Session()
    try:
        resultados = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT :lim
        """), {"lim": window}).fetchall()
        if len(resultados)<window:
            raise ValueError("Histórico insuficiente para LS15")
        ultimos = [list(map(int,r)) for r in reversed(resultados)]
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)
        modelo = carregar_modelo("ls15")
        if modelo is None:
            raise ValueError("Modelo LS15 não carregado")
        n_inputs = len(getattr(modelo,"inputs",[])) or 1
        if n_inputs==1:
            pred = modelo.predict(seq_bin[None,...], verbose=0)[0]
        elif n_inputs==4:
            pred = modelo.predict([seq_bin[None,...], freq_vec[None,:], atraso_vec[None,:], global_vec[None,:]], verbose=0)[0]
        else:
            raise ValueError(f"LS15: modelo com {n_inputs} entradas não suportado")
        if len(pred)<25:
            buf=np.zeros(25, dtype=np.float32)
            buf[:len(pred)]=pred
            pred=buf
        chosen = np.argsort(pred)[-limite:] + 1
        return sorted(chosen.tolist())
    finally:
        db.close()

# 6. LS14
def gerar_palpite_ls14(limite=15, window=50):
    db = Session()
    try:
        resultados = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT :lim
        """), {"lim": window}).fetchall()
        if len(resultados)<window:
            raise ValueError("Histórico insuficiente para LS14")
        ultimos = [list(map(int,r)) for r in reversed(resultados)]
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)
        hist_input = np.array([[len(set(ultimos[-1]).intersection(set(ultimos[-2])))/15.0 if len(ultimos)>=2 else 0.6]], dtype=np.float32)
        modelo = carregar_modelo("ls14")
        if modelo is None:
            raise ValueError("Modelo LS14 não carregado")
        n_inputs = len(getattr(modelo,"inputs",[])) or 2
        if n_inputs==2:
            pred = modelo.predict([seq_bin[None,...], hist_input], verbose=0)[0]
        elif n_inputs==5:
            pred = modelo.predict([seq_bin[None,...], hist_input, freq_vec[None,:], atraso_vec[None,:], global_vec[None,:]], verbose=0)[0]
        else:
            raise ValueError(f"LS14: modelo com {n_inputs} entradas não suportado")
        if len(pred)<25:
            buf=np.zeros(25,dtype=np.float32)
            buf[:len(pred)]=pred
            pred=buf
        chosen = np.argsort(pred)[-limite:] + 1
        return sorted(chosen.tolist())
    except Exception as e:
        st.error(f"Erro ao gerar palpite LS14: {e}")
        return []
    finally:
        db.close()

# 7. Função geral de geração
def gerar_palpite_from_modelo(modelo_nome, qtde_dezenas):
    if modelo_nome=="Aleatório":
        return gerar_palpite_aleatorio(qtde_dezenas)
    elif modelo_nome=="Estatístico":
        return gerar_palpite_estatistico(qtde_dezenas)
    elif modelo_nome=="Pares/Ímpares":
        return gerar_palpite_pares_impares(qtde_dezenas)
    elif modelo_nome=="LS15":
        return gerar_palpite_ls15(qtde_dezenas)
    elif modelo_nome=="LS14":
        return gerar_palpite_ls14(qtde_dezenas)
    else:
        st.warning("Modelo inválido")
        return []

# 8. Gerador principal (Streamlit)
def gerar_palpite_interface():
    st.title("Gerar Bets")
    if "usuario" not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado para gerar bets")
        return
    id_usuario = st.session_state.usuario["id"]
    id_plano = st.session_state.usuario.get("id_plano",0)

    db = Session()
    try:
        row = db.execute(text("SELECT nome FROM planos WHERE id = :id_plano"), {"id_plano": id_plano}).fetchone()
        nome_plano = row[0] if row else "Desconhecido"
    except:
        nome_plano="Desconhecido"
    finally:
        db.close()

    st.markdown(f"<div style='font-family:Poppins,sans-serif;font-size:16px;color:#0b450b;margin-bottom:20px;'>Plano atual: <strong>{nome_plano}</strong></div>", unsafe_allow_html=True)

    permitido, _, palpites_restantes = verificar_limite_palpites(id_usuario)
    if not permitido:
        st.error("Você atingiu o limite de palpites.")
        return

    modelos_disponiveis = ["Aleatório","Estatístico","Pares/Ímpares"]
    if nome_plano in ["Silver","Gold","Plano Pago X"]:
        modelos_disponiveis += ["LS15","LS14"]

    modelo = st.selectbox("Modelo:", modelos_disponiveis)
    num_palpites = st.number_input("Quantos palpites?",1,max(1,palpites_restantes),1)
    min_dezenas=15
    max_dezenas={"Free":15,"Silver":17,"Gold":20}.get(nome_plano,15)
    qtde_dezenas = st.number_input("Dezenas por palpite?",min_dezenas,max_dezenas,min_dezenas)

    if st.button("Gerar Palpites"):
        palpites=[]
        for _ in range(num_palpites):
            p = gerar_palpite_from_modelo(modelo, qtde_dezenas)
            if p:
                salvar_palpite(p, modelo)
                atualizar_contador_palpites(id_usuario)
                palpites.append(p)
        if palpites:
            st.success(f"{len(palpites)} Palpite(s) gerado(s)")
            for p in palpites:
                st.write(p)
