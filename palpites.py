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

def obter_limite_dezenas_por_plano(nome_plano):
    db = Session()
    try:
        resultado = db.execute(text("SELECT palpites_max FROM planos WHERE nome = :nome"), {"nome": nome_plano}).fetchone()
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
            WHERE id_client=:id AND DATE(data_expira_plan)>=CURRENT_DATE AND ativo=true
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

# -------------------- [4] GERAÇÃO --------------------

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
        resultados = db.execute(text("SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 FROM resultados_oficiais")).fetchall()
        if not resultados:
            return gerar_palpite_aleatorio(limite)
        todos_numeros = [num for row in resultados for num in row]
        freq = {n: todos_numeros.count(n) for n in range(1,26)}
        numeros_ordenados = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top = [n for n,_ in numeros_ordenados[:10]]
        mid = [n for n,_ in numeros_ordenados[10:20]]
        low = [n for n,_ in numeros_ordenados[20:]]
        palpite = random.sample(top,min(7,len(top))) + random.sample(mid,min(5,len(mid))) + random.sample(low,min(3,len(low)))
        return sorted(palpite)[:limite]
    except:
        return gerar_palpite_aleatorio(limite)
    finally:
        db.close()

# -------------------- [5] MODELOS LS --------------------

@st.cache_resource
def carregar_modelo_ls15():
    paths = [
        os.path.join(BASE_DIR,"modelo_ls15pp.keras"),
        os.path.join(BASE_DIR,"modelos","modelo_ls15pp.keras")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return load_model(p, compile=False)
            except:
                continue
    st.warning("Modelo LS15 não encontrado.")
    return None

@st.cache_resource
def carregar_modelo_ls14():
    paths = [
        os.path.join(BASE_DIR,"modelo_ls14pp.keras"),
        os.path.join(BASE_DIR,"modelos","modelo_ls14pp.keras")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return load_model(p, compile=False)
            except:
                continue
    st.warning("Modelo LS14 não encontrado.")
    return None

# Funções auxiliares
def to_binary(jogo):
    b = [0]*25
    for n in jogo:
        if 1 <= n <= 25:
            b[n-1] = 1
    return b

def _calc_features_from_window(ultimos):
    seq_bin = np.array([to_binary(j) for j in ultimos], dtype=np.float32)
    window = len(ultimos)
    freq_vec = seq_bin.sum(axis=0)/float(window)
    atraso_vec = np.zeros(25,dtype=np.float32)
    for d in range(1,26):
        atraso = 0
        for jogo in reversed(ultimos):
            atraso += 1
            if d in jogo: break
        atraso_vec[d-1] = min(atraso,window)/float(window)
    last = ultimos[-1]
    soma = sum(last)/(25*15)
    pares = sum(1 for x in last if x%2==0)/15.0
    global_vec = np.array([soma, pares], dtype=np.float32)
    return seq_bin, freq_vec, atraso_vec, global_vec

def gerar_palpite_ls15(limite=15, window=50):
    db = Session()
    try:
        res = db.execute(text("SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 FROM resultados_oficiais ORDER BY concurso DESC LIMIT :lim"), {"lim": window}).fetchall()
        if len(res)<window:
            return gerar_palpite_aleatorio(limite)
        ultimos = [list(map(int,r)) for r in reversed(res)]
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)
        modelo = carregar_modelo_ls15()
        if modelo is None:
            return gerar_palpite_aleatorio(limite)
        pred = modelo.predict([seq_bin[None,...], freq_vec[None,:], atraso_vec[None,:], global_vec[None,:]], verbose=0)[0]
        pred = np.pad(pred,(0,25-len(pred))) if len(pred)<25 else pred
        return sorted(np.argsort(pred)[-limite:]+1)
    finally:
        db.close()

def gerar_palpite_ls14(limite=15, window=50):
    db = Session()
    try:
        res = db.execute(text("SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 FROM resultados_oficiais ORDER BY concurso DESC LIMIT :lim"), {"lim": window}).fetchall()
        if len(res)<window:
            return gerar_palpite_aleatorio(limite)
        ultimos = [list(map(int,r)) for r in reversed(res)]
        seq_bin, freq_vec, atraso_vec, global_vec = _calc_features_from_window(ultimos)
        modelo = carregar_modelo_ls14()
        if modelo is None:
            return gerar_palpite_aleatorio(limite)
        hist_input = np.array([[0.6]],dtype=np.float32)
        pred = modelo.predict([seq_bin[None,...], hist_input, freq_vec[None,:], atraso_vec[None,:], global_vec[None,:]], verbose=0)[0]
        pred = np.pad(pred,(0,25-len(pred))) if len(pred)<25 else pred
        return sorted(np.argsort(pred)[-limite:]+1)
    finally:
        db.close()
