# dashboard.py adaptado para PostgreSQL no Neon.tech
# -------------------- [1] IMPORTS --------------------

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import streamlit as st
from sqlalchemy import text
from db import Session

from sqlalchemy.orm import sessionmaker
# CSS personalizado
# -------------------- [2] CONFIGS --------------------

# -------------------- [3] DEFINIÇÃO DE FUNÇÕES --------------------

def apply_custom_css():
    st.markdown("""
        <style>
            .card {
                padding: 15px;
                margin: 10px 0;
                border-left: 6px solid #6C63FF;
                border-radius: 10px;
                background-color: #f0f2f6;
                text-align: center;
                font-size: 16px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .metric-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }
            .metric-value {
                font-size: 22px;
                color: #6C63FF;
            }
            .scrollable-container {
                max-height: 700px;
                overflow-y: auto;
                padding-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# Dashboard principal

def grafico_frequencia_palpites():
    db = Session()  # Cria uma sessão com o banco
    try:
        result = db.execute(text("SELECT numeros FROM palpites"))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=["numeros"])
    finally:
        db.close()

    # Expande os palpites para colunas
    todos_numeros = df["numeros"].dropna().apply(lambda x: list(map(int, x.split(","))))
    todos_numeros = pd.Series([num for sublist in todos_numeros for num in sublist])

    # Frequência
    frequencia = todos_numeros.value_counts().sort_index()
    df_freq = pd.DataFrame({"Número": frequencia.index, "Frequência": frequencia.values})

    # Gráfico
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=df_freq, x="Número", y="Frequência", palette="Blues", ax=ax)
    ax.set_title("Frequência nos Palpites dos Usuários", fontsize=14)
    ax.set_xlabel("Números")
    ax.set_ylabel("Frequência")

    return fig

def mostrar_dashboard():
    apply_custom_css()
    st.title("Painel Estatístico")

    usuario = st.session_state.get("usuario", {})
    if not usuario:
        st.error("Você precisa estar logado.")
        return

    user_id = usuario["id"]
    plano_id = usuario["id_plano"]
    tipo = usuario.get("tipo", "U")  # U = usuário comum, A = admin

    db = Session()
    try:
        # Último resultado oficial
        ultimo_resultado = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15, data, concurso
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT 1
        """)).fetchone()

        # Estatísticas globais
        total_palpites_plataforma = db.execute(text("SELECT COUNT(*) FROM palpites")).scalar()
        total_usuarios = db.execute(text("SELECT COUNT(*) FROM usuarios")).scalar()
        total_ativos = db.execute(text("SELECT COUNT(DISTINCT id_usuario) FROM palpites")).scalar()

        total_free = db.execute(text("SELECT COUNT(*) FROM usuarios WHERE id_plano = 1")).scalar()
        total_silver = db.execute(text("SELECT COUNT(*) FROM usuarios WHERE id_plano = 2")).scalar()
        total_gold = db.execute(text("SELECT COUNT(*) FROM usuarios WHERE id_plano = 3")).scalar()

        # Estatísticas do usuário logado
        total_user = db.execute(text("SELECT COUNT(*) FROM palpites WHERE id_usuario = :uid"),
                                {"uid": user_id}).scalar()
        total_user_dia = db.execute(text("""
            SELECT COUNT(*) FROM palpites
            WHERE id_usuario = :uid AND DATE(created_at) = CURRENT_DATE
        """), {"uid": user_id}).scalar()

        total_user_mes = db.execute(text("""
            SELECT COUNT(*) FROM palpites
            WHERE id_usuario = :uid
            AND EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE)
            AND EXTRACT(YEAR FROM created_at) = EXTRACT(YEAR FROM CURRENT_DATE)
        """), {"uid": user_id}).scalar()

    finally:
        db.close()

    # ============ CARDS DO USUÁRIO ============ #
    st.subheader("📌 Seu Resumo")
    cols = st.columns(3)

    if plano_id == 1:  # FREE
        with cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Palpites Gerados</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total_user}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif plano_id == 2:  # SILVER
        with cols[0]:
            st.markdown('<div class="card"><div class="metric-title">Palpites Hoje</div>'
                        f'<div class="metric-value">{total_user_dia}</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="card"><div class="metric-title">Palpites no Mês</div>'
                        f'<div class="metric-value">{total_user_mes}</div></div>', unsafe_allow_html=True)

    elif plano_id == 3:  # GOLD
        with cols[0]:
            st.markdown('<div class="card"><div class="metric-title">Palpites Hoje</div>'
                        f'<div class="metric-value">{total_user_dia}</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="card"><div class="metric-title">Palpites no Mês</div>'
                        f'<div class="metric-value">{total_user_mes}</div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="card"><div class="metric-title">Palpites Plataforma</div>'
                        f'<div class="metric-value">{total_palpites_plataforma}</div></div>', unsafe_allow_html=True)

    # ============ ADMIN EXTRA ============ #
    if tipo == "A":
        st.subheader("🛠️ Estatísticas Administrativas")
        cols_admin = st.columns(4)

        with cols_admin[0]:
            st.markdown('<div class="card"><div class="metric-title">Usuários Totais</div>'
                        f'<div class="metric-value">{total_usuarios}</div></div>', unsafe_allow_html=True)
        with cols_admin[1]:
            st.markdown('<div class="card"><div class="metric-title">Usuários Ativos</div>'
                        f'<div class="metric-value">{total_ativos}</div></div>', unsafe_allow_html=True)
        with cols_admin[2]:
            st.markdown('<div class="card"><div class="metric-title">Free</div>'
                        f'<div class="metric-value">{total_free}</div></div>', unsafe_allow_html=True)
        with cols_admin[3]:
            st.markdown('<div class="card"><div class="metric-title">Silver</div>'
                        f'<div class="metric-value">{total_silver}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="card" style="margin-top:15px;">
                <div class="metric-title">Gold</div>
                <div class="metric-value">{total_gold}</div>
            </div>
        """, unsafe_allow_html=True)

    # ============ ÚLTIMO CONCURSO ============ #
    st.subheader("🎲 Último Concurso")
    if ultimo_resultado:
        numeros = ", ".join(map(str, ultimo_resultado[:15]))
        data_sorteio = ultimo_resultado[15]
        concurso = ultimo_resultado[16]
        st.info(f"Concurso {concurso} ({data_sorteio}) → {numeros}")
    else:
        st.warning("Nenhum resultado encontrado.")

    st.markdown("<hr style='margin-top:30px;'>"
                "<div style='text-align:center; font-size:12px; color:gray;'>"
                "Aqui Não é Sorte • é AI </div>", unsafe_allow_html=True)


# As demais funções do dashboard que consultam o banco via pandas.read_sql devem ser adaptadas em seguida.
# Substituir o uso de sqlite3.connect por SQLAlchemy: engine = create_engine(DATABASE_URL)
# Exemplo:
# df = pd.read_sql("SELECT * FROM palpites", engine)

# -------------------- [4] APLICAÇÃO STREAMLIT --------------------
