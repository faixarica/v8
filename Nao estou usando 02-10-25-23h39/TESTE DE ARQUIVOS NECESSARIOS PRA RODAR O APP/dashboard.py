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

    db = Session()
    try:
        ultimo_resultado = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15, data, concurso
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT 1
        """)).fetchone()

        total_palpites = db.execute(text("SELECT COUNT(*) FROM palpites")).scalar()
        total_usuarios_ativos = db.execute(text("SELECT COUNT(DISTINCT id_usuario) FROM palpites")).scalar()
    finally:
        db.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Total de Palpites</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_palpites}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Usuários Ativos</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_usuarios_ativos}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Último Sorteio</div>', unsafe_allow_html=True)
        if ultimo_resultado:
            numeros = ", ".join(map(str, ultimo_resultado[:15]))
            data_sorteio = ultimo_resultado[15]
            concurso = ultimo_resultado[16]
            st.markdown(f'<div class="metric-value">Concurso: {concurso} ({data_sorteio})<br>{numeros}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <hr style="margin-top:30px;">
        <div style='text-align:center; font-size:12px; color:gray;'>
            Aqui Não é Sorte • é AI 
        </div>
    """, unsafe_allow_html=True)

# As demais funções do dashboard que consultam o banco via pandas.read_sql devem ser adaptadas em seguida.
# Substituir o uso de sqlite3.connect por SQLAlchemy: engine = create_engine(DATABASE_URL)
# Exemplo:
# df = pd.read_sql("SELECT * FROM palpites", engine)

# -------------------- [4] APLICAÇÃO STREAMLIT --------------------
