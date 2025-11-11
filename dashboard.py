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
from datetime import datetime, date


# -------------------- [2] CSS PERSONALIZADO --------------------

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

# -------------------- [3] FUNÇÃO AUXILIAR DE DATA --------------------

def _fmt_date_br(x):
    """Converte qualquer tipo de data em formato dd/mm/yyyy"""
    if isinstance(x, (datetime, date)):
        return x.strftime("%d/%m/%Y")
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(x), fmt).strftime("%d/%m/%Y")
        except Exception:
            continue
    return str(x)


# -------------------- [4] DASHBOARD PRINCIPAL --------------------

def grafico_frequencia_palpites():
    db = Session()
    try:
        result = db.execute(text("SELECT numeros FROM palpites"))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=["numeros"])
    finally:
        db.close()

    todos_numeros = df["numeros"].dropna().apply(lambda x: list(map(int, x.split(","))))
    todos_numeros = pd.Series([num for sublist in todos_numeros for num in sublist])
    frequencia = todos_numeros.value_counts().sort_index()
    df_freq = pd.DataFrame({"Número": frequencia.index, "Frequência": frequencia.values})

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=df_freq, x="Número", y="Frequência", palette="Blues", ax=ax)
    ax.set_title("Frequência nos Palpites dos Usuários", fontsize=14)
    ax.set_xlabel("Números")
    ax.set_ylabel("Frequência")

    return fig

def mostrar_telemetria():
    st.markdown("## 📊 Telemetria dos Modelos (FaixaBet AI)")
    db = Session()

    try:
        # --------------------------------------------
        # 1) Quantidade de palpites por modelo
        # --------------------------------------------
        st.markdown("### 🔢 Quantidade de palpites gerados")
        query_qtd = text("""
            SELECT modelo, COUNT(*) AS quantidade
            FROM telemetria
            GROUP BY modelo
            ORDER BY quantidade DESC;
        """)
        df_qtd = pd.read_sql(query_qtd, db.bind)
        st.bar_chart(df_qtd.set_index("modelo"))

        # --------------------------------------------
        # 2) Desempenho médio cruzando resultados oficiais
        # --------------------------------------------
        st.markdown("### 🎯 Desempenho médio de acertos")

        query_perf = text("""
            SELECT 
                t.modelo,
                COUNT(*) AS palpites,
                ROUND(AVG(
                    (
                        SELECT COUNT(*)
                        FROM unnest(string_to_array(t.numeros, ' ')) AS p(num)
                        WHERE p.num = ANY(ARRAY[
                            r.d1, r.d2, r.d3, r.d4, r.d5,
                            r.d6, r.d7, r.d8, r.d9, r.d10,
                            r.d11, r.d12, r.d13, r.d14, r.d15
                        ]::text[])
                    )
                ), 2) AS media_acertos,
                
                SUM(
                    (
                        SELECT COUNT(*)
                        FROM unnest(string_to_array(t.numeros, ' ')) AS p(num)
                        WHERE p.num = ANY(ARRAY[
                            r.d1, r.d2, r.d3, r.d4, r.d5,
                            r.d6, r.d7, r.d8, r.d9, r.d10,
                            r.d11, r.d12, r.d13, r.d14, r.d15
                        ]::text[])
                    ) >= 13
                ) AS qtd_13p,

                SUM(
                    (
                        SELECT COUNT(*)
                        FROM unnest(string_to_array(t.numeros, ' ')) AS p(num)
                        WHERE p.num = ANY(ARRAY[
                            r.d1, r.d2, r.d3, r.d4, r.d5,
                            r.d6, r.d7, r.d8, r.d9, r.d10,
                            r.d11, r.d12, r.d13, r.d14, r.d15
                        ]::text[])
                    ) >= 14
                ) AS qtd_14p,

                SUM(
                    (
                        SELECT COUNT(*)
                        FROM unnest(string_to_array(t.numeros, ' ')) AS p(num)
                        WHERE p.num = ANY(ARRAY[
                            r.d1, r.d2, r.d3, r.d4, r.d5,
                            r.d6, r.d7, r.d8, r.d9, r.d10,
                            r.d11, r.d12, r.d13, r.d14, r.d15
                        ]::text[])
                    ) = 15
                ) AS qtd_15p

            FROM telemetria t
            JOIN resultados_oficiais r 
                ON to_char(t.data_execucao, 'DD/MM/YYYY') = r.data
            GROUP BY t.modelo
            ORDER BY media_acertos DESC;
        """)

        df_perf = pd.read_sql(query_perf, db.bind)
        st.dataframe(df_perf)

        # --------------------------------------------
        # 3) Melhor modelo do dia
        # --------------------------------------------
        if not df_perf.empty:
            best = df_perf.iloc[0]
            st.success(f"🏆 **Melhor modelo até agora:** `{best.modelo}` — média **{best.media_acertos} acertos**")

    except Exception as e:
        st.error(f"Erro ao carregar telemetria: {e}")

    finally:
        db.close()


def mostrar_dashboard():
    apply_custom_css()
    st.title("Painel Estatístico")

    # -------------------------------
    # 🔹 Verifica login
    # -------------------------------
    usuario = st.session_state.get("usuario", {})
    if not usuario:
        st.error("Você precisa estar logado.")
        return

    user_id = usuario["id"]
    plano_id = usuario["id_plano"]
    tipo = usuario.get("tipo", "U")

    # -------------------------------
    # 🔹 Coleta dados principais
    # -------------------------------
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

    # -------------------------------
    # 🔹 Cards por plano
    # -------------------------------
    cols = st.columns(3)  # ✅ define antes de qualquer uso

    if plano_id == 1:  # FREE
        with cols[0]:
            st.markdown(f"""
                <div class="card">
                    <div class="metric-title">Palpites Gerados</div>
                    <div class="metric-value">{total_user}</div>
                </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
                <div class="card">
                    <div class="metric-title">Usuários Ativos</div>
                    <div class="metric-value">{total_ativos}</div>
                </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
                <div class="card">
                    <div class="metric-title">Palpites Plataforma</div>
                    <div class="metric-value">{total_palpites_plataforma}</div>
                </div>
            """, unsafe_allow_html=True)

    elif plano_id == 2:  # SILVER
        with cols[0]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Palpites Hoje</div>
                <div class="metric-value">{total_user_dia}</div></div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Palpites no Mês</div>
                <div class="metric-value">{total_user_mes}</div></div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Total Plataforma</div>
                <div class="metric-value">{total_palpites_plataforma}</div></div>
            """, unsafe_allow_html=True)

    elif plano_id == 3:  # GOLD
        with cols[0]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Palpites Hoje</div>
                <div class="metric-value">{total_user_dia}</div></div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Palpites no Mês</div>
                <div class="metric-value">{total_user_mes}</div></div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
                <div class="card"><div class="metric-title">Palpites Plataforma</div>
                <div class="metric-value">{total_palpites_plataforma}</div></div>
            """, unsafe_allow_html=True)

    # -------------------------------
    # 🔹 Estatísticas administrativas
    # -------------------------------
    if tipo == "A":
        st.subheader("🛠️ Estatísticas Administrativas")
        cols_admin = st.columns(4)
        with cols_admin[0]:
            st.markdown(f'<div class="card"><div class="metric-title">Usuários Totais</div><div class="metric-value">{total_usuarios}</div></div>', unsafe_allow_html=True)
        with cols_admin[1]:
            st.markdown(f'<div class="card"><div class="metric-title">Usuários Ativos</div><div class="metric-value">{total_ativos}</div></div>', unsafe_allow_html=True)
        with cols_admin[2]:
            st.markdown(f'<div class="card"><div class="metric-title">Free</div><div class="metric-value">{total_free}</div></div>', unsafe_allow_html=True)
        with cols_admin[3]:
            st.markdown(f'<div class="card"><div class="metric-title">Silver</div><div class="metric-value">{total_silver}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="card" style="margin-top:15px;">
                <div class="metric-title">Gold</div>
                <div class="metric-value">{total_gold}</div>
            </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # 🔹 Último concurso
    # -------------------------------
    st.subheader("🎲 Resultados Oficiais")

    db = Session()
    try:
        concursos_disp = db.execute(text("""
            SELECT concurso, data
            FROM resultados_oficiais
            ORDER BY concurso DESC
            LIMIT 50
        """)).fetchall()
    finally:
        db.close()

    if not concursos_disp:
        st.warning("Nenhum resultado encontrado.")
        return

    # Monta opções formatadas
    opcoes = [f"{c[0]} - {_fmt_date_br(c[1])}" for c in concursos_disp]
    concurso_sel = st.selectbox(" Escolha o Concurso:", options=opcoes, index=0)
    concurso_num = int(concurso_sel.split(" - ")[0])

    db = Session()
    try:
        resultado = db.execute(text("""
            SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,
                   data, concurso, ganhadores_11, ganhadores_12, ganhadores_13, ganhadores_14, ganhadores_15
            FROM resultados_oficiais
            WHERE concurso = :conc
            LIMIT 1
        """), {"conc": concurso_num}).fetchone()
    finally:
        db.close()

    if not resultado:
        st.warning("Dados não encontrados para o concurso selecionado.")
        return

    numeros = [int(x) for x in resultado[:15]]
    data_sorteio = resultado[15]
    concurso = resultado[16]
    g11, g12, g13, g14, g15 = resultado[17:]
    data_fmt = _fmt_date_br(data_sorteio)

    # 🎯 Card principal
    st.markdown(f"""
        <div class="card" style="background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
            color: white; font-size:18px; text-align:center; font-weight:600;">
            Concurso <span style="font-size:22px;">{concurso}</span> — 
            <span style="font-weight:400;">{data_fmt}</span>
        </div>
    """, unsafe_allow_html=True)

    # 💜 Números sorteados
    bolhas = "".join([
        f"<span style='display:inline-block; background:#6C63FF; color:white; "
        f"border-radius:50%; width:38px; height:38px; line-height:38px; "
        f"margin:3px; font-weight:bold;'>{n:02d}</span>"
        for n in numeros
    ])
    st.markdown(f"<div style='text-align:center; margin-top:10px;'>{bolhas}</div>", unsafe_allow_html=True)

    # 🏆 Ganhadores por Faixa
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 Ganhadores por Faixa")

    cols_g = st.columns(5)
    faixas = [
        ("11 Acertos", g11),
        ("12 Acertos", g12),
        ("13 Acertos", g13),
        ("14 Acertos", g14),
        ("15 Acertos", g15)
    ]
    for i, (titulo, valor) in enumerate(faixas):
        with cols_g[i]:
            st.markdown(f"""
                <div class="card" style="background:#f8fafc;">
                    <div class="metric-title">{titulo}</div>
                    <div class="metric-value" style="color:#10b981;">{valor if valor is not None else '-'}</div>
                </div>
            """, unsafe_allow_html=True)