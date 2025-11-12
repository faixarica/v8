# -*- coding: utf-8 -*-
import streamlit as st
from sqlalchemy import text
from db import Session
from datetime import datetime, date
import pandas as pd
import numpy as np
import random
from typing import Optional
import calendar
# ================================================================
# 🧠 Funções utilitárias
# ================================================================

def _descobrir_coluna_data_palpites_m(db) -> Optional[str]:
    """
    Retorna o nome da coluna de timestamp em palpites_m.
    Procura por candidatos comuns. Cache simples em st.session_state.
    """
    cache_key = "_palpites_m_ts_col"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    candidatos = [
        "dt_criacao", "data_criacao", "created_at", "criado_em",
        "createdon", "data", "dt", "timestamp_criacao"
    ]

    rows = db.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'palpites_m'
    """)).fetchall()
    colunas = {r[0].lower() for r in rows}

    for c in candidatos:
        if c.lower() in colunas:
            st.session_state[cache_key] = c
            return c

    # Se nada bater, avisa claramente:
    st.error("Não encontrei coluna de data em 'palpites_m'. "
             "Crie uma das colunas: dt_criacao, data_criacao, created_at ou criado_em.")
    return None


# ================================================================
# 🧠 Funções utilitárias (versão simplificada — Mega-Sena)
# ================================================================

def obter_limites_por_loteria_e_plano(id_usuario: int):
    """
    Retorna os limites fixos da Mega-Sena.
    Por regra de negócio: mínimo 6 dezenas, máximo 20 dezenas.
    """
    return 6, 20


def verificar_limite_palpites_m(id_usuario: int) -> tuple[bool, str]:
    """
    Controla se o usuário pode gerar palpites da Mega-Sena conforme o plano.
    - Apenas planos Gold e Platinum têm acesso.
    - Limite mensal de palpites fixo conforme o plano.
    """
    db = Session()
    try:
        res = db.execute(text("""
            SELECT u.id, LOWER(p.nome) AS nome_plano
            FROM usuarios u
            JOIN planos p ON p.id = u.id_plano
            WHERE u.id = :id
        """), {"id": id_usuario}).fetchone()

        if not res:
            return False, "Usuário não encontrado."

        nome_plano = res.nome_plano or ""

        # 🔒 Apenas planos Gold e Platinum
        if nome_plano not in ("gold", "platinum"):
            return False, "🚫 Somente planos Gold e Platinum podem gerar palpites da Mega-Sena."

        limites = {"gold": 500, "platinum": 2000}
        limite = limites.get(nome_plano, 0)

        # Contagem mensal
        from datetime import datetime
        import calendar

        agora = datetime.now()
        inicio_mes = agora.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(agora.year, agora.month)[1]
        fim_mes = agora.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)

        # Detecta coluna de data (para suportar bancos diferentes)
        ts_col = _descobrir_coluna_data_palpites_m(db)
        if not ts_col:
            return False, "Estrutura da tabela 'palpites_m' não possui coluna de data."

        sql = f"""
            SELECT COUNT(*) FROM palpites_m
            WHERE id_usuario = :uid
              AND {ts_col} BETWEEN :ini AND :fim
        """
        total = db.execute(text(sql), {"uid": id_usuario, "ini": inicio_mes, "fim": fim_mes}).scalar() or 0

        if total >= limite:
            return False, f"🚫 Limite mensal atingido ({total}/{limite}) para o plano {nome_plano.title()}."
        restante = limite - total
        return True, f"✅ Você ainda pode gerar {restante} palpites ({total}/{limite})."

    except Exception as e:
        return False, f"Erro ao validar limite: {e}"
    finally:
        db.close()


# ================================================================
# 🎯 Geração de Palpites
# ================================================================
def gerar_palpite_ui():
    st.subheader(" Gerar Palpite — Mega-Sena")

    usuario = st.session_state.get("usuario", {})
    if not usuario:
        st.warning("Você precisa estar logado para gerar palpites.")
        return

    # --- Verifica plano
    permitido, msg = verificar_limite_palpites_m(usuario["id"])
    st.info(msg)
    if not permitido:
        return

    # --- Limite de dezenas conforme plano ---
    min_dz, max_dz = obter_limites_por_loteria_e_plano(usuario["id"])
    qtd = st.number_input("Quantidade de dezenas:", min_value=min_dz, max_value=max_dz, value=min_dz)
    modelo = st.selectbox("Modelo de Geração:", ["Aleatório", "Estatístico", "LS16 (Experimental)"])

    if st.button("🎲 Gerar Palpite"):
        dezenas = gerar_palpite_m(modelo, qtd)
        dezenas = evitar_repetidos(dezenas)
        dezenas_fmt = " ".join(f"{n:02d}" for n in dezenas)

        salvar_palpite_m(usuario["id"], dezenas_fmt, modelo)
        st.success("✅ Palpite salvo com sucesso!")

        st.markdown(
            f"<div style='text-align:center; margin-top:15px;'>"
            + "".join(
                f"<span style='display:inline-block; background:#10b981; color:white; "
                f"border-radius:50%; width:40px; height:40px; line-height:40px; margin:4px; "
                f"font-weight:bold;'>{d:02d}</span>" for d in dezenas
            )
            + "</div>", unsafe_allow_html=True
        )


def gerar_palpite_m(modelo: str, qtd: int):
    """Lógica de geração de palpites da Mega-Sena."""
    if modelo == "Aleatório":
        return sorted(random.sample(range(1, 61), qtd))

    elif modelo == "Estatístico":
        df = pd.read_csv("loteriamega.csv")
        bolas = [f"Bola{i}" for i in range(1, 7)]
        todas = df[bolas].values.flatten()
        freq = pd.Series(todas).value_counts().sort_index()
        prob = freq / freq.sum()
        dezenas = np.random.choice(prob.index, size=qtd, replace=False, p=prob.values)
        return sorted(dezenas.tolist())

    elif modelo.startswith("LS16"):
        try:
            from modelo_llm_max.ensemble import gerar_palpite_ensemble
            dezenas = gerar_palpite_ensemble()
            dezenas = [n for n in dezenas if 1 <= n <= 60]
            return sorted(random.sample(dezenas, min(len(dezenas), qtd)))
        except Exception:
            return sorted(random.sample(range(1, 61), qtd))

    else:
        return sorted(random.sample(range(1, 61), qtd))


def evitar_repetidos(dezenas):
    """Evita gerar combinações que já saíram."""
    try:
        df = pd.read_csv("loteriamega.csv")
        existentes = {tuple(sorted(map(int, r))) for r in df.iloc[:, 1:7].values.tolist()}
        while tuple(sorted(dezenas)) in existentes:
            dezenas = sorted(random.sample(range(1, 61), len(dezenas)))
        return dezenas
    except Exception:
        return dezenas

# ================================================================
# 💾 Salvamento
# ================================================================

def salvar_palpite_m(id_usuario, dezenas_fmt, modelo):
    # ✅ Bloqueia palpites vazios ou corrompidos
    if not dezenas_fmt or len(str(dezenas_fmt).strip()) < 5:
        st.warning("⚠️ Palpite inválido ou vazio. Tente novamente.")
        return

    """
    Insere um novo palpite. O campo 'valido' é gerido pelo banco (default = 'N').
    """
    db = Session()
    try:
        ts_col = _descobrir_coluna_data_palpites_m(db)
        if not ts_col:
            raise RuntimeError("Tabela 'palpites_m' sem coluna de data válida.")

        sql = f"""
            INSERT INTO palpites_m (id_usuario, numeros, modelo, {ts_col})
            VALUES (:uid, :nums, :modelo, NOW())
        """
        db.execute(text(sql), {
            "uid": id_usuario,
            "nums": dezenas_fmt,
            "modelo": modelo
        })
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao salvar palpite: {e}")
    finally:
        db.close()

# ================================================================
# 📜 Histórico de Palpites
# ================================================================

def historico_palpites():
    st.subheader("📜 Histórico de Palpites — Mega-Sena")
    usuario = st.session_state.get("usuario", {})
    if not usuario:
        st.warning("Você precisa estar logado.")
        return

    from datetime import date
    data_ini = st.date_input("Data inicial:", date.today().replace(day=1))
    data_fim = st.date_input("Data final:", date.today())

    db = Session()
    try:
        ts_col = _descobrir_coluna_data_palpites_m(db)
        if not ts_col:
            st.error("Não foi possível identificar a coluna de data em 'palpites_m'.")
            return

        sql = f"""
            SELECT id, numeros, modelo, {ts_col} AS dt, valido
            FROM palpites_m
            WHERE id_usuario = :uid
              AND DATE({ts_col}) BETWEEN :ini AND :fim
            ORDER BY {ts_col} DESC
        """
        rows = db.execute(text(sql), {
            "uid": usuario["id"],
            "ini": data_ini,
            "fim": data_fim
        }).fetchall()
    finally:
        db.close()

    if not rows:
        st.info("Nenhum palpite encontrado no período.")
        return

    for r in rows:
        idp, nums, modelo, dt, valido = r
        txt_status = "✅ Validado" if (valido or "").upper() == "S" else "⏳ Não validado"
        cor = "#10b981" if (valido or "").upper() == "S" else "#d97706"

        st.markdown(f"""
            <div style="border:2px solid {cor}; border-radius:12px; padding:10px; margin-bottom:10px; background:#f9fafb;">
                <b>ID:</b> {idp} &nbsp;|&nbsp; <b>📅 {dt.strftime('%d/%m/%Y %H:%M')}</b><br>
                <b>🧠 Modelo:</b> {modelo}<br>
                <b>🎲 Números:</b> {nums}<br>
                <b>📌 Status:</b> <span style="color:{cor}; font-weight:600;">{txt_status}</span>
            </div>
        """, unsafe_allow_html=True)

# ================================================================
# ✅ Validação
# ================================================================
def validar_palpite():
    st.subheader("✅ Validar Palpite — Mega-Sena")
    usuario = st.session_state.get("usuario", {})
    if not usuario:
        st.warning("Você precisa estar logado.")
        return

    db = Session()
    try:
        ts_col = _descobrir_coluna_data_palpites_m(db)
        if not ts_col:
            st.error("Não foi possível identificar a coluna de data em 'palpites_m'.")
            return

        sql = f"""
            SELECT id, numeros, modelo, {ts_col} AS dt, valido
            FROM palpites_m
            WHERE id_usuario = :uid
            ORDER BY {ts_col} DESC
            LIMIT 30
        """
        rows = db.execute(text(sql), {"uid": usuario["id"]}).fetchall()
    finally:
        db.close()

    if not rows:
        st.info("Nenhum palpite para validar.")
        return

    for r in rows:
        idp, nums, modelo, dt, valido = r
        is_validado = (valido or "").upper() == "S"
        status_txt = "✅ Validado" if is_validado else "⏳ Pendente"
        cor = "#10b981" if is_validado else "#d97706"

        st.markdown(f"""
            <div style="border:1px solid {cor}; border-radius:10px; padding:10px; margin-bottom:6px; background:#fff;">
                <b>ID:</b> {idp}<br>
                <b>🧠 Modelo:</b> {modelo}<br>
                <b>📅 Data:</b> {dt.strftime('%d/%m/%Y %H:%M')}<br>
                <b>🎲 Números:</b> {nums}<br>
                <b>📌 Status:</b> <span style="color:{cor}; font-weight:600;">{status_txt}</span>
            </div>
        """, unsafe_allow_html=True)

        if not is_validado:
            if st.button(f"✅ Validar #{idp}", key=f"validar_{idp}"):
                atualizar_status_palpite(idp)
                st.success(f"Palpite #{idp} validado com sucesso!")
                st.rerun()

def atualizar_status_palpite(id_palpite: int):
    """Atualiza o campo 'valido' para 'S'."""
    db = Session()
    try:
        db.execute(text("""
            UPDATE palpites_m
            SET valido = 'S'
            WHERE id = :id
        """), {"id": id_palpite})
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao validar palpite: {e}")
    finally:
        db.close()
