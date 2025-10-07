# usuario.py - informa qtde de usuarios cadastrados e seus planos
import streamlit as st
from sqlalchemy import text
from db import Session
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

def _format_date(value):
    """Formata dt_cadastro de forma resiliente."""
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y")
    try:
        # tenta interpretar ISO-like strings
        return datetime.fromisoformat(str(value)).strftime("%d/%m/%Y")
    except Exception:
        return str(value)

def listar_usuarios():
    st.markdown("### 👥 Usuários Cadastrados")

    try:
        # usa context manager para garantir fechamento da sessão
        with Session() as db:
            result = db.execute(text("""
                SELECT usuarios.usuario, usuarios.tipo, usuarios.id_plano, usuarios.dt_cadastro, planos.nome AS nome_plano
                FROM usuarios
                LEFT JOIN planos ON planos.id = usuarios.id_plano
                ORDER BY usuarios.dt_cadastro DESC
            """))
            rows = result.fetchall()

        if not rows:
            st.info("Nenhum usuário cadastrado.")
            return

        # --- Contadores por plano ---
        total_free = sum(1 for r in rows if str(r.nome_plano or "").lower().startswith("free"))
        total_silver = sum(1 for r in rows if str(r.nome_plano or "").lower().startswith("silver"))
        total_gold = sum(1 for r in rows if str(r.nome_plano or "").lower().startswith("gold"))
        total = len(rows)

        # --- Cards de resumo ---
        st.markdown(
            f"""
            <div style='display:flex; justify-content:space-around; margin-bottom:20px; flex-wrap:wrap; gap:8px;'>
                <div style='background:#e9f5ff; padding:10px 20px; border-radius:10px; text-align:center; min-width:130px;'>
                    <div style='font-size:16px; font-weight:bold; color:#007bff;'>Free</div>
                    <div style='font-size:22px; font-weight:bold;'>{total_free}</div>
                </div>
                <div style='background:#f3f0ff; padding:10px 20px; border-radius:10px; text-align:center; min-width:130px;'>
                    <div style='font-size:16px; font-weight:bold; color:#6f42c1;'>Silver</div>
                    <div style='font-size:22px; font-weight:bold;'>{total_silver}</div>
                </div>
                <div style='background:#fff7e6; padding:10px 20px; border-radius:10px; text-align:center; min-width:130px;'>
                    <div style='font-size:16px; font-weight:bold; color:#f0ad4e;'>Gold</div>
                    <div style='font-size:22px; font-weight:bold;'>{total_gold}</div>
                </div>
                <div style='background:#f8f9fa; padding:10px 20px; border-radius:10px; text-align:center; min-width:130px;'>
                    <div style='font-size:16px; font-weight:bold; color:#333;'>Total</div>
                    <div style='font-size:22px; font-weight:bold;'>{total}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Cards de usuários ---
        for i in range(0, len(rows), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(rows):
                    r = rows[i + j]
                    usuario = r.usuario
                    tipo = str(r.tipo or "").upper()
                    plano = r.nome_plano or "Free"
                    data_fmt = _format_date(r.dt_cadastro)

                    cor_tipo = "#007bff" if tipo == "ADM" else "#28a745"
                    cor_plano = (
                        "#f0ad4e" if plano.lower().startswith("gold")
                        else "#6f42c1" if plano.lower().startswith("silver")
                        else "#17a2b8"
                    )

                    with cols[j]:
                        st.markdown(f"""
                            <div style='background:#fff; padding:14px 16px; border-radius:12px;
                                        border:1px solid #ddd; margin-bottom:14px;
                                        box-shadow:0 2px 4px rgba(0,0,0,0.08);'>
                                <div style='font-size:15px; font-weight:bold; color:#333;'>👤 {usuario}</div>
                                <div style='font-size:13px; margin-top:4px; line-height:1.4;'>
                                    <span style='color:{cor_tipo}; font-weight:500;'>Tipo:</span> {tipo}<br>
                                    <span style='color:{cor_plano}; font-weight:500;'>Plano:</span> {plano}<br>
                                    <span style='color:#999;'>Cadastrado em:</span> {data_fmt}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

    except SQLAlchemyError as db_err:
        logger.exception("Erro ao executar query em listar_usuarios")
        st.error("Erro ao carregar usuários (problema no banco de dados). Verifique os logs.")
    except Exception as e:
        logger.exception("Erro inesperado em listar_usuarios")
        st.error("Erro inesperado ao listar usuários. Verifique os logs.")
