# perfil.py atualizado para PostgreSQL via SQLAlchemy
# Corrigido para tratar corretamente data_nascimento/nascimento_atual
# compatível com Streamlit Cloud e qualquer tipo retornado do banco.

import streamlit as st
from datetime import datetime, date
from passlib.hash import pbkdf2_sha256
from sqlalchemy import text

from db import Session  # conexão
from database import carregar_planos


# -------------------- [FUNÇÃO PRINCIPAL] --------------------
def editar_perfil():
    st.markdown(
        "<div style='text-align:right; color:green; font-size:10px;'><strong>FaixaBet v8.001</strong></div>",
        unsafe_allow_html=True
    )

    # Verifica login
    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado.")
        return

    usuario = st.session_state.usuario
    user_id = usuario['id']

    # -------------------- CARREGAR DADOS DO USUÁRIO --------------------
    db = Session()
    try:
        result = db.execute(text("""
            SELECT usuario, nome_completo, email, telefone, data_nascimento, id_plano 
            FROM usuarios WHERE id = :id
        """), {"id": user_id})
        dados = result.fetchone()
    finally:
        db.close()

    if not dados:
        st.error("Usuário não encontrado.")
        return

    usuario_atual, nome_atual, email_atual, telefone_atual, nascimento_atual, plano_atual_id = dados

    # -------------------- TRATAR DATA DE NASCIMENTO --------------------
    # Pode vir como datetime.date, datetime.datetime, string ou None
    if nascimento_atual:
        try:
            if isinstance(nascimento_atual, str):
                # Tenta converter a partir de formato ISO 'YYYY-MM-DD'
                nascimento_atual = datetime.strptime(nascimento_atual[:10], "%Y-%m-%d").date()
            elif isinstance(nascimento_atual, datetime):
                nascimento_atual = nascimento_atual.date()
            elif not isinstance(nascimento_atual, date):
                # Tipo inesperado (ex.: Decimal, int, etc.)
                nascimento_atual = date.today()
        except Exception:
            nascimento_atual = date.today()
    else:
        nascimento_atual = date.today()

    # -------------------- FORMULÁRIO PERFIL --------------------
    planos = carregar_planos()

    st.subheader("👤 Editar Dados Pessoais")
    nome_usuario = st.text_input("Nome de Usuário", value=usuario_atual or "")
    nome = st.text_input("Nome completo", value=nome_atual or "")
    email = st.text_input("Email", value=email_atual or "")
    telefone = st.text_input("Telefone", value=telefone_atual or "")
    nascimento = st.date_input("Data de nascimento", value=nascimento_atual)

    st.markdown("---")

    # -------------------- FORMULÁRIO SENHA --------------------
    st.subheader("🔒 Alterar Senha")
    nova_senha = st.text_input("Nova Senha", type="password")
    confirmar_senha = st.text_input("Confirmar Nova Senha", type="password")

    # -------------------- SALVAR ALTERAÇÕES --------------------
    if st.button("Salvar Alterações"):
        db = Session()
        try:
            # Verifica se nome de usuário já existe em outro perfil
            result = db.execute(text("""
                SELECT id FROM usuarios 
                WHERE usuario = :usuario AND id != :id
            """), {"usuario": nome_usuario, "id": user_id})
            if result.fetchone():
                st.warning("Nome de usuário já está em uso por outro usuário.")
                return

            # Atualiza dados pessoais
            db.execute(text("""
                UPDATE usuarios 
                SET usuario = :usuario, 
                    nome_completo = :nome, 
                    email = :email, 
                    telefone = :telefone, 
                    data_nascimento = :nascimento 
                WHERE id = :id
            """), {
                "usuario": nome_usuario,
                "nome": nome,
                "email": email,
                "telefone": telefone,
                "nascimento": nascimento,
                "id": user_id
            })

            # Atualiza senha (se informada)
            if nova_senha.strip():
                if nova_senha == confirmar_senha:
                    senha_hash = pbkdf2_sha256.hash(nova_senha)
                    db.execute(text("UPDATE usuarios SET senha = :senha WHERE id = :id"), {
                        "senha": senha_hash,
                        "id": user_id
                    })
                    st.success("Senha atualizada com sucesso.")
                else:
                    st.warning("As senhas não coincidem. Senha não foi alterada.")

            db.commit()
            st.success("Perfil atualizado com sucesso.")

            # Atualiza sessão (opcional, evita inconsistência)
            st.session_state.usuario['usuario'] = nome_usuario
            st.session_state.usuario['nome_completo'] = nome
            st.session_state.usuario['email'] = email
            st.session_state.usuario['telefone'] = telefone
            st.session_state.usuario['data_nascimento'] = nascimento

        except Exception as e:
            db.rollback()
            st.error(f"Erro ao atualizar perfil: {e}")
        finally:
            db.close()
