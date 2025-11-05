# perfil.py atualizado para PostgreSQL via SQLAlchemy
# -------------------- [1] IMPORTS --------------------

import streamlit as st
from datetime import datetime
from passlib.hash import pbkdf2_sha256
from sqlalchemy import text

from db import Session  # ou ajuste conforme seu nome de conexão
#from database import carregar_planos

# -------------------- [2] CONFIGS --------------------

# -------------------- [3] DEFINIÇÃO DE FUNÇÕES --------------------
def editar_perfil():
    st.markdown("<div style='text-align:right; color:green; font-size:10px;'><strong>FaixaBet v8.001</strong></div>", unsafe_allow_html=True)

    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado.")
        return

    usuario = st.session_state.usuario
    user_id = usuario['id']

    # Carregar dados do usuário
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

    planos = carregar_planos()

    st.subheader("👤 Editar Dados Pessoais")
    nome_usuario = st.text_input("Nome de Usuário", value=usuario_atual)
    nome = st.text_input("Nome completo", value=nome_atual)
    email = st.text_input("Email", value=email_atual)
    telefone = st.text_input("Telefone", value=telefone_atual)
    nascimento = st.date_input("Data de nascimento", value=nascimento_atual)

    st.markdown("---")

    st.subheader("🔒 Alterar Senha")
    nova_senha = st.text_input("Nova Senha", type="password")
    confirmar_senha = st.text_input("Confirmar Nova Senha", type="password")

    if st.button("Salvar Alterações"):
        db = Session()
        try:
            # Verifica se nome de usuário está em uso por outro
            result = db.execute(text("""
                SELECT id FROM usuarios 
                WHERE usuario = :usuario AND id != :id
            """), {"usuario": nome_usuario, "id": user_id})
            if result.fetchone():
                st.warning("Nome de usuário já está em uso por outro usuário.")
                return

            # Atualiza dados
            db.execute(text("""
                UPDATE usuarios 
                SET usuario = :usuario, nome_completo = :nome, email = :email, 
                    telefone = :telefone, data_nascimento = :nascimento 
                WHERE id = :id
            """), {
                "usuario": nome_usuario,
                "nome": nome,
                "email": email,
                "telefone": telefone,
                "nascimento": nascimento,
                "id": user_id
            })

            # Atualiza senha (se houver)
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

        except Exception as e:
            db.rollback()
            st.error(f"Erro ao atualizar perfil: {e}")
        finally:
            db.close()

def carregar_planos():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, valor FROM planos")
    planos = cursor.fetchall()
    conn.close()
    return {id: {"nome": nome, "valor": valor} for id, nome, valor in planos}
# -------------------- [4] APLICAÇÃO STREAMLIT --------------------


