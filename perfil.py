# perfil.py — versão revisada (27/10)
# ✅ Corrige faixa de data mínima (>=18 anos)
# ✅ Mantém apenas 1 botão "Salvar Alterações" (verde centralizado)
# ✅ Botão funcional (atualiza tabela de usuários)
# ✅ Exibe aviso amigável se idade < 18 anos

import streamlit as st
from datetime import datetime, date
from passlib.hash import pbkdf2_sha256
from sqlalchemy import text
from db import Session
from database import carregar_planos


def editar_perfil():
    # --------------------------------
    # Verifica login
    # --------------------------------
    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado.")
        return

    usuario = st.session_state.usuario
    user_id = usuario['id']

    # --------------------------------
    # Carregar dados do usuário
    # --------------------------------
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

    # --------------------------------
    # Trata data de nascimento
    # --------------------------------
    if nascimento_atual:
        try:
            if isinstance(nascimento_atual, str):
                nascimento_atual = datetime.strptime(nascimento_atual[:10], "%Y-%m-%d").date()
            elif isinstance(nascimento_atual, datetime):
                nascimento_atual = nascimento_atual.date()
            elif not isinstance(nascimento_atual, date):
                nascimento_atual = date.today()
        except Exception:
            nascimento_atual = date.today()
    else:
        nascimento_atual = date.today()

    # Define limite máximo para seleção de data (usuário deve ter >= 18 anos)

    hoje = date.today()
    limite_max = date(hoje.year - 18, hoje.month, hoje.day)
    limite_min = date(1900, 1, 1)

    # Garante que a data carregada esteja dentro do intervalo permitido
    if nascimento_atual < limite_min or nascimento_atual > limite_max:
        nascimento_atual = limite_max  # força data válida

    # --------------------------------
    # Formulário do perfil
    # --------------------------------
    planos = carregar_planos()

    st.subheader("👤 Editar Dados Pessoais")

    nome_usuario = st.text_input("Nome de Usuário", value=usuario_atual or "")
    nome = st.text_input("Nome completo", value=nome_atual or "")
    email = st.text_input("Email", value=email_atual or "")
    telefone = st.text_input("Telefone", value=telefone_atual or "")
    nascimento = st.date_input(
        "Data de nascimento",
        value=nascimento_atual,
        min_value=limite_min,
        max_value=limite_max,
        format="DD/MM/YYYY"
    )

    st.markdown("---")

    # --------------------------------
    # Formulário de senha
    # --------------------------------
    st.subheader("🔒 Alterar Senha")
    nova_senha = st.text_input("Nova Senha", type="password")
    confirmar_senha = st.text_input("Confirmar Nova Senha", type="password")

    # --------------------------------
    # Botão estilizado (único e funcional)
    # --------------------------------
    st.markdown("""
        <style>
        .save-container {
            display: flex;
            justify-content: center;
            margin-top: 25px;
        }
        div[data-testid="stFormSubmitButton"] button, .save-container button {
            width: 80%;
            background-color: #2ecc71 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 0;
            font-size: 17px;
        }
        .save-container button:hover {
            background-color: #27ae60 !important;
        }
        @media (max-width: 600px) {
            .save-container button {
                width: 100%;
                font-size: 16px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    salvar = st.button("💾 Salvar Alterações", key="save_changes")

    # --------------------------------
    # Lógica de salvamento
    # --------------------------------
    if salvar:
        # ✅ Validação de idade mínima
        idade = (hoje - nascimento).days // 365
        if idade < 18:
            st.error("⚠️ Você precisa ter mais de 18 anos para usar a plataforma.")
            return

        db = Session()
        try:
            # Verifica duplicidade de nome de usuário
            result = db.execute(text("""
                SELECT id FROM usuarios WHERE usuario = :usuario AND id != :id
            """), {"usuario": nome_usuario, "id": user_id})
            if result.fetchone():
                st.warning("Nome de usuário já está em uso por outro usuário.")
                return

            # Atualiza informações básicas
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

            # Atualiza senha se informada
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
            st.success("✅ Perfil atualizado com sucesso.")

            # Atualiza sessão
            st.session_state.usuario.update({
                "usuario": nome_usuario,
                "nome_completo": nome,
                "email": email,
                "telefone": telefone,
                "data_nascimento": nascimento
            })

        except Exception as e:
            db.rollback()
            st.error(f"Erro ao atualizar perfil: {e}")
        finally:
            db.close()
