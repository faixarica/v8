# main v8.01
# -------------------- [1] IMPORTS --------------------

import os
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from db import Session
from dashboard import mostrar_dashboard
from palpites import gerar_palpite, historico_palpites, validar_palpite
# from auth import logout
from perfil import editar_perfil
from financeiro import exibir_aba_financeiro
import requests
import pandas as pd
from passlib.hash import pbkdf2_sha256
import streamlit.components.v1 as components

# -------------------- [2] CONFIGS --------------------

st.set_page_config(page_title="FaixaBet", layout="centered")
# Cabeçalho fixo
st.markdown("""
    <div style='
        width: 100%; 
        text-align: center; 
        padding: 6px 0; 
        font-size: 46px; 
        font-weight: bold; 
        color: green;
        border-bottom: 1px solid #DDD;
    '>Bem-vindo à FaixaBet®
        <hr style="margin: 0; border: 0; border-top: 1px solid #DDD;">
        <div style='text-align:center; font-size:16px; color:black; margin-top:4px;'>
             Aqui Não é Sorte   •      é  AI 
        </div>
    </div>
""", unsafe_allow_html=True)


# -------------------- [3] DEFINIÇÃO DE FUNÇÕES --------------------

def css_global():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
    }
        
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');
        html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
    }
        /* Centraliza o título */
        .main > div > div > div > div {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Título FaixaBet */
        .login-title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            color: #008000;
            margin-bottom: 24px;
        }

        /* Estilo dos inputs e botões */
        input, .stButton button {
            width: 50ch !important;
            max-width: 60%;
            margin-top: 8px;
            padding: 8px;
            border-radius: 8px;
        }

        /* Botões */
        .stButton button {
            background-color: #008000;
            color: white;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #005e00;
        }

        /* Radio Buttons - horizontal e colorido */
        div[role="radiogroup"] > label[data-baseweb="radio"] div[aria-checked="true"] {
            background-color: #00C853;
            border-color: #00C853;
            color: white;
        }
        /* Texto do radio */
        label[data-baseweb="radio"] {
        font-size: 40px !important;
        color: #0d730d !important;
        font-weight: 500;
        }
        /* Cards simulados */
        .login-card {
            padding: 16px;
            background-color: #f9f9f9;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
            margin-top: 16px;
        }

        <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
    }

    .login-card {
        background-color: #f9f9f9;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    .stButton button {
        font-size: 18px !important;
        padding: 10px 24px !important;
        transform: scale(1.1);
    }
    </style>
        </style>
    """, unsafe_allow_html=True)

# tava aqui a def q fazia download do database do sqlite
def registrar_login(id_usuario): 
    try:
        resposta = requests.get("https://ipinfo.io/json", timeout=5)
        dados = resposta.json()
    except:
        dados = {}

    db = Session()    
    try:
        db.execute(text("""
            INSERT INTO log_user (
                id_cliente, data_hora, ip, hostname, city, region, country, loc, org, postal, timezone
            ) VALUES (
                :id_client, now(), :ip, :hostname, :city, :region, :country, :loc, :org, :postal, :timezone
            )
        """), {
            "id_client": id_usuario,
            "ip": dados.get("ip", "desconhecido"),
            "hostname": dados.get("hostname", ""),
            "city": dados.get("city", ""),
            "region": dados.get("region", ""),
            "country": dados.get("country", ""),
            "loc": dados.get("loc", ""),
            "org": dados.get("org", ""),
            "postal": dados.get("postal", ""),
            "timezone": dados.get("timezone", "")
        })
        db.commit()
    finally:
        db.close()

def criar_usuario(nome, email, telefone, data_nascimento, usuario, senha, tipo, id_plano):
    from passlib.hash import pbkdf2_sha256
    db = Session()

    try:
        # Verifica duplicidade de usuário/email
        existe = db.execute(text("""
            SELECT 1 FROM usuarios WHERE usuario = :usuario OR email = :email
        """), {
            "usuario": usuario,
            "email": email
        }).fetchone()

        if existe:
            st.error("Usuário ou email já cadastrado!")
            return False

        # Criptografa a senha
        senha_hash = pbkdf2_sha256.hash(senha)

        # Insere o usuário e recupera ID
        result = db.execute(text("""
            INSERT INTO usuarios (
                nome_completo, email, telefone, data_nascimento, 
                senha, usuario, tipo, id_plano, dt_cadastro
            ) VALUES (
                :nome, :email, :telefone, :data_nascimento,
                :senha, :usuario, :tipo, :id_plano, now()
            ) RETURNING id
        """), {
            "nome": nome,
            "email": email,
            "telefone": telefone,
            "data_nascimento": data_nascimento,
            "senha": senha_hash,
            "usuario": usuario,
            "tipo": tipo,
            "id_plano": id_plano
        })

        id_cliente = result.scalar()
        hoje = datetime.now()
        expiracao = hoje + timedelta(days=30)

        # Insere plano ativo inicial
        db.execute(text("""
            INSERT INTO client_plans (
                id_client, id_plano, ativo, data_inclusao, data_expira_plan
            ) VALUES (
                :id_client, :id_plano, true, :data_inclusao, :data_expira_plan
            )
        """), {
            "id_client": id_cliente,
            "id_plano": id_plano,
            "data_inclusao": hoje,
            "data_expira_plan": expiracao
        })

        db.commit()
        st.success("Cadastro realizado com sucesso! Faça login para continuar.")
        return True

    except Exception as e:
        db.rollback()
        st.error(f"Erro no cadastro: {e}")
        return False

    finally:
        db.close()

def calcular_palpites_periodo(id_usuario):
    db = Session()
    try:
        dia_result = db.execute(text("""
            SELECT COUNT(*) FROM palpites WHERE id_usuario = :id AND DATE(data) = CURRENT_DATE
        """), {"id": id_usuario})
        dia = dia_result.scalar()

        semana_result = db.execute(text("""
            SELECT COUNT(*) FROM palpites 
            WHERE id_usuario = :id AND DATE_PART('week', data) = DATE_PART('week', CURRENT_DATE)
              AND DATE_PART('year', data) = DATE_PART('year', CURRENT_DATE)
        """), {"id": id_usuario})
        semana = semana_result.scalar()

        mes_result = db.execute(text("""
            SELECT COUNT(*) FROM palpites 
            WHERE id_usuario = :id AND DATE_PART('month', data) = DATE_PART('month', CURRENT_DATE)
              AND DATE_PART('year', data) = DATE_PART('year', CURRENT_DATE)
        """), {"id": id_usuario})
        mes = mes_result.scalar()

        return dia or 0, semana or 0, mes or 0
    finally:
        db.close()

# -------------------- [4] APLICAÇÃO STREAMLIT --------------------
# LOGIN / CADASTRO 1
if not st.session_state.get("logged_in", False):
    st.markdown("## Acesso ao Sistema")
    aba = st.radio("Ação", ["Login", "Cadastro"], horizontal=True, label_visibility="collapsed")
    st.write("")

    if aba == "Login":
        usuario_input = st.text_input("Usuário")
        senha_input = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            db = Session()
            user = None
            try:
                if usuario_input == "ufaixa990" and senha_input == "ufaixa990!":
                    st.session_state.logged_in = True
                    st.session_state.usuario = {
                        "id": 0,
                        "nome": "Administrador",
                        "email": "adm@faixabet.com",
                        "tipo": "admin",
                        "id_plano": 0
                    }
                    st.session_state.admin = True
                    st.success("Login administrativo realizado!")
                    st.rerun()

                result = db.execute(
                    text("SELECT id, tipo, usuario, email, senha, ativo FROM usuarios WHERE usuario = :usuario"),
                    {"usuario": usuario_input}
                )
                user = result.fetchone()

                if user:
                    id, tipo, usuario, email, senha_hash, ativo = user

                    result = db.execute(text("SELECT id_plano FROM usuarios WHERE id = :id"), {"id": id})
                    row = result.fetchone()
                    id_plano_armazenado = row[0] if row else 1

                    result = db.execute(text("""
                        SELECT cp.id_plano FROM client_plans cp
                        WHERE cp.id_client = :id AND cp.ativo = true
                        ORDER BY cp.data_inclusao DESC
                        LIMIT 1
                    """), {"id": id})
                    row = result.fetchone()
                    id_plano_ativo = row[0] if row else None
                    id_plano_atual = id_plano_ativo or id_plano_armazenado

                    if id_plano_armazenado != id_plano_atual:
                        try:
                            db.execute(text("UPDATE usuarios SET id_plano = :plano WHERE id = :id"),
                                       {"plano": id_plano_atual, "id": id})
                            db.commit()
                        except:
                            st.warning("Não foi possível atualizar o plano do usuário.")

                    if pbkdf2_sha256.verify(senha_input, senha_hash):
                        if ativo:
                            st.session_state.logged_in = True
                            st.session_state.usuario = {
                                "id": id,
                                "nome": usuario,
                                "email": email,
                                "tipo": tipo,
                                "id_plano": id_plano_atual
                            }
                            if tipo == "admin":
                                st.session_state.admin = True
                            registrar_login(id)
                            st.success("Login realizado com sucesso!")
                            st.rerun()
                        else:
                            st.error("Conta inativa.")
                    else:
                        st.error("Senha incorreta.")
                else:
                    st.error("Usuário não encontrado.")

            except Exception as e:
                st.error(f"Erro durante o login: {e}")
            finally:
                db.close()

        st.markdown('<a class="recover" href="#">Esqueceu a senha?</a>', unsafe_allow_html=True)

    elif aba == "Cadastro":
        nome = st.text_input("Nome Completo*")
        email = st.text_input("Email*")
        telefone = st.text_input("Telefone")
        data_nascimento = st.date_input("Data de Nascimento")
        usuario = st.text_input("Nome de Usuário*")
        senha = st.text_input("Senha*", type="password")
        confirmar = st.text_input("Confirme a Senha*", type="password")

        if st.button("Cadastrar"):
            if senha != confirmar:
                st.error("As senhas não coincidem.")
            else:
                db = Session()
                try:
                    result = db.execute(
                        text("SELECT 1 FROM usuarios WHERE usuario = :usuario OR email = :email"),
                        {"usuario": usuario, "email": email}
                    )
                    if result.fetchone():
                        st.error("Usuário ou email já cadastrado!")
                    else:
                        senha_hash = pbkdf2_sha256.hash(senha)
                        result = db.execute(text("""
                            INSERT INTO usuarios (
                                nome_completo, email, telefone, data_nascimento, 
                                senha, usuario, tipo, id_plano, dt_cadastro
                            ) VALUES (
                                :nome, :email, :telefone, :data_nascimento, 
                                :senha, :usuario, 'cliente', 1, now()
                            ) RETURNING id
                        """), {
                            "nome": nome,
                            "email": email,
                            "telefone": telefone,
                            "data_nascimento": str(data_nascimento),
                            "senha": senha_hash,
                            "usuario": usuario
                        })
                        id_cliente = result.scalar()

                        hoje = datetime.now()
                        expiracao = hoje + timedelta(days=30)

                        db.execute(text("""
                            INSERT INTO client_plans (
                                id_client, id_plano, ativo, data_inclusao, data_expira_plan
                            ) VALUES (
                                :id_client, 1, true, :data_inclusao, :data_expira_plan
                            )
                        """), {
                            "id_client": id_cliente,
                            "data_inclusao": hoje.strftime('%Y-%m-%d'),
                            "data_expira_plan": expiracao.strftime('%Y-%m-%d')
                        })
                        db.commit()
                        st.markdown("""
                            <div style='
                                position: fixed;
                                top: 0; left: 0;
                                width: 100%; height: 100%;
                                background-color: rgba(0, 0, 0, 0.7);
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                z-index: 9999;
                            '>
                                <div style='
                                    background: white;
                                    padding: 30px 40px;
                                    border-radius: 10px;
                                    text-align: center;
                                    box-shadow: 0 0 15px rgba(0,0,0,0.3);
                                    font-family: Poppins, sans-serif;
                                '>
                                    <div style='font-size: 40px;'>✅</div>
                                    <h4 style='color:#0b450b;'>Cadastro realizado com sucesso!</h4>
                                    <p style='font-size: 13px;'>Clique abaixo para continuar.</p>
                                    <a href='/' style='
                                        display: inline-block;
                                        margin-top: 15px;
                                        padding: 8px 16px;
                                        background-color: #0b450b;
                                        color: white;
                                        border-radius: 5px;
                                        text-decoration: none;
                                        font-size: 14px;
                                    '>Voltar ao Menu</a>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    db.rollback()
                    st.error(f"Erro no cadastro: {e}")
                finally:
                    db.close()

if 'admin' not in st.session_state:
# Inicializa variáveis no session_state para evitar erros de atributo inexistente

# LOGIN / CADASTRO 2

 if st.session_state.get("logged_in", False):
    nome_usuario = st.session_state.usuario.get("nome", "Usuário")
    st.sidebar.title(f"Bem-Vindo, {nome_usuario}")

    st.sidebar.markdown("""
        <div style='
            text-align: center; 
            padding: 8px 0; 
            font-size: 26px; 
            font-weight: bold; 
            color: green;
            border-bottom: 1px solid #DDD;
        '>FaixaBet®</div>
    """, unsafe_allow_html=True)
    try:
        print("Senha input?!?:", senha_input)
        print("Senha hash (tipo):", type(senha_hash))
        print("Senha hash (conteúdo):", senha_hash)
    except Exception as e:
        print("Erro ao imprimir senha:", e)
    
    opcao_selecionada = st.sidebar.radio("Menu", ["Dashboard", "Gerar Bets", "Histórico", "Validar", "Financeiro", "Editar Perfil", "Sair"])

    if opcao_selecionada == "Dashboard":
        mostrar_dashboard()
        # Exibe métricas de palpites do usuário logado
        dia, semana, mes = calcular_palpites_periodo(st.session_state.usuario["id"])
        st.markdown("---")
        st.metric("Palpites hoje", dia)
        st.metric("Palpites na semana", semana)
        st.metric("Palpites no mês", mes)
        
    elif opcao_selecionada == "Gerar Bets":
        gerar_palpite()
        
    elif opcao_selecionada == "Histórico":
        historico_palpites()
        
    elif opcao_selecionada == "Validar":
        validar_palpite()
        
    elif opcao_selecionada == "Financeiro":
        st.subheader("Financeiro") # Título da seção
        exibir_aba_financeiro()
        
    elif opcao_selecionada == "Editar Perfil":
        st.subheader("Editar") # Título da seção
        editar_perfil()
        
    elif opcao_selecionada == "Sair":
        logout() # Deve limpar a sessão e fazer st.rerun()

    # Rodapé da sidebar
 else:
    # Opcional: Mensagem se, por algum motivo, este código rodar sem o usuário estar logado
    st.warning("Você precisa estar logado para acessar o menu.")
    # Mova esta função inteira para antes da linha 339 (if opcao_selecionada == "Dashboard":)
   
import streamlit as st

def logout():
    # Limpa o estado da sessão
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Você saiu com sucesso!")
    st.experimental_rerun()

# --- FIM DO BLOCO DE LOGIN / CADASTRO ---

st.sidebar.markdown("<div style='text-align:left; color:green; font-size:16px;'>FaixaBet v8.01</div>", unsafe_allow_html=True)
