import os,secrets,smtplib, requests,base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pds
import hashlib
import requests

from passlib.hash import pbkdf2_sha256
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from db import Session
from dashboard import mostrar_dashboard
from palpites import gerar_palpite_ui, historico_palpites, validar_palpite
from auth import logout
from perfil import editar_perfil
from financeiro import exibir_aba_financeiro
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlparse, parse_qs



# Força a sidebar sempre escondida enquanto não houver login
if not st.session_state.get("logged_in", False):
    st.set_page_config(initial_sidebar_state="collapsed")
    hide_sidebar = """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar, unsafe_allow_html=True)

# Detecta token na URL (reset de senha)
query_params = st.query_params
if "token" in query_params:
    st.session_state.token_reset = query_params["token"]
    st.session_state.recover_step = 2

if "recover_step" not in st.session_state:
   st.session_state.recover_step = 0

# -------------------- [2] CONFIGS --------------------
st.set_page_config(page_title="fAIxaBet", initial_sidebar_state="collapsed")
st.sidebar.empty()

st.set_page_config(page_title="fAIxaBet", layout="centered")
# Cabeçalho fixo
st.markdown("""
    <div style='
        width: 100%; 
        text-align: center; 
        padding: 6px 0; 
        font-size: 44px; 
        font-weight: bold; 
        color: green;
        border-bottom: 1px solid #DDD;
    '>Bem-vindo à fAIxaBet®
        <hr style="margin: 0; border: 0; border-top: 1px solid #DDD;">
        <div style='text-align:center; font-size:19px; color:black; margin-top:4px;'>
            O Futuro da Loteria é Prever
        </div>
    </div>
""", unsafe_allow_html=True)

#  "🔹 Sorte é Aleatória. Aqui é Inteligência.",
#  "Previsões baseadas em dados reais.",
#  "O Futuro da Loteria é Prever.",
#  "Gere seus palpites com o poder da IA.",
#  "FaixaBet — Onde os Números Pensam.",
#  "Inteligência. Não sorte."


# ✅ Backend API rodando externamente (email_api.py)
API_BASE ="https://faixabet-email-api.onrender.com"

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

# Função para enviar e-mail (coloque fora de qualquer bloco condicional)

def enviar_email_recuperacao(destinatario, token):
    try:
        # Configurações do servidor SMTP (exemplo com Gmail)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        remetente_email = "seu_email@gmail.com"  # Substitua pelo seu e-mail
        senha_email = "sua_senha_app"  # Substitua pela sua App Password do Gmail
        
        # Criar mensagem
        msg = MIMEMultipart()
        msg['From'] = remetente_email
        msg['To'] = destinatario
        msg['Subject'] = "Recuperação de Senha - fAIxaBet"
        
        # Corpo do e-mail
        corpo = f"""
        Olá,
        
        Você solicitou a recuperação de senha.
        Use o seguinte token para redefinir sua senha: {token}
        
        Este token é válido por 1 hora.
        
        Se você não solicitou isso, ignore este e-mail.
        
        Atenciosamente,
        Equipe fAIxaBet
        """
        msg.attach(MIMEText(corpo, 'plain'))
        
        # Enviar e-mail
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(remetente_email, senha_email)
        texto = msg.as_string()
        server.sendmail(remetente_email, destinatario, texto)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")
        return False

# -------------------- [4] APLICAÇÃO STREAMLIT --------------------
# =========================================================
# Controle de visibilidade do "modal" (container)
# =========================================================
if "show_recover_modal" not in st.session_state:
    st.session_state.show_recover_modal = False
    
# =========================================================
# Função para gerar token de recuperação
# =========================================================
def gerar_token_recuperacao(user_id, db):
    token = secrets.token_urlsafe(32)
    validade = datetime.utcnow() + timedelta(hours=1)
    try:
        db.execute(text("""
            INSERT INTO password_resets (user_id, token, expira_em)
            VALUES (:user_id, :token, :expira_em)
            ON CONFLICT (user_id) DO UPDATE
            SET token = EXCLUDED.token, expira_em = EXCLUDED.expira_em
        """), {"user_id": user_id, "token": token, "expira_em": validade})
        db.commit()
        return token
    except Exception as e:
        db.rollback()
        st.error(f"Erro ao gerar token de recuperação: {e}")
        return None

# =========================================================
# Função para verificar senha hashPasswordPBKDF2 e
# =========================================================

def verify_pbkdf2_legacy(password, hash_string):
    """
    Tenta validar o hash pbkdf2_sha256 em formatos "legados":
     - salt em base64 (unpadded)
     - salt em hex (bytes.fromhex)
     - salt como string ascii (caso o server tenha passado a representação textual)
    Retorna True se alguma tentativa bater.
    """
    try:
        parts = hash_string.split('$')
        if len(parts) != 4:
            return False
        _, rounds_str, salt_str, checksum = parts
        rounds = int(rounds_str)
    except Exception:
        return False

    chk_norm = checksum.rstrip('=')

    candidates = []
    # 1) tentar decodificar salt como base64 (corrige unpadded)
    try:
        padded = salt_str + ('=' * (-len(salt_str) % 4))
        candidates.append(base64.b64decode(padded))
    except Exception:
        pass
    # 2) tentar decodificar salt como hex
    try:
        candidates.append(bytes.fromhex(salt_str))
    except Exception:
        pass
    # 3) salt como bytes da string (caso o node tenha passado ascii)
    candidates.append(salt_str.encode('utf-8'))

    for salt_bytes in candidates:
        try:
            derived = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt_bytes, rounds, dklen=32)
            derived_b64 = base64.b64encode(derived).decode('ascii').rstrip('=')
            if derived_b64 == chk_norm:
                return True
        except Exception:
            continue
    return False

def verificar_senha(senha_digitada, senha_hash, db=None, user_id=None):
    from passlib.hash import pbkdf2_sha256
    import bcrypt

    if not senha_hash:
        return False

    # Limpeza automática: remove espaços, quebras de linha e retornos de carro
    senha_hash_clean = senha_hash.strip().replace("\n", "").replace("\r", "")

    # PBKDF2 (passlib)
    if senha_hash_clean.startswith("pbkdf2_sha256$") or senha_hash_clean.startswith("$pbkdf2-sha256$"):
        try:
            return pbkdf2_sha256.verify(senha_digitada, senha_hash_clean)
        except Exception:
            return False

    # bcrypt (mantém como está)
    elif senha_hash_clean.startswith("$2a$") or senha_hash_clean.startswith("$2b$"):
        return bcrypt.checkpw(senha_digitada.encode(), senha_hash_clean.encode())

    return False

# =========================================================
# Login
# =========================================================
if not st.session_state.get("logged_in", False):
    st.markdown("## LOGIN")
    aba = st.radio("Ação", ["Conectar"], horizontal=True, label_visibility="collapsed")
    #     aba = st.radio("Ação", ["Conectar", "Cadastro"], horizontal=True, label_visibility="collapsed")

    st.write("")

    if aba == "Conectar":
        # Inicializar estados para o fluxo de recuperação
        if "recover_step" not in st.session_state:
            st.session_state.recover_step = 0  # 0 = login normal, 1 = recuperar email, 2 = redefinir senha

        # Mostrar login normal ou fluxo de recuperação
        if st.session_state.recover_step == 0:
            # === Formulário de Login ===
            with st.form("login_form"):
                usuario_input = st.text_input("Usuário")
                senha_input = st.text_input("Senha", type="password")

                # CSS para o botão Conectar
                st.markdown("""
                    <style>
                        div.stButton > button:first-child {
                            width: 100% !important;
                            background-color:   !important;
                            color: white !important;
                            font-weight: 600 !important;
                            border: none !important;
                            border-radius: 6px !important;
                            height: 48px !important;
                        }
                        div.stButton > button:hover {
                            background-color: #3c7e2d !important;
                        }
                    </style>
                """, unsafe_allow_html=True)

                submitted = st.form_submit_button("Conectar", use_container_width=True)
               
                if submitted:
                    db = Session()
                    sucesso_login = False
                    try:
                        # Login admin
                        if usuario_input == "ufaixa990" and senha_input == "ufaixa990!":
                            st.session_state.logged_in = True
                            st.session_state.usuario = {
                                "id": 0, "nome": "Administrador", "email": "adm@faixabet.com",
                                "tipo": "admin", "id_plano": 0
                            }
                            st.session_state.admin = True
                            st.success("Login administrativo realizado!")
                            sucesso_login = True
                        else:
                            # Usuário normal
                            result = db.execute(text("""
                                SELECT u.id, u.tipo, u.usuario, u.email, u.senha, u.ativo, u.id_plano
                                FROM usuarios u WHERE u.usuario = :usuario
                            """), {"usuario": usuario_input})
                            user = result.fetchone()

                            if user:
                                id, tipo, usuario, email, senha_hash, ativo, id_plano_armazenado = user
                                result = db.execute(text("""
                                    SELECT cp.id_plano FROM client_plans cp
                                    WHERE cp.id_client = :id AND cp.ativo = true
                                    ORDER BY cp.data_inclusao DESC LIMIT 1
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
                                        db.rollback()
                                        st.warning("Não foi possível atualizar o plano do usuário.")

                                # agora (passando db e id para permitir migração)
                                #st.write("DEBUG senha_hash raw:", repr(senha_hash))
                                #senha_hash = senha_hash.strip().replace("\n", "").replace("\r", "")
                                #st.write("DEBUG senha_hash clean:", repr(senha_hash))

                                if senha_hash and verificar_senha(senha_input, senha_hash, db=db, user_id=id):

                                    if ativo:
                                        st.session_state.logged_in = True
                                        st.session_state.usuario = {
                                            "id": id, "nome": usuario, "email": email,
                                            "tipo": tipo, "id_plano": id_plano_atual
                                        }
                                        if tipo == "admin":
                                            st.session_state.admin = True
                                        registrar_login(id)
                                        st.success("Login realizado com sucesso!")
                                        sucesso_login = True
                                    else:
                                        st.error("Conta inativa.")
                                else:
                                    st.error("Senha incorreta.")
                            else:
                                st.error("Usuário não encontrado.")
                    except Exception as e:
                        if "Cancelled" not in str(e):
                            st.error(f"Erro durante o login: {e}")
                    finally:
                        db.close()

                    if sucesso_login:
                        st.rerun()

            # === Botão de Recuperação ===
            # === Botão de Recuperação (corrigido) ===
            st.markdown("""
                <style>
                .forgot-btn button {
                    color: #469536 !important;
                    background: none !important;
                    border: none !important;
                    font-weight: 500 !important;
                    text-decoration: underline !important;
                    cursor: pointer !important;
                    margin-top: 10px;
                }
                </style>
            """, unsafe_allow_html=True)

            col_esq = st.container()
            with col_esq:
                esqueci = st.button("Esqueceu a senha?", key="forgot_btn", help="Clique para redefinir sua senha.")
                if esqueci:
                    st.session_state.recover_step = 1
                    st.rerun()

        # === Etapa 1: Recuperação por E-mail ===
        elif st.session_state.recover_step == 1:
            st.markdown("### 🔑 Recuperar Senha")

            with st.form("recover_form"):
                email_rec = st.text_input("E-mail cadastrado")

                col1, col2 = st.columns([1,1])
                submitted = col1.form_submit_button("Enviar link de recuperação")
                cancel = col2.form_submit_button("Voltar ao login")

            if submitted:
                if not email_rec:
                    st.error("Por favor, insira um e-mail.")
                else:
                    import requests
                    try:
                        resp = requests.post(
                            f"{API_BASE}/password/forgot",
                            json={"email": email_rec},
                            timeout=10
                        )
                        st.session_state.recover_step = 0
                        st.session_state.last_recover_message = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Erro ao enviar solicitação: {e}")

            if cancel:
                st.session_state.recover_step = 0
                st.rerun()
        # ===================== TELA 2: DIGITAR NOVA SENHA =====================
        elif st.session_state.get("recover_step") == 2:
            st.markdown("### 🔒 Redefinir senha")

            token = st.session_state.get("token_reset", "")
            if not token:
                st.error("Token inválido ou expirado.")
                st.stop()

            with st.form("reset_form"):
                nova = st.text_input("Nova senha", type="password")
                conf = st.text_input("Confirmar senha", type="password")
                submit = st.form_submit_button("Salvar nova senha")

            if submit:
                if len(nova) < 6:
                    st.error("A senha deve ter pelo menos 6 caracteres.")
                elif nova != conf:
                    st.error("As senhas não coincidem.")
                else:
                    r = requests.post(
                        f"{API_BASE}/password/reset",
                        json={"token": token, "new_password": nova},
                        timeout=15
                    )
                    if r.ok:
                        st.success("✅ Sua senha foi alterada! Faça login novamente.")
                        # limpa fluxo
                        st.session_state.recover_step = 0
                        st.session_state.token_reset = None
                        st.rerun()
                    else:
                        # opcional: mostrar erro retornado
                        try:
                            err = r.json().get("error", "")
                        except Exception:
                            err = r.text
                    st.error(f"❌ Erro ao redefinir a senha. {err if err else 'Tente novamente.'}")


    elif aba == "Cadastro":
       # st.info("⚠️ Tela de cadastro ainda não implementada.")
        
        hoje = date.today()
        idade_minima = 18

        try:
            data_maxima = date(hoje.year - idade_minima, hoje.month, hoje.day)
        except ValueError:
            data_maxima = date(hoje.year - idade_minima, 2, 28)

        nome = st.text_input("Nome Completo*")
        email = st.text_input("Email*")
        telefone = st.text_input("Telefone")
        
        data_nascimento = st.date_input(
            "Data de Nascimento*",
            value=data_maxima,
            min_value=date(1900, 1, 1),
            max_value=data_maxima,
            help=f"Você deve ter pelo menos {idade_minima} anos"
        )
        
        usuario = st.text_input("Nome de Usuário*")
        senha = st.text_input("Senha*", type="password")
        confirmar = st.text_input("Confirme a Senha*", type="password")

        if st.button("Cadastrar"):
            if data_nascimento > data_maxima:
                st.error(f"Você deve ter pelo menos {idade_minima} anos.")
            elif senha != confirmar:
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
#========================== fim       

if 'admin' not in st.session_state:
    # Inicializa variáveis no session_state para evitar erros de atributo inexistente
    pass

# ==========================================================
# LOGIN / CADASTRO 2 — com suporte a múltiplas loterias
# ==========================================================
if st.session_state.get("logged_in", False):

    # --- Cabeçalho lateral ---
    usuario = st.session_state.get("usuario", {})
    nome_usuario = usuario.get("nome", "Usuário")
    tipo_user = usuario.get("tipo", "").upper()

    st.sidebar.title(f"Bem-vindo(a), {nome_usuario}")
    st.sidebar.markdown("""
        <div style='text-align:center; padding:8px 0; font-size:26px;
        font-weight:bold; color:green; border-bottom:1px solid #DDD;'>
            fAIxaBet®
        </div>
    """, unsafe_allow_html=True)

    # --- Pergunta a loteria que o usuário quer trabalhar ---
    st.sidebar.markdown("### Escolha a Loteria")
    loteria_escolhida = st.sidebar.selectbox(
        "Loteria:",
        ["Lotofácil", "Mega-Sena"],
        index=0
    )
    st.session_state["loteria"] = loteria_escolhida

    # =======================================================
    # MENU PRINCIPAL (apenas após login válido)
    # =======================================================
    if tipo_user in ["A", "ADM", "ADMIN"]:
        st.sidebar.markdown("### ⚙️ Painel Administrativo")
        menu_itens = [
            "Painel Estatístico",
            "Gerar Novas Bets",
            "Histórico",
            "Validar Bets Gerada",
            "Assinatura ",
            "Editar Perfil",
            "Telemetria",
            "Usuários",
            "Notificar",
            "Resultados",
            "Evolução",
            "Sair"
        ]
    else:
        menu_itens = [
            "Painel Estatístico",
            "Gerar Novas Bets",
            "Histórico",
            "Validar Bets Gerada",
            "Assinatura ",
            "Editar Perfil",
            "Sair"
        ]

    opcao_selecionada = st.sidebar.radio("Menu", menu_itens)

    # =======================================================
    # ROTEAMENTO DAS OPÇÕES
    # =======================================================
    if opcao_selecionada == "Painel Estatístico":
        mostrar_dashboard()
        dia, semana, mes = calcular_palpites_periodo(usuario["id"])
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Palpites Hoje", dia)
        col2.metric("Palpites na Semana", semana)
        col3.metric("Palpites no Mês", mes)

    elif opcao_selecionada in ["Gerar Novas Bets", "Histórico", "Validar Bets Gerada"]:
        # 🔸 Importa o módulo correto conforme a loteria escolhida
        if loteria_escolhida == "Mega-Sena":
            from mega.palpites_m import gerar_palpite_ui, historico_palpites, validar_palpite
        else:
            from palpites import gerar_palpite_ui, historico_palpites, validar_palpite

        if opcao_selecionada == "Gerar Novas Bets":
            gerar_palpite_ui()
        elif opcao_selecionada == "Histórico":
            historico_palpites()
        elif opcao_selecionada == "Validar Bets Gerada":
            validar_palpite()

    elif opcao_selecionada == "Assinatura ":
        exibir_aba_financeiro()

    elif opcao_selecionada == "Editar Perfil":
        editar_perfil()

    elif opcao_selecionada == "Telemetria":
        from dashboard import mostrar_telemetria
        mostrar_telemetria()

    elif opcao_selecionada == "Usuários":
        from admin.usuarios import listar_usuarios
        listar_usuarios()

    elif opcao_selecionada == "Notificar":
        from admin.notificacoes import enviar_notificacoes_acertos
        enviar_notificacoes_acertos()

    elif opcao_selecionada == "Resultados":
        import resultados
        resultados.importar_resultado()

    elif opcao_selecionada == "Evolução":
        import verificar_palpites
        verificar_palpites.executar_verificacao()

    elif opcao_selecionada == "Sair":
        logout()

# ==========================================================
# Se não estiver logado, mostra apenas o formulário de login
# ==========================================================

# --- FIM DO BLOCO DE LOGIN / CADASTRO ---
    st.sidebar.markdown("<div style='text-align:left; color:green; font-size:16px;'>fAIxaBet v8.14</div>", unsafe_allow_html=True)
