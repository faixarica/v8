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

# -------------------- CONFIG (precisa ser o PRIMEIRO comando Streamlit) --------------------
st.set_page_config(
    page_title="fAIxaBet",
    page_icon="🍀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- CONTROLE DE SESSÃO --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "recover_step" not in st.session_state:
    st.session_state.recover_step = 0

# Detecta token na URL (reset de senha)
query_params = st.query_params

if "token" in query_params:
    token = query_params["token"]
    st.session_state.token_reset = token
    st.session_state.recover_step = 2
    st.query_params.clear()  # remove o token da URL (importante)
    st.rerun()  # força redesenho já no modo "redefinir senha"


if "recover_step" not in st.session_state:
   st.session_state.recover_step = 0



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
# LOGIN + RECUPERAÇÃO DE SENHA
# =========================================================
if not st.session_state.get("logged_in", False):

    # Esconde sidebar enquanto não logado
    st.sidebar.empty()
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    # Inicializa estado do fluxo (0=login, 1=recuperar email, 2=redefinir)
    if "recover_step" not in st.session_state:
        st.session_state.recover_step = 0

    # Mensagem após solicitar recuperação
    if st.session_state.get("last_recover_message", False):
        st.success("✅ Se o e-mail existir, enviamos o link de recuperação.")
        st.session_state.last_recover_message = False

    st.markdown("## LOGIN")

    # ======================== PASSO 0 → LOGIN NORMAL ========================
    if st.session_state.recover_step == 0:

        with st.form("login_form"):
            usuario_input = st.text_input("Usuário")
            senha_input = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Conectar", use_container_width=True)

        if submitted:
            db = Session()
            sucesso_login = False
            try:
                # Adm master
                if usuario_input == "ufaixa990" and senha_input == "ufaixa990!":
                    st.session_state.logged_in = True
                    st.session_state.usuario = {
                        "id": 0, "nome": "Administrador", "email": "adm@faixabet.com",
                        "tipo": "admin", "id_plano": 0
                    }
                    st.session_state.admin = True
                    sucesso_login = True

                else:
                    result = db.execute(text("""
                        SELECT u.id, u.tipo, u.usuario, u.email, u.senha, u.ativo, u.id_plano
                        FROM usuarios u WHERE u.usuario = :usuario
                    """), {"usuario": usuario_input})
                    user = result.fetchone()

                    if user:
                        id, tipo, usuario, email, senha_hash, ativo, id_plano_armazenado = user

                        if senha_hash and verificar_senha(senha_input, senha_hash, db=db, user_id=id):
                            if ativo:
                                st.session_state.logged_in = True
                                st.session_state.usuario = {
                                    "id": id, "nome": usuario, "email": email,
                                    "tipo": tipo, "id_plano": id_plano_armazenado
                                }
                                if tipo == "admin":
                                    st.session_state.admin = True
                                registrar_login(id)
                                sucesso_login = True
                            else:
                                st.error("Conta inativa.")
                        else:
                            st.error("Senha incorreta.")
                    else:
                        st.error("Usuário não encontrado.")

            except Exception as e:
                if "Cancelled" not in str(e):
                    st.error(f"Erro no login: {e}")
            finally:
                db.close()

            if sucesso_login:
                st.rerun()

        # Botão esqueci a senha
        if st.button("Esqueceu a senha?"):
            st.session_state.recover_step = 1
            st.rerun()

    # ======================== PASSO 1 → ENVIAR LINK ========================
    elif st.session_state.recover_step == 1:

        st.markdown("### 🔑 Recuperar Senha")

        with st.form("recover_form"):
            email_rec = st.text_input("E-mail cadastrado")
            col1, col2 = st.columns([1,1])
            submitted = col1.form_submit_button("Enviar link de recuperação")
            cancel = col2.form_submit_button("Voltar ao login")

        if submitted:
            if email_rec.strip():
                import requests
                try:
                    requests.post(
                        f"{API_BASE}/password/forgot",
                        json={"email": email_rec},
                        timeout=10
                    )
                    st.session_state.last_recover_message = True
                    st.session_state.recover_step = 0
                    st.rerun()
                except Exception as e:
                    if "RerunData" not in str(e):
                        st.error(f"Erro ao enviar solicitação: {e}")
            else:
                st.error("Digite um e-mail.")

        if cancel:
            st.session_state.recover_step = 0
            st.rerun()

    # ======================== PASSO 2 → DIGITAR NOVA SENHA ========================
    elif st.session_state.recover_step == 2:

        st.markdown("### 🔒 Redefinir Senha")

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
                    timeout=10
                )
                if r.ok:
                    st.success("✅ Senha alterada! Faça login.")
                    st.session_state.recover_step = 0
                    st.session_state.token_reset = None
                    st.rerun()
                else:
                    st.error("❌ Erro ao redefinir a senha. Tente novamente.")

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
    st.sidebar.markdown("<div style='text-align:left; color:green; font-size:16px;'>fAIxaBet v8.15</div>", unsafe_allow_html=True)
