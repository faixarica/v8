import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st

def enviar_email_recuperacao(email_destino, token):
    """Envia o e-mail de recuperação de senha com token único."""
    try:
        # 🔹 Detectar ambiente (local vs Streamlit Cloud)
        if "EMAIL_USER" in st.secrets:
            EMAIL_HOST = st.secrets["EMAIL_HOST"]
            EMAIL_PORT = int(st.secrets["EMAIL_PORT"])
            EMAIL_USER = st.secrets["EMAIL_USER"]
            EMAIL_PASS = st.secrets["EMAIL_PASS"]
            EMAIL_USE_TLS = st.secrets.get("EMAIL_USE_TLS", "True") == "True"
        else:
            from dotenv import load_dotenv
            load_dotenv()
            EMAIL_HOST = os.getenv("EMAIL_HOST")
            EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
            EMAIL_USER = os.getenv("EMAIL_USER")
            EMAIL_PASS = os.getenv("EMAIL_PASS")
            EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True") == "True"

        # 🔹 Montar mensagem
        corpo_html = f"""
        <html>
        <body style="font-family: Poppins, sans-serif; color:#333;">
            <h2 style="color:#469536;">Recuperação de Senha — fAIxaBet</h2>
            <p>Olá! Recebemos uma solicitação para redefinir sua senha.</p>
            <p>Use o token abaixo para redefinir sua senha:</p>
            <div style="background:#f0f0f0; padding:10px; font-size:18px; font-weight:bold; width:fit-content;">
                {token}
            </div>
            <p>Se você não fez esta solicitação, ignore este e-mail.</p>
            <p style="font-size:13px; color:#777;">Atenciosamente,<br>Equipe fAIxaBet</p>
        </body>
        </html>
        """

        msg = MIMEText(corpo_html, "html")
        msg["Subject"] = "Recuperação de senha — fAIxaBet"
        msg["From"] = EMAIL_USER
        msg["To"] = email_destino

        # 🔹 Enviar
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
            if EMAIL_USE_TLS:
                smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)

        return True
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}")
        return False
    