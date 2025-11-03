from flask import Flask, request, jsonify
from flask_cors import CORS
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib, os

app = Flask(__name__)
CORS(app, origins=["https://faixabet.com.br", "http://127.0.0.1:5500"])

# -----------------------------------------
# Configurações de e-mail (Locaweb ou email-ssl)
# -----------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "email-ssl.com.br")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "sac@faixabet.com.br")
SMTP_PASS = os.getenv("SMTP_PASS", "*A*{.^9bue")
USE_TLS = True

# -----------------------------------------
# Rota principal de teste
# -----------------------------------------
@app.route("/")
def home():
    return "Backend fAIxaBet ativo ✅"

# -----------------------------------------
# Rota para envio de palpites por e-mail
# -----------------------------------------
@app.route("/send_palpite", methods=["POST"])
def send_palpite():
    try:
        data = request.get_json()
        email = data.get("email")
        sorteio = ", ".join(map(str, data.get("sorteio", [])))
        ai = ", ".join(map(str, data.get("ai", [])))
        jogador = ", ".join(map(str, data.get("jogador", [])))

        if not email:
            return jsonify({"status": "error", "message": "E-mail inválido"}), 400

        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = email
        msg["Subject"] = "🎯 Seus palpites do simulador fAIxaBet"

        corpo = f"""
        🎯 Sorteio: {sorteio}
        🤖 AI: {ai}
        👤 Jogador: {jogador}
        """

        msg.attach(MIMEText(corpo, "plain"))

        print(f"📨 Enviando e-mail para {email} via {SMTP_HOST}:{SMTP_PORT}")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if USE_TLS:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print("✅ E-mail enviado com sucesso!")
        return jsonify({"status": "ok", "message": "E-mail enviado com sucesso!"})

    except Exception as e:
        print("❌ Erro ao enviar e-mail:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------------------
# Inicialização (somente local)
# -----------------------------------------
if __name__ == "__main__":
    print("🚀 Servidor Flask iniciado na porta 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
