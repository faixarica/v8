from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import os
import stripe
from dotenv import load_dotenv
from stripe_webhook import carregar_planos

from database import (
    update_invoice_status_by_stripe_id,
    update_invoice_payment_flag_by_stripe_id,
    update_client_plans_status,
    update_plan_dates,
    insert_client_plan)

# Teste o caminho absoluto do arquivo atual
print(f"📂 Caminho do arquivo atual: {os.path.abspath(__file__)}")

# Agora força o caminho completo do .env, independente de onde está o script:
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

print(f"📄 Caminho final do .env: {env_path}")

# Carrega o .env do caminho exato
load_dotenv(dotenv_path=env_path)

# Verifica o carregamento
print(f"🔧 Segredo carregado 1: {os.getenv('STRIPE_ENDPOINT_SECRET')}")

# Configuração
app = Flask(__name__)
#load_dotenv()


stripe.api_key = os.getenv("STRIPE_API_KEY")
endpoint_secret = os.getenv("STRIPE_ENDPOINT_SECRET")

print(f"🔧 Segredo carregado 2 : {os.getenv('STRIPE_ENDPOINT_SECRET')}")


@app.route("/api/stripe-webhook", methods=["POST"])
def stripe_webhook():
    print("✅ Webhook recebeu POST!")

    payload = request.data
    sig_header = request.headers.get("stripe-signature")

    print(f"🔧 Payload recebido: {payload}")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except stripe.error.SignatureVerificationError as e:
        return jsonify({"error": "Assinatura inválida"}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]  # 👈 movido para cima

        print("📌 DEBUG WEBHOOK")
        print(f"🔑 Session ID: {session.get('id')}")
        print(f"🧾 Payment Intent: {session.get('payment_intent')}")
        print(f"📎 Metadata: {session.get('metadata')}")

        metadata = session.get("metadata", {})
        session_id = session.get("id")

        client_id = metadata.get("client_id")
        plan_id = metadata.get("plan_id")

        if not client_id or not plan_id:
            return jsonify({"error": "Metadados ausentes"}), 400

        client_id = int(client_id)
        plan_id = int(plan_id)

        # Atualizações
        update_invoice_status_by_stripe_id(session_id, status="Pago")
        update_invoice_payment_flag_by_stripe_id(session_id, flag="S")

        start_date = datetime.today().date()
        expiration_date = start_date + timedelta(days=30)

        update_client_plan_status(client_id, status="Ativo")
        update_client_plans_status(client_id, plan_id, status="A")
        update_plan_dates(client_id, plan_id, start_date, expiration_date)
        insert_client_plan(client_id, plan_id, start_date, expiration_date)

    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    app.run(port=5001)
