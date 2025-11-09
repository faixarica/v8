import streamlit as st
import requests

API_BASE = "https://faixabet-email-api.onrender.com"

st.title("🔐 Redefina sua Senha")

token = st.query_params.get("token", None)
if not token:
    st.error("Token inválido ou ausente.")
    st.stop()

st.success("Token validado! Defina sua nova senha abaixo.")

nova_senha = st.text_input("Nova Senha", type="password")
confirmar = st.text_input("Confirmar Nova Senha", type="password")

if st.button("Atualizar Senha"):
    if not nova_senha or not confirmar:
        st.error("Preencha os dois campos.")
    elif nova_senha != confirmar:
        st.error("As senhas não coincidem.")
    elif len(nova_senha) < 6:
        st.error("A senha deve ter pelo menos 6 caracteres.")
    else:
        try:
            r = requests.post(
                f"{API_BASE}/password/reset",
                json={"token": token, "new_password": nova_senha},
                timeout=10
            )
            if r.json().get("ok"):
                st.success("✅ Senha atualizada com sucesso! Faça login novamente.")
            else:
                st.error("❌ Erro: " + r.json().get("error", "Desconhecido"))
        except Exception as e:
            st.error(f"Erro na requisição: {e}")
