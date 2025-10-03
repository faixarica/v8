import streamlit as st
import requests
from datetime import datetime
from sqlalchemy.sql import text
from db import Session
import webbrowser

BACKEND_URL = "https://backend-v8.onrender.com"

def exibir_aba_financeiro():
    if 'usuario' not in st.session_state or st.session_state.usuario is None:
        st.error("Você precisa estar logado.")
        return

    user_id = st.session_state.usuario["id"]
    plano_id = st.session_state.usuario["id_plano"]

    db = Session()
    try:
        # 1. Carrega todos os planos
        result = db.execute(text("SELECT id, nome, valor, palpites_dia, loteria FROM planos"))
        planos_raw = result.fetchall()
        planos = {p[0]: {"nome": p[1], "valor": p[2], "palpites": p[3], "loteria": p[4]} for p in planos_raw}

        if not planos:
            st.error("Erro: Não foi possível carregar os planos disponíveis.")
            return

        # 2. Detalhes do plano atual
        plano_atual = planos.get(plano_id, {"nome": "Desconhecido", "valor": 0, "palpites": 0, "loteria": "Desconhecido"})
        st.subheader("Informações Financeiras")
        st.markdown(f"""
        <div style="background-color:#f3f3f3; padding: 20px; border-radius: 12px; border: 1px solid #ccc">
            <h4 style="margin: 0 0 10px 0;"><b>Plano Atual:</b> {plano_atual["nome"]}</h4>
            <p style="margin: 0;"><b>- Valor Mensal:</b> R$ {plano_atual["valor"]:.2f}</p>
            <p style="margin: 0;"><b>- Palpites Disponíveis/mês:</b> {plano_atual["palpites"]}</p>
            <p style="margin: 0;"><b>- Loteria:</b> {plano_atual["loteria"]}</p>
        </div>
        """, unsafe_allow_html=True)

        # 3. Últimos pagamentos
        pagamentos = db.execute(text("""
            SELECT data_pagamento, forma_pagamento, valor, data_validade, id_plano
            FROM financeiro
            WHERE id_cliente = :uid
            ORDER BY data_pagamento DESC
            LIMIT 5
        """), {"uid": user_id}).fetchall()

        if pagamentos:
            st.markdown("### Seus Pagamentos")
            for data_pgto, forma, valor, validade, plano_pgto_id in pagamentos:
                nome_plano = planos.get(plano_pgto_id, {}).get("nome", "Desconhecido")
                st.write(f"- [{data_pgto}] {nome_plano} - R$ {valor:.2f} via {forma} | válido até {validade}")
        else:
            st.info("Nenhum Pagamento Registrado Ainda.")

        # 4. Alterar/assinar novo plano
        st.markdown("---")
        st.markdown("### 💳 Alterar Plano e Efetuar Pagamento")

        nomes_planos_disponiveis = [planos[p]["nome"] for p in planos if p != plano_id]
        nome_to_id = {planos[p]["nome"]: p for p in planos}

        if nomes_planos_disponiveis:
            with st.form("form_novo_plano"):
                novo_plano_nome = st.selectbox("Escolha um Novo Plano", nomes_planos_disponiveis)
                submitted = st.form_submit_button("Prosseguir para Pagamento")

            if submitted:
                novo_id = nome_to_id[novo_plano_nome]
                valor_plano = planos[novo_id]["valor"]
                plano_backend = planos[novo_id]["nome"].strip().lower()

                if valor_plano > 0:
                    try:
                        # 🔹 rota especial para troca de plano (usuário já existe)
                        resp = requests.post(
                        f"{BACKEND_URL}/api/change-plan",
                        json={
                            "user_id": user_id,
                            "plan": plano_backend,
                            "email": st.session_state.usuario["email"],
                            "full_name": st.session_state.usuario.get("nome_completo", ""),
                            "username": st.session_state.usuario.get("usuario", "")
                        },
                        timeout=10
                    )

                        if resp.status_code == 200:
                            data = resp.json()
                            checkout_url = data.get("checkoutUrl")
                            if checkout_url:
                                st.success(f"Mudança para {novo_plano_nome} iniciada!")
                                st.markdown(f"[Clique aqui para pagar com segurança]({checkout_url})")
                                webbrowser.open_new_tab(checkout_url)
                            else:
                                st.error("Erro: não foi possível obter URL do checkout.")
                        else:
                            st.error(f"Erro ao trocar de plano: {resp.text}")

                    except Exception as e:
                        st.error(f"Falha ao trocar de plano: {e}")
                else:
                    st.warning("Plano gratuito não requer pagamento.")
        else:
            st.info("Você já está no plano mais básico ou não há outros planos disponíveis.")

        # 5. Cancelar plano pago
        st.markdown("---")
        st.markdown("### ❌ Cancelar Plano Pago")
        if plano_id != 1:
            if st.button("Cancelar Plano e Voltar para o FREE"):
                try:
                    # 1. Atualiza usuários
                    db.execute(text("UPDATE usuarios SET id_plano = 1 WHERE id = :uid"), {"uid": user_id})
                    # 2. Marca todos os client_plans ativos como inativos
                    db.execute(text("UPDATE client_plans SET ativo = false WHERE id_client = :uid AND ativo = true"), {"uid": user_id})
                    db.commit()

                    # Atualiza session_state
                    st.session_state.usuario["id_plano"] = 1
                    st.success("Plano Cancelado. Agora Você Está no Plano FREE.")
                    st.experimental_rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Erro ao cancelar o plano: {e}")
        else:
            st.info("Você já está no Plano FREE.")
    except Exception as e:
        st.error(f"Erro inesperado ao carregar área financeira: {e}")
    finally:
        db.close()
