import streamlit as st
from datetime import datetime, timedelta
from sqlalchemy.sql import text
from db import Session

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

        # 4. Simular novo plano
        st.markdown("---")
        st.markdown("### 💳 Simular Novo Plano e Pagamento")

        nomes_planos_disponiveis = [planos[p]["nome"] for p in planos if p != plano_id]
        nome_to_id = {planos[p]["nome"]: p for p in planos}

        if nomes_planos_disponiveis:
            with st.form("form_simulacao_plano"):
                novo_plano_nome = st.selectbox("Escolha um Novo Plano", nomes_planos_disponiveis)
                forma_pagamento = st.selectbox("Forma de Pagamento", ["Cartão", "Débito", "Pix"])
                submitted = st.form_submit_button("Confirmar Simulação de Pagamento")

            if submitted:
                novo_id = nome_to_id[novo_plano_nome]
                novo_valor = planos[novo_id]["valor"]
                hoje = datetime.now()
                validade = hoje + timedelta(days=30)

                try:
                    db.execute(text("""
                        INSERT INTO financeiro (id_cliente, id_plano, data_pagamento, forma_pagamento, valor, data_validade)
                        VALUES (:uid, :pid, :data, :forma, :valor, :validade)
                    """), {
                        "uid": user_id,
                        "pid": novo_id,
                        "data": hoje,
                        "forma": forma_pagamento,
                        "valor": novo_valor,
                        "validade": validade.date()
                    })

                    db.execute(text("""
                        UPDATE usuarios SET id_plano = :pid WHERE id = :uid
                    """), {"pid": novo_id, "uid": user_id})

                    db.execute(text("""
                        UPDATE client_plans SET ativo = FALSE WHERE id_client = :uid AND ativo = TRUE
                    """), {"uid": user_id})

                    db.execute(text("""
                        INSERT INTO client_plans (id_client, id_plano, data_inclusao, data_expira_plan, ativo, palpites_dia_usado)
                        VALUES (:uid, :pid, :data, :validade, TRUE, 0)
                    """), {
                        "uid": user_id,
                        "pid": novo_id,
                        "data": hoje,
                        "validade": validade.date()
                    })
                    db.commit()

                    st.session_state.usuario["id_plano"] = novo_id
                    st.success(f"Pagamento do Plano {novo_plano_nome} Registrado Com Sucesso!")
                    st.rerun()

                except Exception as e:
                    db.rollback()
                    st.error(f"Erro ao processar o pagamento: {e}")
        else:
            st.info("Você já está no plano mais básico ou não há outros planos disponíveis.")

        # 5. Cancelar plano pago
        st.markdown("---")
        st.markdown("### ❌ Cancelar Plano Pago")

        if plano_id != 1:
            if st.button("Cancelar Plano e Voltar para o FREE"):
                try:
                    db.execute(text("UPDATE usuarios SET id_plano = 1 WHERE id = :uid"), {"uid": user_id})
                    db.execute(text("UPDATE client_plans SET ativo = 0 WHERE id_client = :uid AND ativo = 1"), {"uid": user_id})
                    db.commit()

                    st.session_state.usuario["id_plano"] = 1
                    st.success("Plano Cancelado. Agora Você Está no Plano FREE.")
                    st.rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Erro ao cancelar o plano: {e}")
        else:
            st.info("Você já está no Plano FREE.")

    except Exception as e:
        st.error(f"Erro inesperado ao carregar área financeira: {e}")
    finally:
        db.close()
