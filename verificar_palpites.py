import streamlit as st
import pandas as pd
from datetime import datetime
from db import Session

def buscar_resultado_oficial(data_str):
    """
    Busca um resultado oficial para a data fornecida no formato 'dd/mm/yyyy'.
    A tabela resultados_oficiais armazena a data como texto nesse formato.
    """
    session = Session()
    try:
        result = session.execute(
            text("SELECT * FROM resultados_oficiais WHERE data = :data"),
            {"data": data_str}
        )
        row = result.fetchone()
        return row
    except Exception as e:
        print(f"[ERRO] Falha ao buscar resultado oficial para {data_str}: {e}")
        return None
    finally:
        session.close()


def buscar_palpites_por_data(data_str):
    session = Session()
    try:
        data_obj = datetime.strptime(data_str, "%d/%m/%Y")
        data_inicio = data_obj.strftime("%Y-%m-%d 00:00:00")
        data_fim = data_obj.strftime("%Y-%m-%d 23:59:59")

        query = """
            SELECT * FROM palpites
            WHERE data BETWEEN :data_inicio AND :data_fim
        """

        df = pd.read_sql_query(
            query,
            con=session.bind,
            params={"data_inicio": data_inicio, "data_fim": data_fim}
        )

        return df
    except Exception as e:
        print(f"Erro ao buscar palpites: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def contar_acertos(palpites_df, numeros_oficiais):
    acertos = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    numeros_oficiais_set = set(map(int, numeros_oficiais))

    for _, row in palpites_df.iterrows():
        try:
            palpite_str = row['numeros']
            numeros_palpite = set(map(int, [n.strip() for n in palpite_str.split(',') if n.strip()]))
            qtd = len(numeros_palpite.intersection(numeros_oficiais_set))
            if qtd in acertos:
                acertos[qtd] += 1
        except Exception as e:
            st.warning(f"Palpite ignorado (erro ao ler): {e}")
            continue

    return acertos

# --- Interface Streamlit ---

st.set_page_config(page_title="sPainel de Acertos ", layout="centered")
st.markdown("<h1 style='color: green;'>Painel de Acertos - FaixaBet</h1>", unsafe_allow_html=True)

st.markdown("<h3>Escolha a Data do Concurso</h3>", unsafe_allow_html=True)
data_obj = st.date_input("", format="DD/MM/YYYY")
data_str = data_obj.strftime("%d/%m/%Y")

if st.button(" Verificar Palpites  🔍"):
    st.info(f" Data selecionada: **{data_str}**  🎯")

    resultado = buscar_resultado_oficial(data_str)
    if resultado:
        numeros_oficiais = resultado[2:17]  # n1 até n15
        st.success(f" Resultado oficial encontrado para o concurso {resultado[0]} ✅")
        st.markdown(
            f"<p style='font-size:20px;'>Números sorteados: <strong>{sorted(map(int, numeros_oficiais))}</strong></p>",
            unsafe_allow_html=True
        )

        palpites_df = buscar_palpites_por_data(data_str)
        total_palpites = len(palpites_df)

        if total_palpites == 0:
            st.warning("⚠️ Nenhum palpite encontrado nessa data.")
        else:
            st.info(f"Foram gerados **{total_palpites}** palpites em {data_str}. 📊 ")
            acertos = contar_acertos(palpites_df, numeros_oficiais)

            st.markdown("###  Estatísticas de Acertos  📈")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💥 15 acertos", acertos[15])
            col2.metric("🔥 14 acertos", acertos[14])
            col3.metric("✅ 13 acertos", acertos[13])
            col4.metric("🔸 12 acertos", acertos[12])
            col5.metric("🟡 11 acertos", acertos[11])

            percentual_15 = (acertos[15] / total_palpites) * 100
            st.markdown(f"**% de acertos 15 pontos:** `{percentual_15:.4f}%`  📈 ")

            if acertos[15] > 0:
                st.success(f"{acertos[15]} palpites acertaram os 15 números!  🎉 ")
            else:
                st.info("Nenhum palpite acertou os 15 números.")
    else:
        st.error(f"Nenhum resultado oficial encontrado para {data_str}. ❌ ")
