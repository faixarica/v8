# verificar_palpites.py
import streamlit as st
import pandas as pd
from datetime import datetime
from db import Session
from sqlalchemy import text

def buscar_resultado_oficial(data_str):
    """
    Busca o concurso mais recente para a data fornecida no formato 'dd/mm/yyyy'.
    Seleciona explicitamente colunas para garantir a ordem/nome das colunas.
    """
    session = Session()
    try:
        q = text("""
            SELECT concurso, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, data
            FROM resultados_oficiais
            WHERE data = :data
            ORDER BY concurso DESC
            LIMIT 1
        """)
        row = session.execute(q, {"data": data_str}).fetchone()
        return row
    except Exception as e:
        print(f"[ERRO] buscar_resultado_oficial: {e}")
        return None
    finally:
        session.close()


def buscar_palpites_por_data(data_str):
    """
    Tenta várias estratégias para localizar palpites do dia informado:
      1) data BETWEEN 'YYYY-MM-DD 00:00:00' AND 'YYYY-MM-DD 23:59:59' (quando campo é datetime)
      2) data = 'dd/mm/YYYY' (quando campo é texto nesse formato)
      3) to_date(data,'DD/MM/YYYY') = :date (Postgres) - tenta, mas tolera falha
      4) fallback: busca todos e filtra em pandas por substring
    Retorna (df, debug_messages_list)
    """
    session = Session()
    debug = []
    try:
        data_obj = datetime.strptime(data_str, "%d/%m/%Y")
        inicio = data_obj.strftime("%Y-%m-%d 00:00:00")
        fim = data_obj.strftime("%Y-%m-%d 23:59:59")

        # 1) Tentar BETWEEN (datetime)
        try:
            q1 = text("SELECT * FROM palpites WHERE data BETWEEN :inicio AND :fim")
            df1 = pd.read_sql_query(q1, con=session.bind, params={"inicio": inicio, "fim": fim})
            if not df1.empty:
                debug.append(f"Encontrado via BETWEEN ({len(df1)} palpites).")
                return df1, debug
            else:
                debug.append("BETWEEN retornou 0 registros.")
        except Exception as e:
            debug.append(f"BETWEEN query falhou: {e}")

        # 2) Tentar igualdade com 'dd/mm/YYYY' (texto)
        try:
            q2 = text("SELECT * FROM palpites WHERE data = :data_str")
            df2 = pd.read_sql_query(q2, con=session.bind, params={"data_str": data_str})
            if not df2.empty:
                debug.append(f"Encontrado via igualdade texto ({len(df2)} palpites).")
                return df2, debug
            else:
                debug.append("Igualdade (texto) retornou 0 registros.")
        except Exception as e:
            debug.append(f"Igualdade texto falhou: {e}")

        # 3) Tentar conversão via to_date (Postgres)
        try:
            # compara com date (YYYY-MM-DD)
            q3 = text("SELECT * FROM palpites WHERE to_date(data, 'DD/MM/YYYY') = :date_only")
            df3 = pd.read_sql_query(q3, con=session.bind, params={"date_only": data_obj.strftime("%Y-%m-%d")})
            if not df3.empty:
                debug.append(f"Encontrado via to_date (Postgres) ({len(df3)} palpites).")
                return df3, debug
            else:
                debug.append("to_date retornou 0 registros.")
        except Exception as e:
            debug.append(f"to_date tentativa falhou (provavelmente não-Postgres ou formato inesperado): {e}")

        # 4) Fallback: buscar todos e filtrar no pandas (último recurso)
        try:
            q4 = text("SELECT * FROM palpites")
            df_all = pd.read_sql_query(q4, con=session.bind)
            if not df_all.empty and "data" in df_all.columns:
                # Filtrar por substring que contenha a data_str; cobre diversos formatos
                mask = df_all["data"].astype(str).str.contains(data_str, na=False)
                df_filtered = df_all[mask]
                if not df_filtered.empty:
                    debug.append(f"Fallback: filtrado localmente ({len(df_filtered)} palpites).")
                    return df_filtered, debug
                else:
                    debug.append("Fallback: nenhum palpite encontrado após filtrar localmente.")
            else:
                debug.append("Fallback: tabela palpites vazia ou sem coluna 'data'.")
        except Exception as e:
            debug.append(f"Fallback (ler tudo) falhou: {e}")

        # nenhum encontrado
        return pd.DataFrame(), debug
    finally:
        session.close()


def contar_acertos_e_anotar(df, numeros_oficiais):
    """
    Adiciona coluna 'acertos' no df (número de interseções entre palpite e numeros_oficiais).
    Retorna (acertos_totais_dict, df).
    """
    acertos_totais = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    # normaliza numeros_oficiais: remove None/0 e garante ints
    nums = []
    for n in numeros_oficiais:
        try:
            if n is None:
                continue
            ni = int(n)
            if ni > 0:
                nums.append(ni)
        except Exception:
            continue
    numeros_oficiais_set = set(nums)

    lista_acertos = []
    for _, row in df.iterrows():
        try:
            pal = str(row.get("numeros", ""))  # coluna que guarda os numeros do palpite
            numeros_palpite = set(int(x.strip()) for x in pal.split(",") if x.strip())
            qtd = len(numeros_palpite.intersection(numeros_oficiais_set))
            lista_acertos.append(qtd)
            if qtd in acertos_totais:
                acertos_totais[qtd] += 1
        except Exception:
            lista_acertos.append(0)

    df = df.copy()
    df["acertos"] = lista_acertos
    return acertos_totais, df


# ---------------- Interface Streamlit ----------------
st.set_page_config(page_title="Painel de Acertos", layout="centered")
st.markdown("<h1 style='color: green;'>Painel de Acertos - FaixaBet</h1>", unsafe_allow_html=True)
st.markdown("<h3>Escolha a Data do Concurso</h3>", unsafe_allow_html=True)

data_obj = st.date_input("", format="DD/MM/YYYY")
data_str = data_obj.strftime("%d/%m/%Y")

if st.button(" Verificar Palpites  🔍"):
    st.info(f" Data selecionada: **{data_str}**  🎯")

    resultado = buscar_resultado_oficial(data_str)
    if not resultado:
        st.error(f"Nenhum resultado oficial encontrado para {data_str}. ❌ ")
    else:
        # extrai números oficiais pelos nomes das colunas (n1..n15)
        try:
            numeros_oficiais = [resultado[f"n{i}"] for i in range(1, 16)]
        except Exception:
            # fallback caso o Row não aceite indexação por nome (pou provável porque query selecionou explicitamente)
            valores = list(resultado)
            # sabemos que pos 0 = concurso, pos 1..15 = n1..n15
            numeros_oficiais = valores[1:16]

        # limpa e apresenta os números sorteados
        numeros_sorted = sorted([int(x) for x in numeros_oficiais if x is not None and int(x) > 0])

        # Pega o número do concurso de forma segura (funciona para Row ou tupla)
        try:
            concurso = resultado._mapping["concurso"]
        except (AttributeError, KeyError, TypeError):
            concurso = resultado[0]

        st.success(f" Resultado oficial encontrado para o concurso {concurso} ✅")
        st.markdown(
            f"<p style='font-size:20px;'>Números sorteados: <strong>{numeros_sorted}</strong></p>",
            unsafe_allow_html=True
        )

        # buscar palpites com múltiplas estratégias
        palpites_df, debug_msgs = buscar_palpites_por_data(data_str)
        for m in debug_msgs:
            st.caption(m)

        total_palpites = len(palpites_df)
        if total_palpites == 0:
            st.warning("⚠️ Nenhum palpite encontrado nessa data.")
            st.info("Tente confirmar o formato do campo 'data' na tabela palpites (DATETIME vs texto 'dd/mm/YYYY').")
        else:
            st.info(f"Foram gerados **{total_palpites}** palpites em {data_str}. 📊 ")

            acertos, palpites_df = contar_acertos_e_anotar(palpites_df, numeros_oficiais)

            # Estatísticas
            st.markdown("###  Estatísticas de Acertos  📈")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💥 15 acertos", acertos.get(15, 0))
            col2.metric("🔥 14 acertos", acertos.get(14, 0))
            col3.metric("✅ 13 acertos", acertos.get(13, 0))
            col4.metric("🔸 12 acertos", acertos.get(12, 0))
            col5.metric("🟡 11 acertos", acertos.get(11, 0))

            # percentuais seguros
            if total_palpites > 0:
                percentual_15 = (acertos.get(15, 0) / total_palpites) * 100
            else:
                percentual_15 = 0.0
            st.markdown(f"**% de acertos 15 pontos:** `{percentual_15:.4f}%`  📈 ")

            if acertos.get(15, 0) > 0:
                st.success(f"{acertos[15]} palpites acertaram os 15 números!  🎉 ")
            else:
                st.info("Nenhum palpite acertou os 15 números.")

            # Grid com palpites ordenado por acertos (mostrar colunas úteis)
            st.markdown("### Palpites detalhados 📋")
            # reorganiza colunas para exibir acertos e outras colunas comuns (ajuste se sua tabela tiver outros nomes)
            cols_display = ["id", "id_usuario", "numeros", "data", "plano", "acertos"]
            available = [c for c in cols_display if c in palpites_df.columns]
            # mostra dataframe ordenado
            st.dataframe(palpites_df.sort_values(by="acertos", ascending=False)[available + 
                         [c for c in palpites_df.columns if c not in available]].reset_index(drop=True))
