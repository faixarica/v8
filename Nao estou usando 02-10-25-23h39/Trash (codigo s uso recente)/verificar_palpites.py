# verificar_palpites_avancado.py
import streamlit as st
import pandas as pd
from datetime import datetime, date
from db import Session
from sqlalchemy import text

# =========================
# Helpers de parsing/normalização
# =========================
def _parse_date_any(v):
    """Converte v para date (robusto para str, date, datetime). Retorna None se não der."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        try:
            return pd.to_datetime(v).date()
        except Exception:
            return None
    s = str(v).strip()
    # tentativas comuns
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception:
            pass
    # fallback pandas
    try:
        x = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return None if pd.isna(x) else x.date()
    except Exception:
        return None

def _add_data_dia(df, col="data"):
    """Garante coluna 'data_dia' (tipo date) no df."""
    if df.empty or col not in df.columns:
        df["data_dia"] = pd.NaT
        return df
    df = df.copy()
    df["data_dia"] = df[col].apply(_parse_date_any)
    # padroniza para Timestamp para facilitar .dt
    df["data_dia"] = pd.to_datetime(df["data_dia"], errors="coerce")
    return df

def _filtrar_validados(df, somente_validados: bool):
    """Aplica filtro de 'validados' usando colunas comuns se existirem."""
    if not somente_validados or df.empty:
        return df, None

    df = df.copy()
    candidatos = ["validado", "aposta_confirmada", "confirmado", "foi_apostado", "aposta", "status", "status_aposta", "pago"]
    col_usada = None

    for c in candidatos:
        if c in df.columns:
            col_usada = c
            # normaliza valores para boolean
            vals = df[c].astype(str).str.strip().str.upper()
            truthy = {"1", "TRUE", "T", "S", "SIM", "Y", "YES", "OK", "CONFIRMADO", "CONFIRMADA", "PAGO", "EFETUADA", "CONCLUIDA", "CONCLUÍDA"}
            df = df[vals.isin(truthy)]
            break

    return df, col_usada
# _mask_modelos
def _mask_modelos(df):
    """
    Retorna máscara booleana para linhas cujo modelo seja:
      - LSTM
      - Comece com 'LS' (ex.: LS14, LS15)
    Checa várias colunas possíveis.
    """
    if df.empty:
        return pd.Series([], dtype=bool)
    cols = [c for c in ["modelo", "modelo_nome", "algoritmo", "metodo", "gerador", "origem"] if c in df.columns]
    if not cols:
        return pd.Series([False] * len(df))

    mask = pd.Series([False] * len(df))
    for c in cols:
        colu = df[c].astype(str).str.upper().str.strip()
        m = colu.str.startswith("LS") | (colu == "LSTM")
        mask = mask | m
    return mask

# =========================
# Consultas flexíveis (mês)
# =========================
def _query_resultados_mes(ano: int, mes: int):
    """
    Tenta:
      1) EXTRACT em coluna 'data' nativa (DATE/TIMESTAMP)
      2) EXTRACT em TO_DATE(data,'DD/MM/YYYY') (quando TEXT)
      3) Fallback: lê tudo e filtra no pandas
    Retorna (df, estrategia_str, debug_msgs)
    """
    debug = []
    session = Session()
    try:
        # 1) data nativa
        try:
            q = text("""
                SELECT concurso, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10,
                       n11, n12, n13, n14, n15, data
                FROM resultados_oficiais
                WHERE EXTRACT(YEAR FROM data) = :ano
                  AND EXTRACT(MONTH FROM data) = :mes
                ORDER BY data ASC
            """)
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Resultados: estratégia 1 (EXTRACT nativo).")
                return df, "nativo", debug
            debug.append("Resultados: estratégia 1 vazia.")
        except Exception as e:
            debug.append(f"Resultados: estratégia 1 falhou: {e}")

        # 2) data texto 'DD/MM/YYYY'
        try:
            q = text("""
                SELECT concurso, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10,
                       n11, n12, n13, n14, n15, data
                FROM resultados_oficiais
                WHERE EXTRACT(YEAR FROM TO_DATE(data, 'DD/MM/YYYY')) = :ano
                  AND EXTRACT(MONTH FROM TO_DATE(data, 'DD/MM/YYYY')) = :mes
                ORDER BY TO_DATE(data, 'DD/MM/YYYY') ASC
            """)
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Resultados: estratégia 2 (TO_DATE texto).")
                return df, "texto", debug
            debug.append("Resultados: estratégia 2 vazia.")
        except Exception as e:
            debug.append(f"Resultados: estratégia 2 falhou: {e}")

        # 3) fallback: tudo + filtro pandas
        try:
            q = text("""
                SELECT concurso, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10,
                       n11, n12, n13, n14, n15, data
                FROM resultados_oficiais
            """)
            df_all = pd.read_sql_query(q, con=session.bind)
            df_all = _add_data_dia(df_all, "data")
            ok = df_all["data_dia"].notna()
            df = df_all[ok & (df_all["data_dia"].dt.year == ano) & (df_all["data_dia"].dt.month == mes)].sort_values("data_dia")
            debug.append("Resultados: estratégia 3 (fallback pandas).")
            return df, "fallback", debug
        except Exception as e:
            debug.append(f"Resultados: estratégia 3 falhou: {e}")
            return pd.DataFrame(), "erro", debug
    finally:
        session.close()

def _query_palpites_mes(ano: int, mes: int):
    """Mesma lógica flexível para palpites."""
    debug = []
    session = Session()
    try:
        # 1) nativo
        try:
            q = text("""
                SELECT *
                FROM palpites
                WHERE EXTRACT(YEAR FROM data) = :ano
                  AND EXTRACT(MONTH FROM data) = :mes
            """)
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Palpites: estratégia 1 (EXTRACT nativo).")
                return df, "nativo", debug
            debug.append("Palpites: estratégia 1 vazia.")
        except Exception as e:
            debug.append(f"Palpites: estratégia 1 falhou: {e}")

        # 2) texto 'DD/MM/YYYY'
        try:
            q = text("""
                SELECT *
                FROM palpites
                WHERE EXTRACT(YEAR FROM TO_DATE(data, 'DD/MM/YYYY')) = :ano
                  AND EXTRACT(MONTH FROM TO_DATE(data, 'DD/MM/YYYY')) = :mes
            """)
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Palpites: estratégia 2 (TO_DATE texto).")
                return df, "texto", debug
            debug.append("Palpites: estratégia 2 vazia.")
        except Exception as e:
            debug.append(f"Palpites: estratégia 2 falhou: {e}")

        # 3) fallback pandas
        try:
            q = text("SELECT * FROM palpites")
            df_all = pd.read_sql_query(q, con=session.bind)
            df_all = _add_data_dia(df_all, "data")
            ok = df_all["data_dia"].notna()
            df = df_all[ok & (df_all["data_dia"].dt.year == ano) & (df_all["data_dia"].dt.month == mes)]
            debug.append("Palpites: estratégia 3 (fallback pandas).")
            return df, "fallback", debug
        except Exception as e:
            debug.append(f"Palpites: estratégia 3 falhou: {e}")
            return pd.DataFrame(), "erro", debug
    finally:
        session.close()

# =========================
# Contagem de acertos
# =========================
def contar_acertos_em_df(df_palpites, numeros_oficiais):
    """Adiciona coluna 'acertos' contando interseção com numeros_oficiais."""
    if df_palpites.empty:
        df_palpites["acertos"] = []
        return df_palpites

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
    oficial_set = set(nums)

    acertos_list = []
    for _, row in df_palpites.iterrows():
        try:
            pal = str(row.get("numeros", ""))
            pal_set = set(int(x.strip()) for x in pal.split(",") if x.strip())
            acertos_list.append(len(pal_set & oficial_set))
        except Exception:
            acertos_list.append(0)
    df_palpites = df_palpites.copy()
    df_palpites["acertos"] = acertos_list
    return df_palpites

# =========================
# UI
# =========================
st.set_page_config(page_title="Painel Evolutivo", layout="wide")
# --------------------------
# UI e filtros
# --------------------------
st.set_page_config(page_title="Painel Evolutivo", layout="wide")
st.markdown("<h1 class='title'>📊 Painel Evolutivo de Palpites</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Filtre por mês, visualize apenas apostas validadas e acompanhe a evolução dos acertos dos modelos LSXX.</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    ano = c1.selectbox("📅 Ano", list(range(2020, datetime.now().year + 1)), index=list(range(2020, datetime.now().year + 1)).index(datetime.now().year))
    mes = c2.selectbox("📅 Mês", list(range(1, 13)), index=datetime.now().month - 1)
    somente_validados = c3.checkbox("✅ Somente validados (apostados)")
    todos_do_mes = c4.checkbox("📅 Comparar todos os palpites com todos os resultados do mês")
if st.button("🔎 Analisar", use_container_width=True):
    # --- Buscar dados ---
    res_df, res_estrat, res_dbg = _query_resultados_mes(ano, mes)
    pal_df, pal_estrat, pal_dbg = _query_palpites_mes(ano, mes)

    # Normalizar datas
    res_df = _add_data_dia(res_df, "data")
    pal_df = _add_data_dia(pal_df, "data")

    # Filtro 'validados'
    pal_df_filtered, col_val = _filtrar_validados(pal_df, somente_validados)
    if somente_validados and col_val is None:
        st.info("⚠️ Nenhuma coluna de validação identificada. Exibindo todos.")
        pal_df_filtered = pal_df

    # Separar LSXX
    mask_ls = _mask_modelos(pal_df_filtered)
    pal_ls = pal_df_filtered[mask_ls].copy()

    # KPI e evolução
    kpi_contagem = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    evol_rows = []
    total_palpites_mes = len(pal_df_filtered)

    # --------------------------
    # Garantir coluna 'acertos'
    # --------------------------
    if "acertos" not in pal_df_filtered.columns:
        pal_df_filtered["acertos"] = 0

    # --------------------------
    # Lógica de acertos
    # --------------------------
    if todos_do_mes:
        # Acertos máximos de cada palpite contra todos os resultados do mês
        acertos_mes = []
        for _, pal in pal_df_filtered.iterrows():
            try:
                pal_set = set(int(x.strip()) for x in str(pal.get("numeros","")).split(",") if x.strip())
                max_acertos = 0
                for _, res in res_df.iterrows():
                    numeros_oficiais = [res.get(f"n{i}") for i in range(1,16)]
                    nums_set = set([int(x) for x in numeros_oficiais if x is not None])
                    max_acertos = max(max_acertos, len(pal_set & nums_set))
                acertos_mes.append(max_acertos)
            except Exception:
                acertos_mes.append(0)
        pal_df_filtered["acertos"] = acertos_mes

        # Atualizar KPIs
        for ac in kpi_contagem.keys():
            kpi_contagem[ac] = (pal_df_filtered["acertos"] == ac).sum()

        # Evolução LSXX
        for _, res in res_df.iterrows():
            numeros = [res.get(f"n{i}") for i in range(1,16)]
            dia = res["data_dia"].date() if pd.notna(res["data_dia"]) else None
            if dia is None: continue
            pal_dia_ls = pal_ls.copy()
            if not pal_dia_ls.empty:
                pal_dia_ls = contar_acertos_em_df(pal_dia_ls, numeros)
                for ac in kpi_contagem.keys():
                    evol_rows.append({
                        "data": res["data_dia"],
                        "acertos": ac,
                        "quantidade": int((pal_dia_ls["acertos"] == ac).sum())
                    })
    else:
        # Mantém lógica diária
        for _, res in res_df.iterrows():
            numeros = [res.get(f"n{i}") for i in range(1,16)]
            dia = res["data_dia"].date() if pd.notna(res["data_dia"]) else None
            if dia is None: continue

            pal_dia = pal_df_filtered[pal_df_filtered["data_dia"].dt.date == dia]
            if not pal_dia.empty:
                pal_dia = contar_acertos_em_df(pal_dia, numeros)
                for ac in kpi_contagem.keys():
                    kpi_contagem[ac] += (pal_dia["acertos"] == ac).sum()

            pal_dia_ls = pal_ls[pal_ls["data_dia"].dt.date == dia]
            if not pal_dia_ls.empty:
                pal_dia_ls = contar_acertos_em_df(pal_dia_ls, numeros)
                for ac in kpi_contagem.keys():
                    evol_rows.append({
                        "data": res["data_dia"],
                        "acertos": ac,
                        "quantidade": int((pal_dia_ls["acertos"] == ac).sum())
                    })

    # --------------------------
    # Cards de estatísticas
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### 📅 Mês {mes:02d}/{ano} • Total de palpites: **{total_palpites_mes}**", unsafe_allow_html=True)
    if somente_validados:
        st.markdown("**Filtro ativo:** apenas palpites validados ✅", unsafe_allow_html=True)
    cols = st.columns(5)
    labels = ["💥 15", "🔥 14", "✅ 13", "🔸 12", "🟡 11"]
    pts = [15,14,13,12,11]
    for c, lab, p in zip(cols, labels, pts):
        c.metric(f"{lab} acertos", kpi_contagem[p])
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # Gráfico evolutivo LSXX
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Evolução de acertos (Modelos LSXX)", unsafe_allow_html=True)
    evol_df = pd.DataFrame(evol_rows)
    if not evol_df.empty:
        pivot = evol_df.pivot_table(index="data", columns="acertos", values="quantidade", aggfunc="sum", fill_value=0).sort_index()
        long_df = pivot.reset_index().melt(id_vars="data", var_name="acertos", value_name="quantidade")
        long_df["tipo"] = long_df["acertos"].apply(lambda x: "13+" if x>=13 else "11-12")
        
        if PLOTLY_OK:
            fig = px.line(long_df, x="data", y="quantidade", color="acertos", line_dash="tipo", markers=True,
                        title="Evolução diária de acertos LSXX")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pivot, use_container_width=True)

    
    # --------------------------
    # Tabela detalhada + filtro dia da semana
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔎 Palpites detalhados", unsafe_allow_html=True)
    if not pal_df_filtered.empty:
        pal_df_filtered["dia_semana"] = pal_df_filtered["data_dia"].dt.day_name()
        dias = pal_df_filtered["dia_semana"].unique().tolist()
        dia_filter = st.multiselect("Filtrar por dia da semana", options=dias, default=dias)
        st.dataframe(pal_df_filtered[pal_df_filtered["dia_semana"].isin(dia_filter)].sort_values(by="acertos", ascending=False))
    else:
        st.info("Nenhum palpite encontrado.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # Exportar palpites 13+ acertos
    # --------------------------
    top13 = pal_df_filtered[pal_df_filtered["acertos"] >= 13]
    if not top13.empty:
        csv = top13.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exportar palpites 13+ acertos (CSV)",
            data=csv,
            file_name=f"palpites_13plus_{mes:02d}_{ano}.csv",
            mime="text/csv"
        )
