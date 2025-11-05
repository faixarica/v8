# verificar_palpites_avancado.py (versão revisada)
import streamlit as st
import pandas as pd
from datetime import datetime, date
from db import Session
from sqlalchemy import text

st.markdown("""
<head>
  <!-- SEO AVANÇADO -->
  <meta name="robots" content="index, follow">
  <meta name="keywords" content="Palpites lotofácil, Palpites lotomania, jogar na loteria, estatísticas loteria, apostas inteligentes, IA para loteria, fAIxaBet"/>
  <meta name="author" content="fAIxaBet"/>
  <link rel="canonical" href="https://faixabet.com.br"/>
  <!-- Favicon -->
  <link rel="icon" href="https://faixabet.com.br/favicon.ico" type="image/x-icon"/>
  <title>fAIxaBet</title>
</head>
""", unsafe_allow_html=True)

st.markdown("""
<div class="container mx-auto px-4">
  <nav class="flex justify-between items-center py-6">
    <!-- Logo -->
    <a href="/" class="text-3xl font-bold flex items-center" style="font-family: 'Poppins', sans-serif;">
      <span class="text-blue-300">f</span>
      <span class="text-green-300">A</span>
      <span class="text-yellow-300">I</span>
      <span class="text-blue-300">x</span>
      <span class="text-green-600">a</span>
      <span class="font-light">Bet</span>
    </a>
  </nav>
</div>
""", unsafe_allow_html=True)


# ---------- CONFIG ----------
# Valores de prêmio por acerto (exemplos). Ajuste conforme sua regra/premiação real.
PRIZES = {
    15: 3.50,  # prêmio para 15 acertos (exemplo)
    14: 3.50,      # prêmio unitário para 14 acertos (exemplo)
    13: 3.50,       # ...
    12: 3.50,
    11: 3.50,
}
HIT_GROUPS = [15, 14, 13, 12, 11]
# ---------------------------

# Detectar plotly
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# =========================
# Helpers de parsing/normalização
# =========================
def _parse_date_any(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        try:
            return pd.to_datetime(v).date()
        except Exception:
            return None
    s = str(v).strip()
    # tentativas comuns
    fmts = ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y")
    for fmt in fmts:
        try:
            # usar slice compatível com o formato (protege quando há time)
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception:
            pass
    try:
        x = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return None if pd.isna(x) else x.date()
    except Exception:
        return None

def _add_data_dia(df, col="data"):
    df = df.copy()
    if df.empty or col not in df.columns:
        df["data_dia"] = pd.NaT
        return df
    df["data_dia"] = df[col].apply(_parse_date_any)
    df["data_dia"] = pd.to_datetime(df["data_dia"], errors="coerce")
    return df

def _filtrar_validados(df, somente_validados: bool):
    if not somente_validados or df.empty:
        return df.copy(), None

    df = df.copy()
    candidatos = ["validado", "aposta_confirmada", "confirmado", "foi_apostado", "aposta", "status", "status_aposta", "pago"]
    col_usada = None

    for c in candidatos:
        if c in df.columns:
            col_usada = c
            vals = df[c].astype(str).str.strip().str.upper()
            truthy = {"1", "TRUE", "T", "S", "SIM", "Y", "YES", "OK", "CONFIRMADO", "CONFIRMADA", "PAGO", "EFETUADA", "CONCLUIDA", "CONCLUÍDA"}
            df = df[vals.isin(truthy)].copy()
            break

    return df, col_usada

def _mask_modelos(df):
    if df is None:
        return pd.Series([], dtype=bool)
    df = df.copy()
    cols = [c for c in ["modelo", "modelo_nome", "algoritmo", "metodo", "gerador", "origem"] if c in df.columns]
    if not cols:
        return pd.Series([False] * len(df), index=df.index)
    mask = pd.Series([False] * len(df), index=df.index)
    for c in cols:
        colu = df[c].fillna("").astype(str).str.upper().str.strip()
        m = colu.str.startswith("LS") | (colu == "LSTM")
        mask = mask | m
    return mask

# =========================
# Consultas flexíveis (mês)
# =========================
def _query_resultados_mes(ano: int, mes: int):
    debug = []
    session = Session()
    try:
        # 1) tentativa EXTRACT nativo
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

        # 2) tentativa TO_DATE (texto)
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

        # 3) fallback
        try:
            q = text("SELECT concurso, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, data FROM resultados_oficiais")
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
    debug = []
    session = Session()
    try:
        try:
            q = text("SELECT * FROM palpites WHERE EXTRACT(YEAR FROM data) = :ano AND EXTRACT(MONTH FROM data) = :mes")
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Palpites: estratégia 1 (EXTRACT nativo).")
                return df, "nativo", debug
            debug.append("Palpites: estratégia 1 vazia.")
        except Exception as e:
            debug.append(f"Palpites: estratégia 1 falhou: {e}")

        try:
            q = text("SELECT * FROM palpites WHERE EXTRACT(YEAR FROM TO_DATE(data, 'DD/MM/YYYY')) = :ano AND EXTRACT(MONTH FROM TO_DATE(data, 'DD/MM/YYYY')) = :mes")
            df = pd.read_sql_query(q, con=session.bind, params={"ano": ano, "mes": mes})
            if not df.empty:
                debug.append("Palpites: estratégia 2 (TO_DATE texto).")
                return df, "texto", debug
            debug.append("Palpites: estratégia 2 vazia.")
        except Exception as e:
            debug.append(f"Palpites: estratégia 2 falhou: {e}")

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
def _parse_numeros_field(raw):
    """
    Recebe campo 'numeros' que pode ser string '1,2,3...' ou lista; retorna set[int].
    """
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        out = set()
        for x in raw:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    s = str(raw)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = set()
    for p in parts:
        try:
            out.add(int(p))
        except Exception:
            # tentar extrair dígitos
            digits = ''.join(ch for ch in p if ch.isdigit())
            if digits:
                try:
                    out.add(int(digits))
                except Exception:
                    pass
    return out

def contar_acertos_em_df(df_palpites, numeros_oficiais):
    df = df_palpites.copy()
    if df.empty:
        df["acertos"] = pd.Series(dtype=int)
        return df
    # prepara set oficiais
    nums = []
    for n in numeros_oficiais:
        try:
            if n is None: continue
            ni = int(n)
            if ni > 0:
                nums.append(ni)
        except Exception:
            continue
    oficial_set = set(nums)
    acertos = []
    for _, row in df.iterrows():
        try:
            pal_set = _parse_numeros_field(row.get("numeros", ""))
            acertos.append(len(pal_set & oficial_set))
        except Exception:
            acertos.append(0)
    df["acertos"] = acertos
    return df

# =========================
# UI
# =========================
st.set_page_config(page_title="Analise Evolutiva", layout="wide")
st.markdown("<h1 class='title'>📊 Analise Evolutiva dos Palpites/Bets</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Filtre por mês, visualize apenas apostas validadas e compare modelos LSTMs vs outros geradores.</div>", unsafe_allow_html=True)

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

    res_df = _add_data_dia(res_df, "data")
    pal_df = _add_data_dia(pal_df, "data")

    pal_df_filtered, col_val = _filtrar_validados(pal_df, somente_validados)
    if somente_validados and col_val is None:
        st.info("⚠️ Nenhuma coluna de validação identificada. Exibindo todos.")
        pal_df_filtered = pal_df.copy()

    # garantir colunas
    if "acertos" not in pal_df_filtered.columns:
        pal_df_filtered["acertos"] = 0

    # separar modelos
    mask_ls = _mask_modelos(pal_df_filtered)
    pal_ls = pal_df_filtered[mask_ls].copy()
    pal_outros = pal_df_filtered[~mask_ls].copy()

    total_palpites_mes = len(pal_df_filtered)

    # acumuladores
    kpi_contagem = {k: 0 for k in HIT_GROUPS}
    evol_rows = []

    # função auxiliar para processar um palpite vs todos os resultados do mês
    def _max_acertos_contra_resultados(pal_set, resultados):
        max_ac = 0
        for _, res in resultados.iterrows():
            numeros_oficiais = [res.get(f"n{i}") for i in range(1,16)]
            nums_set = set(int(x) for x in numeros_oficiais if x is not None)
            max_ac = max(max_ac, len(pal_set & nums_set))
        return max_ac

    if todos_do_mes:
        # para cada palpite, calcula o máximo de acertos contra qualquer resultado do mês
        acertos_list = []
        for _, pal in pal_df_filtered.iterrows():
            try:
                pal_set = _parse_numeros_field(pal.get("numeros",""))
                ac_max = _max_acertos_contra_resultados(pal_set, res_df) if not res_df.empty else 0
                acertos_list.append(ac_max)
            except Exception:
                acertos_list.append(0)
        pal_df_filtered["acertos"] = acertos_list

        # kpi total
        for ac in HIT_GROUPS:
            kpi_contagem[ac] = int((pal_df_filtered["acertos"] == ac).sum())

        # evolução para LSXX: para cada resultado do mês conta quantos palpites LSXX atingiram cada ac nessa data (max contra aquele resultado)
        for _, res in res_df.iterrows():
            numeros = [res.get(f"n{i}") for i in range(1,16)]
            dia_ts = res["data_dia"]
            if pd.isna(dia_ts):
                continue
            # contar LSXX vs esse resultado
            if not pal_ls.empty:
                pal_tmp = contar_acertos_em_df(pal_ls.copy(), numeros)
                for ac in HIT_GROUPS:
                    evol_rows.append({
                        "data": dia_ts,
                        "acertos": ac,
                        "quantidade": int((pal_tmp["acertos"] == ac).sum())
                    })
    else:
        # lógica diária: cada resultado do dia avalia apenas palpites daquele dia
        for _, res in res_df.iterrows():
            numeros = [res.get(f"n{i}") for i in range(1,16)]
            dia = res["data_dia"].date() if pd.notna(res["data_dia"]) else None
            if dia is None:
                continue
            pal_dia = pal_df_filtered[pal_df_filtered["data_dia"].dt.date == dia]
            if not pal_dia.empty:
                pal_dia = contar_acertos_em_df(pal_dia, numeros)
                for ac in HIT_GROUPS:
                    kpi_contagem[ac] += int((pal_dia["acertos"] == ac).sum())

            pal_dia_ls = pal_ls[pal_ls["data_dia"].dt.date == dia]
            if not pal_dia_ls.empty:
                pal_dia_ls = contar_acertos_em_df(pal_dia_ls, numeros)
                for ac in HIT_GROUPS:
                    evol_rows.append({
                        "data": res["data_dia"],
                        "acertos": ac,
                        "quantidade": int((pal_dia_ls["acertos"] == ac).sum())
                    })

    # --------------------------
    # Cards de estatísticas (quantidade + cálculo prêmio)
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### 📅 Mês {mes:02d}/{ano} • Total de palpites: **{total_palpites_mes}**", unsafe_allow_html=True)
    if somente_validados:
        st.markdown("**Filtro ativo:** apenas palpites validados ✅", unsafe_allow_html=True)

    cols = st.columns(5)
    for c, ac in zip(cols, HIT_GROUPS):
        qtd = kpi_contagem.get(ac, 0)
        premio_unit = PRIZES.get(ac, 0.0)
        total_pago = qtd * premio_unit
        c.metric(label=f"{ac} acertos", value=qtd, delta=f"R$ {total_pago:,.2f}")
        c.caption = None  # placeholder; Streamlit não usa .caption em metric, deixei para clareza

    # soma total paga e percentual de palpites que ganharam (>=11)
    total_pago_geral = sum(kpi_contagem.get(ac, 0) * PRIZES.get(ac, 0.0) for ac in HIT_GROUPS)
    palpites_com_premio = sum(kpi_contagem.get(ac, 0) for ac in HIT_GROUPS)
    taxa_ganhadores = (palpites_com_premio / total_palpites_mes * 100) if total_palpites_mes > 0 else 0.0

    st.markdown(f"**Total investido estimado (somando todas as faixas):** R$ {total_pago_geral:,.2f}", unsafe_allow_html=True)
    st.markdown(f"**Palpites que pagaram prêmio (>=11):** {palpites_com_premio} ({taxa_ganhadores:.2f}% do total de palpites)", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # Comparação LSTM (LSXX) vs Outros
    # --------------------------
    def resumo_por_grupo(df_group, nome):
        resumo = {}
        resumo["nome"] = nome
        resumo["total_palpites"] = len(df_group)
        if df_group.empty:
            resumo.update({"media_acertos": 0.0, "percent_11plus": 0.0, "qtd_11plus": 0})
            return resumo
        media = float(df_group["acertos"].mean()) if "acertos" in df_group.columns else 0.0
        qtd_11plus = int((df_group["acertos"] >= 11).sum())
        pct_11plus = (qtd_11plus / len(df_group) * 100) if len(df_group) > 0 else 0.0
        resumo.update({"media_acertos": media, "percent_11plus": pct_11plus, "qtd_11plus": qtd_11plus})
        return resumo

    # garantir coluna 'acertos' para os grupos antes do resumo
    if "acertos" not in pal_ls.columns:
        pal_ls["acertos"] = 0
    if "acertos" not in pal_outros.columns:
        pal_outros["acertos"] = 0

    resumo_ls = resumo_por_grupo(pal_ls, "LSTM / LSXX")
    resumo_outros = resumo_por_grupo(pal_outros, "Outros")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Comparativo: LSTM (LSXX) vs Modelos Convencionais", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Palpites LSTM", resumo_ls["total_palpites"], f"11+ : {resumo_ls['qtd_11plus']} ({resumo_ls['percent_11plus']:.2f}%)")
    a2.metric("Palpites Outros", resumo_outros["total_palpites"], f"11+ : {resumo_outros['qtd_11plus']} ({resumo_outros['percent_11plus']:.2f}%)")
    a3.metric("Média acertos (LSTM vs Outros)", f"{resumo_ls['media_acertos']:.2f} vs {resumo_outros['media_acertos']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # gráfico comparativo simples
    comp_df = pd.DataFrame([
        {"grupo": "LSTM/LSXX", "palpites": resumo_ls["total_palpites"], "11plus": resumo_ls["qtd_11plus"], "media_acertos": resumo_ls["media_acertos"]},
        {"grupo": "Outros", "palpites": resumo_outros["total_palpites"], "11plus": resumo_outros["qtd_11plus"], "media_acertos": resumo_outros["media_acertos"]},
    ])
    if PLOTLY_OK:
        fig = px.bar(comp_df.melt(id_vars="grupo", value_vars=["palpites","11plus"]), x="grupo", y="value", color="variable", barmode="group", title="Comparativo LSTM vs Outros (palpites e 11+)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(comp_df)

    # --------------------------
    # Gráfico evolutivo LSXX
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Evolução de acertos (Modelos LSXX)", unsafe_allow_html=True)
    evol_df = pd.DataFrame(evol_rows)
    if not evol_df.empty:
        pivot = evol_df.pivot_table(index="data", columns="acertos", values="quantidade", aggfunc="sum", fill_value=0).sort_index()
        if PLOTLY_OK:
            long_df = pivot.reset_index().melt(id_vars="data", var_name="acertos", value_name="quantidade")
            fig = px.line(long_df, x="data", y="quantidade", color="acertos", markers=True, title="Evolução diária de acertos LSXX")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pivot, use_container_width=True)
    else:
        st.info("Sem dados evolutivos para LSXX esse mês.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # Tabela detalhada + filtro dia da semana
    # --------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔎 Palpites detalhados", unsafe_allow_html=True)
    if not pal_df_filtered.empty:
        pal_df_filtered["dia_semana"] = pal_df_filtered["data_dia"].dt.day_name()
        dias = pal_df_filtered["dia_semana"].dropna().unique().tolist()
        dia_filter = st.multiselect("Filtrar por dia da semana", options=dias, default=dias)
        st.dataframe(pal_df_filtered[pal_df_filtered["dia_semana"].isin(dia_filter)].sort_values(by="acertos", ascending=False))
    else:
        st.info("Nenhum palpite encontrado.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # Exportar palpites 13+ acertos
    # --------------------------
    if "acertos" in pal_df_filtered.columns:
        top13 = pal_df_filtered[pal_df_filtered["acertos"] >= 13]
        if not top13.empty:
            csv = top13.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Exportar palpites 13+ acertos (CSV)",
                data=csv,
                file_name=f"palpites_13plus_{mes:02d}_{ano}.csv",
                mime="text/csv"
            )
