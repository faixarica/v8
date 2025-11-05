    # repete5.py v2.0 (19/05/2025 -14h00) - cria alimenta a tabela repete com todas ou apenas a ultima sorteios repetidos c 11,12,13,14 ou 15 numeros.
# mostrando quantas vezes deteminada sequencia ja foi sorteada.
# Autor: aff
#  27/07/25 - esse codigo esta atualizado
#  -- ele precisa dropar a tabela repete no database faixaricadb.db no botao "Iniciar Tabela", caso queira inserir varios resultados 
#  -- o botão Inserir e Analisar Ultimo Concuro, deve ser usado qdo for atualizar os dados diariamente, pq sera inserido uma um.
#  -- o botao Analisar Todos os Concursos, deve ser usado qdo for analisar todos os concursos, ou seja, ele ira comparar todos os concursos, 
# ------- nesse caso, o botao Iniciar Tabela repete, deve ser usado antes de analisar todos os concursos.
import sqlite3
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool
from functools import partial

# Configurações
LIMIAR_REPETICAO = 11
DB_PATH = 'faixaricadb.db'
CSV_PATH = 'loteria.csv'

def conectar():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def criar_tabelas():
    conn = conectar()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY,
            data_sorteio TEXT,
            num_1 INTEGER, num_2 INTEGER, num_3 INTEGER, num_4 INTEGER, num_5 INTEGER,
            num_6 INTEGER, num_7 INTEGER, num_8 INTEGER, num_9 INTEGER, num_10 INTEGER,
            num_11 INTEGER, num_12 INTEGER, num_13 INTEGER, num_14 INTEGER, num_15 INTEGER
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS repete (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concurso_atual INTEGER NOT NULL,
            concurso_anterior INTEGER NOT NULL,
            qtd_repetidos INTEGER NOT NULL,
            data_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS controle_execucao (
            id INTEGER PRIMARY KEY,
            ultima_execucao DATETIME
        )
    ''')
    c.execute("INSERT OR IGNORE INTO controle_execucao (id, ultima_execucao) VALUES (1, NULL)")

    conn.commit()
    conn.close()

def buscar_resultados():
    conn = conectar()
    c = conn.cursor()
    c.execute("SELECT id, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9, num_10, num_11, num_12, num_13, num_14, num_15 FROM resultados ORDER BY id")
    resultados = c.fetchall()
    conn.close()
    return [(row[0], set(row[1:])) for row in resultados]


def salvar_ultima_execucao(data):
    conn = conectar()
    c = conn.cursor()
    c.execute("UPDATE controle_execucao SET ultima_execucao = ? WHERE id = 1", (data,))
    conn.commit()
    conn.close()


def ja_existe_registro(concurso_atual, concurso_anterior):
    conn = conectar()
    c = conn.cursor()
    c.execute('''
        SELECT 1 FROM repete
        WHERE concurso_atual = ? AND concurso_anterior = ?
    ''', (concurso_atual, concurso_anterior))
    existe = c.fetchone() is not None
    conn.close()
    return existe

def gravar_repetidos_em_lote(registros):
    conn = conectar()
    c = conn.cursor()
    c.executemany('''
        INSERT INTO repete (concurso_atual, concurso_anterior, qtd_repetidos)
        VALUES (?, ?, ?)
    ''', registros)
    conn.commit()
    conn.close()


def comparar_par_wrapper(i, resultados):
    id_atual, set_atual = resultados[i]
    registros = []
    for j in range(i):
        id_anterior, set_anterior = resultados[j]
        coincidencias = len(set_atual & set_anterior)
        if coincidencias >= LIMIAR_REPETICAO:
            registros.append((id_atual, id_anterior, coincidencias))
    return registros

def processar_todos_em_paralelo(resultados, barra_progresso):
    total = len(resultados) - 1
    func = partial(comparar_par_wrapper, resultados=resultados)

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for idx, grupo in enumerate(pool.imap_unordered(func, range(1, len(resultados)))):
            if grupo:
                gravar_repetidos_em_lote(grupo)
            barra_progresso.progress((idx + 1) / total, text=f"🔍 Processando... {int((idx + 1) / total * 100)}%")

def inserir_ultimo_concurso_csv():
    if not os.path.exists(CSV_PATH):
        st.error("Arquivo loteria.csv NÃO encontrado!")
        return False

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        st.warning("Arquivo CSV está vazio.")
        return False

    df = df.sort_values(by='Concurso')
    ultimo_csv = df.iloc[-1]
    concurso = int(ultimo_csv['Concurso'])
    data = datetime.strptime(ultimo_csv['Data Sorteio'], '%d/%m/%Y').date()
    numeros = [int(n) for n in ultimo_csv[2:].tolist() if not pd.isna(n)]

    conn = conectar()
    c = conn.cursor()
    c.execute("SELECT MAX(id), MAX(data_sorteio) FROM resultados")
    row = c.fetchone()
    ultimo_concurso_db = row[0] if row[0] else 0
    ultima_data_db = datetime.fromisoformat(row[1]).date() if row[1] else None

    ontem = datetime.today().date() - timedelta(days=1)
    if ontem.weekday() == 6:  # domingo
        ontem -= timedelta(days=1)

    if concurso != ultimo_concurso_db + 1:
        st.warning(f"Número do concurso ({concurso}) NÃO é subsequente ao último ({ultimo_concurso_db}).")
        return False

    if data != ontem:
        st.warning(f"Data do Concurso ({data}) NÃO corresponde a Ontem ({ontem}).")
        return False

    c.execute('''
        INSERT INTO resultados (
            id, data_sorteio,
            num_1, num_2, num_3, num_4, num_5,
            num_6, num_7, num_8, num_9, num_10,
            num_11, num_12, num_13, num_14, num_15
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (concurso, data.strftime('%Y-%m-%d'), *numeros))

    conn.commit()
    conn.close()
    st.success(f"Concurso {concurso} de {data.strftime('%d/%m/%Y')} inserido com sucesso!")
    return True


def verificar_ultimo_concurso():
    resultados = buscar_resultados()
    if not resultados:
        st.warning("⚠️ Nenhum Concurso Encontrado.")
        return

    ultimo_id, ultimo_set = resultados[-1]
    st.info(f"🔍 Verificando o último Concurso: #{ultimo_id}")

    registros = []
    for id_concurso, numeros in resultados[:-1]:
        if ja_existe_registro(ultimo_id, id_concurso):
            continue

        coincidencias = len(ultimo_set & numeros)
        if coincidencias >= LIMIAR_REPETICAO:
            registros.append((ultimo_id, id_concurso, coincidencias))

    if registros:
        gravar_repetidos_em_lote(registros)

    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    salvar_ultima_execucao(agora)
    st.success(f"✅ Último Concurso #{ultimo_id} Verificado com Sucesso!")

def verificar_todos():
    resultados = buscar_resultados()
    if not resultados:
        st.warning("⚠️ Nenhum Concurso Encontrado.")
        return

    barra = st.progress(0, text="Iniciando Análise...")
    processar_todos_em_paralelo(resultados, barra)

    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    salvar_ultima_execucao(agora)
    st.success("✅ Todos os Concursos foram verificados com SUCESSO!")

def resetar_tabela_repete():
    conn = conectar()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS repete")
    conn.commit()
    conn.close()
    st.success("🗑️ Tabela 'Repete' foi Apagada com SUCESSO!")

def main():
    st.set_page_config(page_title="Análise de Repetições", layout="wide")
    st.title("📊 Análise de Repetições de Números")
    st.markdown("Ferramenta para análise inteligente de padrões de repetições na Lotofácil.")

    criar_tabelas()

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("📤 Inserir e Analisar Último Concurso"):
            if inserir_ultimo_concurso_csv():
                verificar_ultimo_concurso()

        if st.button("🔁 Analisar Todos os Concursos"):
            verificar_todos()

    with col2:
        if st.button("🛑 Iniciar Tabela repete"):
            resetar_tabela_repete()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
