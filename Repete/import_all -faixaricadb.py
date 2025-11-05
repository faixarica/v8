
## importa_all.py  v2.0 (16/05/2025) - esse codigo importa todos ou o ultimo registra no arquivo loterias.csv p/  a tabela resultados
import pandas as pd
import sqlite3
from datetime import datetime
import os

# Caminhos configuráveis
CSV_PATH = 'loteria.csv'  # Arquivo CSV com os resultados anteriores
DB_PATH = 'faixaricadb.db'  # Banco de dados principal


def conectar():
    """Conecta ao banco de dados SQLite."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Cria todas as tabelas necessárias se ainda não existirem."""
    conn = conectar()
    c = conn.cursor()

    # Tabela usuarios
    c.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            sobrenome TEXT,
            data_nascimento DATE,
            telefone TEXT,
            email TEXT UNIQUE,
            cidade TEXT,
            uf TEXT(2),
            usuario TEXT UNIQUE,
            senha TEXT,
            plano TEXT DEFAULT 'Free',
            data_atualizacao_plano DATETIME
        )
    ''')

    # Tabela palpites
    c.execute('''
        CREATE TABLE IF NOT EXISTS palpites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            metodo TEXT,
            data_geracao DATETIME,
            acertos INTEGER DEFAULT 0,
            valido TEXT DEFAULT 'N',
            num_1 INTEGER, num_2 INTEGER, num_3 INTEGER, num_4 INTEGER, num_5 INTEGER,
            num_6 INTEGER, num_7 INTEGER, num_8 INTEGER, num_9 INTEGER, num_10 INTEGER,
            num_11 INTEGER, num_12 INTEGER, num_13 INTEGER, num_14 INTEGER, num_15 INTEGER,
            concurso_maior_coincidencia INTEGER,
            maior_coincidencia INTEGER,
            FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
        )
    ''')

    # Tabela logins
    c.execute('''
        CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            data_login DATETIME,
            FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
        )
    ''')

    # Tabela resultados
    c.execute('''
       CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- ou INTEGER PRIMARY KEY se quiser controlar
            concurso INTEGER NOT NULL,              -- ← Adicionada esta linha
             Data_Sorteio DATE,
            num_1 INTEGER, num_2 INTEGER, num_3 INTEGER, num_4 INTEGER, num_5 INTEGER,
            num_6 INTEGER, num_7 INTEGER, num_8 INTEGER, num_9 INTEGER, num_10 INTEGER,
            num_11 INTEGER, num_12 INTEGER, num_13 INTEGER, num_14 INTEGER, num_15 INTEGER,
            gan_11 INTEGER, gan_12 INTEGER, gan_13 INTEGER, gan_14 INTEGER, gan_15 INTEGER
        )
    ''')

    # Tabela repete
    c.execute('''
        CREATE TABLE IF NOT EXISTS repete (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concurso_atual INTEGER,
            concurso_anterior INTEGER,
            qtd_repetidos INTEGER,
            data_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabela controle_execucao
    c.execute('''
        CREATE TABLE IF NOT EXISTS controle_execucao (
            id INTEGER PRIMARY KEY,
            ultima_execucao DATETIME
        )
    ''')
    c.execute("INSERT OR IGNORE INTO controle_execucao (id, ultima_execucao) VALUES (1, NULL)")

    conn.commit()
    conn.close()
    print("✅ Banco de dados inicializado com sucesso.")


def importar_csv_para_dataframe(caminho_csv):
    """Importa o arquivo CSV para um DataFrame pandas."""
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho_csv}")

    try:
        df = pd.read_csv(caminho_csv, sep=',', encoding='utf-8')
        df.columns = df.columns.str.strip()

        # Renomear colunas para nomes padronizados
        df.rename(columns={
            'Concurso': 'concurso',''
            'Data Sorteio': 'Data_Sorteio',
            'Bola1': 'num_1',
            'Bola2': 'num_2',
            'Bola3': 'num_3',
            'Bola4': 'num_4',
            'Bola5': 'num_5',
            'Bola6': 'num_6',
            'Bola7': 'num_7',
            'Bola8': 'num_8',
            'Bola9': 'num_9',
            'Bola10': 'num_10',
            'Bola11': 'num_11',
            'Bola12': 'num_12',
            'Bola13': 'num_13',
            'Bola14': 'num_14',
            'Bola15': 'num_15',
            'Ganhadores 15 acertos': 'gan_15',
            'Ganhadores 14 acertos': 'gan_14',
            'Ganhadores 13 acertos': 'gan_13',
            'Ganhadores 12 acertos': 'gan_12',
            'Ganhadores 11 acertos': 'gan_11',
            'Concurso': 'concurso'  # ← Adicione esta linha
        }, inplace=True)

        # Verificar se todas as colunas esperadas estão presentes
        colunas_esperadas = ['concurso','Data_Sorteio', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5',
                             'num_6', 'num_7', 'num_8', 'num_9', 'num_10', 'num_11',
                             'num_12', 'num_13', 'num_14', 'num_15', 'gan_11', 'gan_12',
                             'gan_13', 'gan_14', 'gan_15']
        if not all(col in df.columns for col in colunas_esperadas):
            faltando = set(colunas_esperadas) - set(df.columns)
            raise ValueError(f"Colunas faltando no CSV: {faltando}")

        # Converter Data_Sorteio para datetime
        df['Data_Sorteio'] = pd.to_datetime(df['Data_Sorteio'], dayfirst=True, errors='coerce')
        if df['Data_Sorteio'].isnull().any():
            raise ValueError("❌ Há datas inválidas no CSV.")

        return df

    except Exception as e:
        raise RuntimeError(f"Erro ao processar o CSV: {e}")


def obter_datas_existentes(conn):
    """Obtém todas as datas de sorteios já existentes no banco de dados."""
    try:
        query = "SELECT Data_Sorteio FROM resultados"
        df_banco = pd.read_sql_query(query, conn)
        df_banco['Data_Sorteio'] = pd.to_datetime(df_banco['Data_Sorteio'])
        return set(df_banco['Data_Sorteio'])
    except Exception as e:
        raise RuntimeError(f"Erro ao consultar datas no banco de dados: {e}")


def filtrar_resultados_antigos(df):
    """Filtra apenas os resultados anteriores à data atual."""
    hoje = pd.to_datetime(datetime.today().date())
    df_anteriores = df[df['Data_Sorteio'] < hoje]
    if df_anteriores.empty:
        print("ℹ️ Nenhum resultado anterior à data de hoje encontrado no CSV.")
    return df_anteriores


def inserir_novos(conn, df_novos):
    """Insere novos resultados no banco de dados."""
    try:
        df_novos.to_sql('resultados', conn, if_exists='append', index=False)
        print(f"✅ {len(df_novos)} novos concursos importados com sucesso.")
    except Exception as e:
        raise RuntimeError(f"Erro ao inserir novos resultados no banco de dados: {e}")


def incluir_ultimo_sorteio(conn):
    """Inclui manualmente o último sorteio no banco."""
    print("\n🎲 Incluir Último Sorteio")
    try:
        while True:
            data_str = input("Digite a data do sorteio (DD/MM/YYYY): ").strip()
            if data_str.lower() == 'cancelar':
                print("❌ Operação cancelada.")
                return
            try:
                data = datetime.strptime(data_str, "%d/%m/%Y").date()
                break
            except ValueError:
                print("❌ Formato inválido. Use DD/MM/YYYY ou digite 'cancelar'.")

        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM resultados WHERE Data_Sorteio = ?", (data,))
        if c.fetchone()[0] > 0:
            print("❌ Já existe um sorteio com essa data.")
            return

        numeros = []
        print("Digite os 15 números sorteados (entre 1 e 25):")
        for i in range(1, 16):
            while True:
                n = input(f"Número {i}: ")
                if n.lower() == 'cancelar':
                    print("❌ Operação cancelada.")
                    return
                try:
                    numero = int(n)
                    if 1 <= numero <= 25 and numero not in numeros:
                        numeros.append(numero)
                        break
                    else:
                        print("⚠️ Número inválido ou repetido. Deve estar entre 1 e 25.")
                except ValueError:
                    print("⚠️ Digite um número inteiro válido ou 'cancelar'.")

        ganhadores = {}
        for acertos in range(11, 16):
            while True:
                g = input(f"Ganhadores com {acertos} acertos: ")
                if g.lower() == 'cancelar':
                    print("❌ Operação cancelada.")
                    return
                try:
                    ganhadores[f'gan_{acertos}'] = int(g)
                    if ganhadores[f'gan_{acertos}'] < 0:
                        print("⚠️ Ganhadores devem ser zero ou positivos.")
                    else:
                        break
                except ValueError:
                    print("⚠️ Digite um número inteiro válido ou 'cancelar'.")

        c.execute('''
            INSERT INTO resultados (
                concurso,Data_Sorteio, num_1, num_2, num_3, num_4, num_5, 
                num_6, num_7, num_8, num_9, num_10, num_11, 
                num_12, num_13, num_14, num_15, gan_11, gan_12, 
                gan_13, gan_14, gan_15
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.strftime("%Y-%m-%d"),
            *numeros,
            ganhadores['gan_11'],
            ganhadores['gan_12'],
            ganhadores['gan_13'],
            ganhadores['gan_14'],
            ganhadores['gan_15']
        ))
        conn.commit()
        print("✅ Sorteio incluído com sucesso!")

    except Exception as e:
        print(f"❌ Erro ao incluir sorteio: {e}")
        conn.rollback()


def main():
    """Função principal para importar resultados do CSV ou incluir manualmente."""
    print("\n🔁 Bem-vindo ao Assistente de Importação\n")

    # Garante que o banco está pronto
    init_db()

    try:
        conn = conectar()
        print("Escolha uma opção:")
        print("1. Importar resultados do CSV")
        print("2. Incluir manualmente o último sorteio")
        opcao = input("Opção: ").strip().lower()

        if opcao == '1':
            print("🔍 Importando dados do CSV...")
            df = importar_csv_para_dataframe(CSV_PATH)
            df_anteriores = filtrar_resultados_antigos(df)

            if df_anteriores.empty:
                print("ℹ️ Nenhum resultado antigo para importar.")
                return

            datas_existentes = obter_datas_existentes(conn)
            df_faltantes = df_anteriores[~df_anteriores['Data_Sorteio'].isin(datas_existentes)]

            if df_faltantes.empty:
                print("✅ Todos os concursos já estão cadastrados.")
                return

            print(f"\n📊 Foram encontrados {len(df_faltantes)} concursos anteriores não cadastrados.")
            resposta = input("Deseja importar? [s/n]: ").strip().lower()

            if resposta == 's':
                inserir_novos(conn, df_faltantes)
            else:
                print("❌ Importação cancelada.")

        elif opcao == '2':
            incluir_ultimo_sorteio(conn)
        else:
            print("❌ Opção inválida.")

        conn.close()

    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")


if __name__ == "__main__":
    main()