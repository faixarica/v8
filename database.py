import sqlite3
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def get_db():
    try:
        conn = sqlite3.connect('database.db')
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {str(e)}")
        return None 

def carregar_planos():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, valor FROM planos")
    planos = cursor.fetchall()
    conn.close()
    return {id: {"nome": nome, "valor": valor} for id, nome, valor in planos}

def tabela_existe(cursor, nome_tabela):
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (nome_tabela,))
    return cursor.fetchone() is not None

def inicializar_banco_dados():
    conn = get_db()
    if conn is None:
        return
    
    cursor = conn.cursor()
    
    if not tabela_existe(cursor, "usuarios"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome_completo TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                usuario TEXT NOT NULL UNIQUE,
                senha TEXT NOT NULL,
                tipo TEXT DEFAULT 'U',
                id_plano INTEGER DEFAULT 1,
                ativo INTEGER DEFAULT 1,
                telefone TEXT,
                data_nascimento TEXT,
                dt_cadastro TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (id_plano) REFERENCES planos(id) ON DELETE SET DEFAULT
            );
        """)
        print("Tabela 'usuarios' criada.")
    
    if not tabela_existe(cursor, "planos"):
        cursor.execute("""
            CREATE TABLE planos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                palpites_dia INTEGER NOT NULL,
                valor REAL NOT NULL,
                validade_dias INTEGER NOT NULL,
                bonus TEXT,
                status TEXT DEFAULT 'A'
            )
        """)
        print("Tabela 'planos' criada.")
    
    if not tabela_existe(cursor, "client_plans"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_client INTEGER NOT NULL,
                id_plano INTEGER NOT NULL,
                data_inclusao TEXT NOT NULL DEFAULT (datetime('now')),
                data_expira_plan TEXT NOT NULL DEFAULT (datetime('now', '+30 days')),
                palpites_dia_usado INTEGER DEFAULT 0,
                ativo INTEGER,
                FOREIGN KEY (id_client) REFERENCES usuarios(id) ON DELETE CASCADE,
                FOREIGN KEY (id_plano) REFERENCES planos(id) ON DELETE SET NULL
            );
        """)
        print("Tabela 'client_plans' criada.")
      
    if not tabela_existe(cursor, "resultados_oficiais"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resultados_oficiais (
                concurso INTEGER PRIMARY KEY,
                data TEXT NOT NULL,
                n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER,
                n6 INTEGER, n7 INTEGER, n8 INTEGER, n9 INTEGER, n10 INTEGER,
                n11 INTEGER, n12 INTEGER, n13 INTEGER, n14 INTEGER, n15 INTEGER,
                ganhadores_15 INTEGER, ganhadores_14 INTEGER, ganhadores_13 INTEGER,
                ganhadores_12 INTEGER, ganhadores_11 INTEGER
            );
        """)
        print("Tabela 'resultados_oficiais' criada.")
    
    if not tabela_existe(cursor, "palpites"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS palpites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                numeros TEXT NOT NULL,
                modelo TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT (datetime('now')),
                status TEXT,
                premiado TEXT DEFAULT 'N',
                concurso_premio INTEGER,
                FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
            );
        """)
        print("Tabela 'palpites' criada.")
        conn.commit()
    
    if not tabela_existe(cursor, "financeiro"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financeiro (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_cliente INTEGER,
                id_plano INTEGER,
                data_pagamento TEXT,
                forma_pagamento TEXT,
                valor REAL,
                data_validade TEXT,
                estono TEXT DEFAULT 'N',
                data_estorno TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (id_cliente) REFERENCES usuarios(id) ON DELETE CASCADE,
                FOREIGN KEY (id_plano) REFERENCES planos(id) ON DELETE SET NULL
            );
        """)
        print("Tabela 'financeiro' criada.")
        conn.commit()

    if not tabela_existe(cursor, "log_user"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS log_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_cliente INTEGER,
                data_hora TEXT,
                ip TEXT,
                hostname TEXT,
                city TEXT,
                region TEXT,
                country TEXT,
                loc TEXT,
                org TEXT,
                postal TEXT,
                timezone TEXT,
                FOREIGN KEY (id_cliente) REFERENCES usuarios(id)
            );
        """)
        print("Tabela 'log_user' criada.")
        conn.commit()
    
    if not tabela_existe(cursor, "loterias"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loterias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome_loteria TEXT,
                num_premio INTEGER,
                aposta INTEGER
            );
        """)
        print("Tabela 'loterias' criada.")
        conn.commit()

    conn.close()

# Chamada para inicializar o banco de dados sempre que o módulo for importado
inicializar_banco_dados()

# Executar diretamente
if __name__ == "__main__":
    inicializar_banco_dados()
