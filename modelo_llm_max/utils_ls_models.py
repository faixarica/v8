# utils_ls_models.py
import sys
import os
import logging
import numpy as np
from sqlalchemy import text

# --- CARREGA .env DA PASTA V8 ---
V8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(V8_DIR, '.env'))

# --- IMPORTA db.py DE V8 SEM ALTERAR sys.path ---
# Usamos importação relativa ou caminho explícito
db_path = os.path.join(V8_DIR, 'db.py')
if os.path.exists(db_path):
    # Carrega db.py manualmente (sem alterar sys.path global)
    import importlib.util
    spec = importlib.util.spec_from_file_location("db", db_path)
    db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_module)
    Session = db_module.Session
else:
    raise FileNotFoundError(f"db.py não encontrado em {V8_DIR}")

logging.basicConfig(level=logging.INFO)
# ...
def to_binary(jogo):
    b = np.zeros(25, dtype=np.uint8)
    for n in jogo:
        try:
            idx = int(n) - 1
            if 0 <= idx < 25:
                b[idx] = 1
            else:
                logging.warning(f"[to_binary] Valor fora do intervalo 1-25: {n}")
        except Exception as e:
            logging.error(f"[to_binary] Erro ao processar {n}: {e}")
    return b

def fetch_history(last_n=None, include_repeats=False):
    db = Session()
    try:
        if last_n:
            q = text("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso DESC
                LIMIT :lim
            """)
            rows = db.execute(q, {"lim": last_n}).fetchall()
            rows = list(reversed(rows))
        else:
            q = text("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso ASC
            """)
            rows = db.execute(q).fetchall()

        rows = [[int(v) for v in r] for r in rows]

        if include_repeats:
            try:
                rep_rows = db.execute(text("SELECT concurso_atual, qtd_repetidos FROM repete")).fetchall()
                rep_map = {int(r[0]): int(r[1]) for r in rep_rows}
            except Exception:
                rep_map = {}
            return rows, rep_map

        return rows
    finally:
        db.close()