# prepare_real_data_db_v3.py — v3.0 (31/10/2025)
# Compatível com Python 3.11+, SQLAlchemy (db.Session)
# Gera artefatos tipados + legado, com validações e features básicas.

import os, json
import numpy as np
from sqlalchemy import text
from datetime import date
from db import Session

OUT_DIR = "./dados"

DTYPE = np.dtype([
    ("concurso",  "i4"),
    ("data",      "M8[D]"),
    ("numeros",   "i1", (15,)),
    ("pares",     "i1"),
    ("soma",      "i2"),
    ("baixas",    "i1"),   # 1–13
    ("altas",     "i1"),   # 14–25
    ("g11",       "i4"),
    ("g12",       "i4"),
    ("g13",       "i4"),
    ("g14",       "i4"),
    ("g15",       "i4"),
])

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _fetch_rows():
    """
    Busca colunas principais + winners se existirem.
    data_norm preferida; cai para data TEXT se necessário.
    """
    db = Session()
    try:
        # Tenta trazer tudo (se não existir, virá NULL e tratamos)
        sql = text("""
            SELECT
              concurso,
              COALESCE(data_norm::date, NULL) as data_norm,
              data,
              n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,
              NULLIF(ganhadores_11, NULL) as g11,
              NULLIF(ganhadores_12, NULL) as g12,
              NULLIF(ganhadores_13, NULL) as g13,
              NULLIF(ganhadores_14, NULL) as g14,
              NULLIF(ganhadores_15, NULL) as g15
            FROM resultados_oficiais
            WHERE n1 IS NOT NULL
            ORDER BY concurso ASC
        """)
        return db.execute(sql).fetchall()
    finally:
        db.close()

def _coalesce_date(data_norm, data_txt):
    # Usa data_norm se vier; caso contrário tenta parse básico de YYYY-MM-DD / DD/MM/YYYY
    if data_norm:
        return np.datetime64(str(data_norm), 'D')
    if not data_txt:
        return np.datetime64('NaT')
    s = str(data_txt).strip()
    try:
        if "-" in s:  # YYYY-MM-DD
            return np.datetime64(s[:10], 'D')
        if "/" in s:  # DD/MM/YYYY
            d, m, y = s[:10].split("/")
            return np.datetime64(f"{y}-{m}-{d}", 'D')
    except:
        pass
    return np.datetime64('NaT')

def _validate_and_features(nums):
    """
    Retorna (nums_sorted, pares, soma, baixas, altas) ou None se inválido.
    Regras:
      - 15 dezenas
      - valores 1..25
      - únicos
    """
    if nums is None or len(nums) != 15:
        return None
    nums = [int(x) for x in nums]
    if any((x < 1 or x > 25) for x in nums):
        return None
    if len(set(nums)) != 15:
        return None
    nums_sorted = np.array(sorted(nums), dtype=np.int8)
    pares = int(np.sum(nums_sorted % 2 == 0))
    soma = int(np.sum(nums_sorted))
    baixas = int(np.sum(nums_sorted <= 13))
    altas  = 15 - baixas
    return nums_sorted, pares, soma, baixas, altas

def main():
    _ensure_dir(OUT_DIR)
    rows_sql = _fetch_rows()
    n = len(rows_sql)

    rep_map = {}
    winners = []
    struct = np.zeros(n, dtype=DTYPE)  # vamos preencher apenas válidos; depois compacta
    j = 0
    invalidados = 0

    for row in rows_sql:
        (concurso, data_norm, data_txt,
         *nums_and_wins) = row

        nums = nums_and_wins[:15]
        g11,g12,g13,g14,g15 = (nums_and_wins[15:20]
                               if len(nums_and_wins) >= 20 else (None,)*5)

        vf = _validate_and_features(nums)
        if not vf:
            invalidados += 1
            continue

        nums_sorted, pares, soma, baixas, altas = vf
        d = _coalesce_date(data_norm, data_txt)

        # Atualiza rep_map de paridade:
        chave = f"{pares}p_{15-pares}i"
        rep_map[chave] = rep_map.get(chave, 0) + 1

        struct[j]["concurso"] = int(concurso)
        struct[j]["data"]     = d
        struct[j]["numeros"]  = nums_sorted
        struct[j]["pares"]    = pares
        struct[j]["soma"]     = soma
        struct[j]["baixas"]   = baixas
        struct[j]["altas"]    = altas
        struct[j]["g11"]      = int(g11) if g11 is not None else 0
        struct[j]["g12"]      = int(g12) if g12 is not None else 0
        struct[j]["g13"]      = int(g13) if g13 is not None else 0
        struct[j]["g14"]      = int(g14) if g14 is not None else 0
        struct[j]["g15"]      = int(g15) if g15 is not None else 0

        winners.append([int(concurso),
                        struct[j]["g11"], struct[j]["g12"], struct[j]["g13"],
                        struct[j]["g14"], struct[j]["g15"]])
        j += 1

    # Compacta (remove slots não usados por invalidação)
    struct = struct[:j]
    winners = np.array(winners, dtype=np.int32) if winners else np.zeros((0,6), dtype=np.int32)

    # Salva artefatos
    np.save(os.path.join(OUT_DIR, "rows_struct.npy"), struct)
    # Legado (compat): concurso + 15 dezenas
    legacy_rows = np.column_stack([
        struct["concurso"],
        struct["numeros"]
    ])
    np.save(os.path.join(OUT_DIR, "rows.npy"), legacy_rows)
    np.save(os.path.join(OUT_DIR, "rep_map.npy"), rep_map)
    np.save(os.path.join(OUT_DIR, "winners.npy"), winners)

    meta = {
        "total_sql": n,
        "validos": int(len(struct)),
        "invalidos": int(invalidados),
        "primeiro_concurso": int(struct["concurso"][0]) if len(struct) else None,
        "ultimo_concurso": int(struct["concurso"][-1]) if len(struct) else None,
        "datas": {
            "min": str(struct["data"].min()) if len(struct) else None,
            "max": str(struct["data"].max()) if len(struct) else None,
        },
        "artefatos": ["rows_struct.npy","rows.npy","rep_map.npy","winners.npy"]
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Extração finalizada")
    print(f" • Lidos do SQL: {n}")
    print(f" • Válidos: {len(struct)} | Inválidos: {invalidados}")
    print(f" • Artefatos => {OUT_DIR}/rows_struct.npy, rows.npy, rep_map.npy, winners.npy")
    print(f" • Meta => {OUT_DIR}/meta.json")

if __name__ == "__main__":
    main()
