# v8/db_utils.py
import os
import csv
import json
import random
from pathlib import Path

def _parse_numbers(raw):
    """
    Aceita formatos como:
      - lista JSON: [1,2,3,...]
      - string "1,2,3,..."
      - string "1 2 3 ..."
    Retorna lista de 15 ints ordenados.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        nums = [int(x) for x in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        try:
            # tenta JSON
            val = json.loads(s)
            if isinstance(val, list):
                nums = [int(x) for x in val]
            else:
                # cai para split
                nums = [int(x) for x in s.replace(",", " ").split()]
        except json.JSONDecodeError:
            nums = [int(x) for x in s.replace(",", " ").split()]
    else:
        return None

    # normaliza: únicos, entre 1..25, pega no máx 15 e ordena
    nums = sorted({n for n in nums if 1 <= n <= 25})
    if len(nums) < 15:
        # se vier menos, completa aleatoriamente só para não quebrar treino
        falta = 15 - len(nums)
        pool = [n for n in range(1, 26) if n not in nums]
        nums += random.sample(pool, falta)
        nums = sorted(nums[:15])
    else:
        nums = nums[:15]
    return nums

def _read_csv(path, limit=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # tenta detectar coluna com dezenas
        candidate_cols = [c for c in reader.fieldnames if c and c.lower() in ("numbers", "dezenas", "dezenas15", "nums")]
        if not candidate_cols:
            # tenta colunas d01..d15
            dcols = [c for c in reader.fieldnames if c.lower().startswith("d") and c[1:].isdigit()]
            for r in reader:
                if dcols:
                    nums = [int(r[c]) for c in sorted(dcols) if r[c]]
                else:
                    # última alternativa: junta todas as colunas
                    nums = []
                    for c in reader.fieldnames:
                        try:
                            v = int(r[c])
                            if 1 <= v <= 25:
                                nums.append(v)
                        except:
                            pass
                rows.append({"numbers": _parse_numbers(nums)})
        else:
            col = candidate_cols[0]
            for r in reader:
                rows.append({"numbers": _parse_numbers(r[col])})
    if limit:
        rows = rows[-limit:]
    # filtra nulos
    rows = [r for r in rows if r["numbers"] is not None]
    return rows

def _read_json(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    for item in data:
        if isinstance(item, dict):
            # tenta chaves comuns
            for key in ("numbers", "dezenas", "dezenas15", "nums"):
                if key in item:
                    rows.append({"numbers": _parse_numbers(item[key])})
                    break
        elif isinstance(item, list):
            rows.append({"numbers": _parse_numbers(item)})
        elif isinstance(item, str):
            rows.append({"numbers": _parse_numbers(item)})
    if limit:
        rows = rows[-limit:]
    rows = [r for r in rows if r["numbers"] is not None]
    return rows

def _generate_synthetic(n_rows):
    """Gera sorteios sintéticos válidos (15 dezenas únicas de 1..25)."""
    out = []
    base = list(range(1, 26))
    for _ in range(n_rows):
        out.append({"numbers": sorted(random.sample(base, 15))})
    return out

def fetch_history(last_n=1500, include_repeats=False):
    """
    Retorna lista de dicts no formato:
      [{ "numbers": [int, int, ..., int] }, ...]
    Tenta ler de arquivos locais comuns; se não achar, gera dados sintéticos.
    """
    # caminhos mais comuns (ajuste se necessário)
    candidates = [
        "data/lotofacil_history.csv",
        "lotofacil_history.csv",
        "data/history.csv",
        "history.csv",
        "data/lotofacil_history.json",
        "lotofacil_history.json",
        "data/history.json",
        "history.json",
    ]

    for p in candidates:
        path = Path(p)
        if path.exists():
            if path.suffix.lower() == ".csv":
                return _read_csv(path, limit=last_n)
            if path.suffix.lower() == ".json":
                return _read_json(path, limit=last_n)

    # fallback: sintético para não travar o treino
    return _generate_synthetic(last_n)
    