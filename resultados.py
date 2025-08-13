# resultados_import_debug.py

import csv
import time
from sqlalchemy.orm import Session
from sqlalchemy import text
from db import Session  # Session ligada ao engine

def importar_dados_debug():
    print("🚀 Iniciando função importar_dados_debug()")
    try:
        # 1️⃣ Abrir conexão com banco
        print("🔗 Abrindo conexão com banco...")
        with Session() as db:
            print("✅ Conexão aberta com sucesso!")

            # 2️⃣ Ler CSV
            csv_file = "loteria.csv"
            print(f"📂 Tentando ler CSV: {csv_file}")
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                print(f"✅ CSV carregado. Total de linhas: {len(reader)}")
            except FileNotFoundError:
                print(f"❌ Arquivo CSV não encontrado: {csv_file}")
                return
            except Exception as e:
                print(f"❌ Erro ao ler CSV: {e}")
                return

            if not reader:
                print("⚠️ CSV vazio. Nada a importar.")
                return

            # 3️⃣ Loop de inserção
            inicio = time.time()
            total_ok = 0
            total_err = 0

            for idx, ln in enumerate(reader, start=1):
                print(f"🔹 Processando linha {idx}: concurso {ln.get('Concurso', 'N/A')}")
                try:
                    dados = {
                        "concurso": int(ln["Concurso"]),
                        "data": ln["Data Sorteio"],
                        "n1": int(ln["Bola1"]),
                        "n2": int(ln["Bola2"]),
                        "n3": int(ln["Bola3"]),
                        "n4": int(ln["Bola4"]),
                        "n5": int(ln["Bola5"]),
                        "n6": int(ln["Bola6"]),
                        "n7": int(ln["Bola7"]),
                        "n8": int(ln["Bola8"]),
                        "n9": int(ln["Bola9"]),
                        "n10": int(ln["Bola10"]),
                        "n11": int(ln["Bola11"]),
                        "n12": int(ln["Bola12"]),
                        "n13": int(ln["Bola13"]),
                        "n14": int(ln["Bola14"]),
                        "n15": int(ln["Bola15"]),
                        "ganhadores_15": int(ln["Ganhadores 15 acertos"]),
                        "ganhadores_14": int(ln["Ganhadores 14 acertos"]),
                        "ganhadores_13": int(ln["Ganhadores 13 acertos"]),
                        "ganhadores_12": int(ln["Ganhadores 12 acertos"]),
                        "ganhadores_11": int(ln["Ganhadores 11 acertos"]),
                    }

                    db.execute(text("""
                        INSERT INTO resultados_oficiais (
                            concurso, data, n1, n2, n3, n4, n5,
                            n6, n7, n8, n9, n10,
                            n11, n12, n13, n14, n15,
                            ganhadores_15, ganhadores_14, ganhadores_13,
                            ganhadores_12, ganhadores_11
                        )
                        VALUES (
                            :concurso, :data, :n1, :n2, :n3, :n4, :n5,
                            :n6, :n7, :n8, :n9, :n10,
                            :n11, :n12, :n13, :n14, :n15,
                            :ganhadores_15, :ganhadores_14, :ganhadores_13,
                            :ganhadores_12, :ganhadores_11
                        )
                        ON CONFLICT (concurso) DO UPDATE SET
                            data = EXCLUDED.data,
                            n1 = EXCLUDED.n1, n2 = EXCLUDED.n2, n3 = EXCLUDED.n3,
                            n4 = EXCLUDED.n4, n5 = EXCLUDED.n5, n6 = EXCLUDED.n6,
                            n7 = EXCLUDED.n7, n8 = EXCLUDED.n8, n9 = EXCLUDED.n9,
                            n10 = EXCLUDED.n10, n11 = EXCLUDED.n11, n12 = EXCLUDED.n12,
                            n13 = EXCLUDED.n13, n14 = EXCLUDED.n14, n15 = EXCLUDED.n15,
                            ganhadores_15 = EXCLUDED.ganhadores_15,
                            ganhadores_14 = EXCLUDED.ganhadores_14,
                            ganhadores_13 = EXCLUDED.ganhadores_13,
                            ganhadores_12 = EXCLUDED.ganhadores_12,
                            ganhadores_11 = EXCLUDED.ganhadores_11
                    """), dados)

                    total_ok += 1
                except Exception as e:
                    print(f"❌ Erro no concurso {ln.get('Concurso', 'N/A')}: {e}")
                    total_err += 1

            db.commit()
            duracao = time.time() - inicio
            print(f"\n🎯 Importação finalizada. Sucesso: {total_ok}, erros: {total_err}, tempo: {duracao:.2f}s")

    except Exception as e_outer:
        print("🔥 Erro crítico durante a importação:", e_outer)

# ⚡ Chamada direta da função para teste
if __name__ == "__main__":
    importar_dados_debug()
