import csv
import sqlite3

def importar_dados():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_oficiais (
            concurso INTEGER PRIMARY KEY,
            data TEXT NOT NULL,
            n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER,
            n6 INTEGER, n7 INTEGER, n8 INTEGER, n9 INTEGER, n10 INTEGER,
            n11 INTEGER, n12 INTEGER, n13 INTEGER, n14 INTEGER, n15 INTEGER,
            ganhadores_15 INTEGER, ganhadores_14 INTEGER, ganhadores_13 INTEGER,
            ganhadores_12 INTEGER, ganhadores_11 INTEGER
        )
    ''')
    
    total_importados = 0
    erros = 0

    with open('loteria.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        
        for linha in reader:
            try:
                dados = [
                    int(linha['Concurso']),
                    linha['Data Sorteio'],
                    *[int(linha[f'Bola{i}']) for i in range(1, 16)],
                    int(linha['Ganhadores 15 acertos']),
                    int(linha['Ganhadores 14 acertos']),
                    int(linha['Ganhadores 13 acertos']),
                    int(linha['Ganhadores 12 acertos']),
                    int(linha['Ganhadores 11 acertos'])
                ]
                
                cursor.execute('''
                    INSERT OR REPLACE INTO resultados_oficiais VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', dados)

                total_importados += 1

            except Exception as e:
                erros += 1

    conn.commit()
    conn.close()

    return {
        "total": total_importados,
        "importados": [],  # você pode capturar e incluir concursos aqui se quiser
        "erros": [f"{erros} erro(s) encontrado(s)"] if erros > 0 else []
    }

if __name__ == "__main__":
    importar_dados()