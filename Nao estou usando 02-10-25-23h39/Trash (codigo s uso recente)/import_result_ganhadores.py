import pandas as pd
import sqlite3
from datetime import datetime

CSV_PATH = 'loteria.csv'
DB_PATH = 'faixaricadb.db'

def importar_csv_para_dataframe(caminho_csv):
    df = pd.read_csv(caminho_csv, sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()

    df.rename(columns={
        'Concurso': 'concurso',
        'Data Sorteio': 'Data_Sorteio',
        'Ganhadores 15 acertos': 'gan_15',
        'Ganhadores 14 acertos': 'gan_14',
        'Ganhadores 13 acertos': 'gan_13',
        'Ganhadores 12 acertos': 'gan_12',
        'Ganhadores 11 acertos': 'gan_11',
        'Bola1': 'num_1', 'Bola2': 'num_2', 'Bola3': 'num_3', 'Bola4': 'num_4', 'Bola5': 'num_5',
        'Bola6': 'num_6', 'Bola7': 'num_7', 'Bola8': 'num_8', 'Bola9': 'num_9', 'Bola10': 'num_10',
        'Bola11': 'num_11', 'Bola12': 'num_12', 'Bola13': 'num_13', 'Bola14': 'num_14', 'Bola15': 'num_15'
    }, inplace=True)

    df['Data_Sorteio'] = pd.to_datetime(df['Data_Sorteio'], dayfirst=True).dt.date

    colunas_finais = ['concurso', 'Data_Sorteio'] + [f'num_{i}' for i in range(1, 16)] + \
                     [f'gan_{i}' for i in [11, 12, 13, 14, 15]]

    return df[colunas_finais]

def obter_maior_concurso(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(concurso) FROM resultados")
    resultado = cursor.fetchone()[0]
    return resultado if resultado else 0

def inserir_multiplas_linhas(conn, df):
    df.to_sql('resultados', conn, if_exists='append', index=False)
    print(f"✅ {len(df)} concursos importados com sucesso.")

def main():
    df_csv = importar_csv_para_dataframe(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    maior_concurso_banco = obter_maior_concurso(conn)

    df_novos = df_csv[df_csv['concurso'] > maior_concurso_banco].sort_values('concurso')

    if df_novos.empty:
        print("ℹ️ Nenhum novo concurso a importar.")
    else:
        print(f"📥 {len(df_novos)} concursos novos encontrados (do #{df_novos['concurso'].min()} ao #{df_novos['concurso'].max()})")
        resposta = input("Deseja importar agora? [s/n]: ").strip().lower()
        if resposta == 's':
            inserir_multiplas_linhas(conn, df_novos)
        else:
            print("ℹ️ Operação cancelada.")

    conn.close()

if __name__ == "__main__":
    main()
