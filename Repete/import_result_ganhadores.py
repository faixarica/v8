import pandas as pd
import sqlite3
from datetime import datetime
import os

CSV_PATH = 'loteria.csv'
DB_PATH = 'faixaricadb.db'

def importar_csv_para_dataframe(caminho_csv):
    df = pd.read_csv(caminho_csv, sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Data Sorteio': 'Data_Sorteio',
        'Ganhadores 15 acertos': 'Ganhadores_15_acertos',
        'Ganhadores 14 acertos': 'Ganhadores_14_acertos',
        'Ganhadores 13 acertos': 'Ganhadores_13_acertos',
        'Ganhadores 12 acertos': 'Ganhadores_12_acertos',
        'Ganhadores 11 acertos': 'Ganhadores_11_acertos'
    }, inplace=True)
    df['Data_Sorteio'] = pd.to_datetime(df['Data_Sorteio'], dayfirst=True)
    return df

def obter_ultima_data_do_banco(conn):
    query = "SELECT MAX(Data_Sorteio) FROM resultados"
    cursor = conn.cursor()
    cursor.execute(query)
    resultado = cursor.fetchone()[0]
    return pd.to_datetime(resultado) if resultado else None

def inserir_linha(conn, linha_df):
    linha_df.to_sql('resultados', conn, if_exists='append', index=False)
    print("✅ Último resultado importado com sucesso.")

def main():
    df = importar_csv_para_dataframe(CSV_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    ultima_data_no_banco = obter_ultima_data_do_banco(conn)
    
    ultimo_resultado_df = df.sort_values('Data_Sorteio').iloc[[-1]]
    data_ultimo_csv = ultimo_resultado_df['Data_Sorteio'].values[0]

    if ultima_data_no_banco is None or data_ultimo_csv > ultima_data_no_banco:
        data_formatada = pd.to_datetime(data_ultimo_csv).strftime('%d/%m/%Y')
        resposta = input(f"Deseja importar o último resultado com data de hoje ({data_formatada})? [s/n]: ").strip().lower()
        if resposta == 's':
            inserir_linha(conn, ultimo_resultado_df)
        else:
            print("ℹ️ Operação cancelada pelo usuário.")
    else:
        print("ℹ️ O último resultado já está no banco de dados.")

    conn.close()

if __name__ == "__main__":
    main()
