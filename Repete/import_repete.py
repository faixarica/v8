# import_repete.py
import csv
from db import engine
import pandas as pd
from sqlalchemy import text

def import_repete_csv(file_path, batch_size=10000):
    """
    Importa dados do CSV para a tabela repete em batches para melhor performance
    """
    print(f"Iniciando importação do arquivo: {file_path}")
    
    # Criar uma conexão
    with engine.connect() as conn:
        # Contar total de linhas (opcional, para progress tracking)
        total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8')) - 1  # -1 para header
        print(f"Total de linhas a importar: {total_lines}")
        
        # Ler o CSV em chunks usando pandas para melhor performance
        chunk_count = 0
        
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            # Converter data_registro para datetime se necessário
            chunk['data_registro'] = pd.to_datetime(chunk['data_registro'])
            
            # Preparar os dados para inserção
            # Remover a coluna 'id' já que é SERIAL no PostgreSQL
            data_to_insert = chunk.drop('id', axis=1)
            
            # Converter para lista de dicionários
            records = data_to_insert.to_dict('records')
            
            # Inserir em batch
            if records:  # Verificar se há registros
                # Criar a query de inserção
                insert_query = text("""
                    INSERT INTO repete (concurso_atual, concurso_anterior, qtd_repetidos, data_registro)
                    VALUES (:concurso_atual, :concurso_anterior, :qtd_repetidos, :data_registro)
                """)
                
                # Executar a inserção em batch
                conn.execute(insert_query, records)
                conn.commit()  # Commit do batch
                
                chunk_count += len(records)
                print(f"Importadas {chunk_count} linhas...")
        
        print(f"Importação concluída! Total de {chunk_count} registros inseridos.")

def import_repete_csv_alternative(file_path):
    """
    Método alternativo usando csv.reader para arquivos muito grandes
    """
    print(f"Iniciando importação do arquivo: {file_path}")
    
    batch_size = 10000
    batch = []
    count = 0
    
    with engine.connect() as conn:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Preparar o registro (ignorar o id do CSV pois é SERIAL)
                record = {
                    'concurso_atual': int(row['concurso_atual']),
                    'concurso_anterior': int(row['concurso_anterior']),
                    'qtd_repetidos': int(row['qtd_repetidos']),
                    'data_registro': row['data_registro']
                }
                
                batch.append(record)
                
                # Quando o batch atinge o tamanho definido, insere no banco
                if len(batch) >= batch_size:
                    insert_query = text("""
                        INSERT INTO repete (concurso_atual, concurso_anterior, qtd_repetidos, data_registro)
                        VALUES (:concurso_atual, :concurso_anterior, :qtd_repetidos, :data_registro)
                    """)
                    conn.execute(insert_query, batch)
                    conn.commit()
                    count += len(batch)
                    print(f"Importadas {count} linhas...")
                    batch = []  # Limpar o batch
            
            # Inserir os registros restantes
            if batch:
                insert_query = text("""
                    INSERT INTO repete (concurso_atual, concurso_anterior, qtd_repetidos, data_registro)
                    VALUES (:concurso_atual, :concurso_anterior, :qtd_repetidos, :data_registro)
                """)
                conn.execute(insert_query, batch)
                conn.commit()
                count += len(batch)
                print(f"Importadas {count} linhas...")
        
        print(f"Importação concluída! Total de {count} registros inseridos.")

if __name__ == "__main__":
    # Certifique-se de que o arquivo repete.csv está no mesmo diretório
    csv_file_path = "repete.csv"
    
    try:
        # Usar o método com pandas (mais rápido)
        import_repete_csv(csv_file_path)
        
        # Ou usar o método alternativo se preferir:
        # import_repete_csv_alternative(csv_file_path)
        
    except Exception as e:
        print(f"Erro durante a importação: {e}")