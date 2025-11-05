# esse codigo inserir os dados do arquivo CSV 'repete.csv' na tabela 'repete' do banco de dados SQLite
# mas antes executar esse codigo, precisa apagar todos os dados da tabela 'repete' para evitar duplicação

import sqlite3
import csv

# Conectar ao banco de dados SQLite
conn = sqlite3.connect('database.db')  # Substitua pelo nome do seu banco
cursor = conn.cursor()

# Desativar temporariamente a verificação de chave primária para inserir IDs manualmente
cursor.execute("PRAGMA foreign_keys = OFF")

# Abrir e ler o arquivo CSV
with open('repete.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        id_registro = int(row['id'])
        concurso_atual = int(row['concurso_atual'])
        concurso_anterior = int(row['concurso_anterior'])
        qtd_repetidos = int(row['qtd_repetidos'])
        data_registro = row['data_registro']

        cursor.execute('''
            INSERT INTO repete (id, concurso_atual, concurso_anterior, qtd_repetidos, data_registro)
            VALUES (?, ?, ?, ?, ?)
        ''', (id_registro, concurso_atual, concurso_anterior, qtd_repetidos, data_registro))

# Reativar chaves estrangeiras (se necessário)
cursor.execute("PRAGMA foreign_keys = ON")

# Confirmar as transações e fechar a conexão
conn.commit()
conn.close()

print("Importação concluída com IDs preservados!")