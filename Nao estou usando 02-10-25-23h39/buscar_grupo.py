# buscar_grupo.py
from models P import ResultadosOficiais
from db import Session
import sys

def buscar_grupo_sorteados(numeros_pesquisados):
    """
    Busca se um grupo exato de 15 números já foi sorteado na tabela 'resultados_oficiais'.
    Compara ignorando ordem — ou seja, [1,2,3] == [3,2,1].
    
    :param numeros_pesquisados: Lista de 15 inteiros.
    :return: Lista de dicionários com concursos encontrados, ou [] se não houver.
    """
    if len(numeros_pesquisados) != 15:
        raise ValueError("❌ O grupo deve conter exatamente 15 números.")

    # Ordena os números pesquisados para comparação
    numeros_pesquisados = sorted(numeros_pesquisados)
    session = Session()

    try:
        print("🔍 Consultando banco de dados... (pode levar alguns segundos)")
        
        # Busca todos os registros da tabela
        resultados = session.query(ResultadosOficiais).all()
        
        encontrados = []
        
        for registro in resultados:
            # Extrai os 15 números sorteados e ordena
            numeros_sorteio = sorted([
                registro.n1, registro.n2, registro.n3, registro.n4, registro.n5,
                registro.n6, registro.n7, registro.n8, registro.n9, registro.n10,
                registro.n11, registro.n12, registro.n13, registro.n14, registro.n15
            ])
            
            # Compara os conjuntos ordenados
            if numeros_pesquisados == numeros_sorteio:
                encontrados.append({
                    'concurso': registro.concurso,
                    'data': registro.data,
                    'numeros': numeros_sorteio
                })
        
        return encontrados

    except Exception as e:
        print(f"❌ Erro ao consultar o banco: {e}")
        return []
    
    finally:
        session.close()

# === Exemplo de uso ===
if __name__ == "__main__":
    # 🔢 Grupo de 15 números que deseja verificar
    numeros_desejados = [3, 7, 12, 15, 18, 21, 23, 25, 27, 30, 33, 36, 39, 42, 45]
    
    print(f"🔎 Buscando o grupo: {sorted(numeros_desejados)}...\n")
    
    resultados = buscar_grupo_sorteados(numeros_desejados)
    
    if resultados:
        print(f"✅ Encontrado {len(resultados)} ocorrência(s):\n")
        for r in resultados:
            print(f"   🎯 Concurso: {r['concurso']} | Data: {r['data']}")
            print(f"      Números: {r['numeros']}\n")
    else:
        print("❌ Nenhum resultado encontrado. Este grupo de 15 números ainda não foi sorteado.")