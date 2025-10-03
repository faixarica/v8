# treinar_modelos.py
# usar antes de inciar  treinamento [ pip install -r requirements_train.txt ]

import sqlite3
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

def to_binario(jogo):
    binario = [0] * 25
    for n in jogo:
        binario[n - 1] = 1
    return binario

def gerar_palpite_lstm_treinamento():
    print("Treinando modelo LSTM completo...")
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Busca resultados
    cursor.execute("""
        SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 
        FROM resultados_oficiais
    """)
    resultados = cursor.fetchall()
    conn.close()
    
    if len(resultados) < 20:
        raise ValueError("Histórico insuficiente para treinar a LSTM.")
    
    # Prepara dados em binário
    binarios = [to_binario(r) for r in resultados]
    X, y = [], []
    for i in range(len(binarios) - 5):
        X.append(binarios[i:i+5])
        y.append(binarios[i+5])
    X, y = np.array(X), np.array(y)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina o modelo
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(5, 25)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(25, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    model.save("modelo_lstm.h5")
    print("Modelo LSTM completo salvo como 'modelo_lstm.h5'")

def gerar_palpite_lstm_14_treinamento():
    print("Treinando modelo LSTM 14...")
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Busca resultados para LSTM 14
    cursor.execute("""
        SELECT n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 
        FROM resultados_oficiais
        ORDER BY concurso DESC
        LIMIT 100
    """)
    ultimos = cursor.fetchall()
    conn.close()
    
    if len(ultimos) < 10:
        raise ValueError("Histórico insuficiente para treinar LSTM 14.")
    
    # Prepara dados (lógica de treinamento específica para LSTM 14)
    binarios = [to_binario(r) for r in ultimos]
    X, y = [], []
    # Aqui você coloca a lógica específica de treinamento para o LSTM 14
    # Exemplo básico:
    for i in range(len(binarios) - 5):
        X.append([binarios[i+j] for j in range(5)])
        y.append(binarios[i+5])
    
    if len(X) == 0:
        raise ValueError("Dados insuficientes para treinar LSTM 14")
        
    X, y = np.array(X), np.array(y)
    
    # Treina o modelo específico para LSTM 14
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(5, 25)),
        Dropout(0.2),
        Dense(25, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=5, batch_size=8, verbose=1)
    model.save("modelo_lstm_14.h5")
    print("Modelo LSTM 14 salvo como 'modelo_lstm_14.h5'")

if __name__ == "__main__":
    # Executar localmente para gerar os modelos
    gerar_palpite_lstm_treinamento()
    gerar_palpite_lstm_14_treinamento()