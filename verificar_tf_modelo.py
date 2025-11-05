import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

print("📦 TensorFlow versão:", tf.__version__)

# Verifica se o arquivo existe
if not os.path.exists("modelo_lstm_14.h5"):
    print("❌ Arquivo 'modelo_lstm_14.h5' não encontrado no diretório atual.")
else:
    try:
        print("🔄 Carregando modelo...")
        model = load_model("modelo_lstm_14.h5")

        print("✅ Modelo carregado com sucesso.")
        entrada_teste = np.random.rand(1, 5, 25)
        resultado = model.predict(entrada_teste)

        print("✅ Previsão executada com sucesso. Resultado:")
        print(resultado)
    except Exception as e:
        print("❌ Erro ao carregar ou executar o modelo:")
        print(e)
        print("Certifique-se de que o arquivo está no diretório correto e que o TensorFlow está instalado corretamente.")