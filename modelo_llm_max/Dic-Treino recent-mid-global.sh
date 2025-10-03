Dic - treinos (recente / médio / global)

1. Rodar diferentes treinos (recente / médio / global)

Você pode executar 3 comandos diferentes (um para cada janela histórica) e salvar os modelos em pastas separadas:

📌 Modelo Recente (curto prazo, capta tendências rápidas)
python train_llm_loteria.py --model both --last_n 500 --epochs 50 --batch 32 --out ./models/recent

📌 Modelo Médio (equilíbrio entre curto e longo prazo)
python train_llm_loteria.py --model both --last_n 1000 --epochs 60 --batch 32 --out ./models/mid

📌 Modelo Global (usa tudo desde o começo, capta longo prazo)
python train_llm_loteria.py --model both --last_n 1550 --epochs 80 --batch 64 --out ./models/global


👉 Assim, no final, você terá 3 modelos diferentes salvos:

./models/recent/best_hits.keras

./models/mid/best_hits.keras

./models/global/best_hits.keras

🔹 2. Comparar resultados

Durante o treino, o código já imprime val_mean_hits.
Anote ou salve em arquivo os valores de cada modelo, assim você pode comparar:

Qual teve maior média de acertos (mean_hits).

Qual teve mais estabilidade (não só picos).

🔹 3. Ensemble (juntar modelos)

Depois que tiver os modelos treinados, você pode criar um script rápido para carregar todos e combinar as previsões.