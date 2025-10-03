Dicionario treinamento modelo.KERAS

O código fornecido é um **arquivo multifuncional em Python** que combina:

1. **Treinamento de modelos de deep learning** para previsão de resultados de loterias (especificamente, um jogo com 25 números, onde 15 são sorteados — como a Lotofácil brasileira).
2. **Conversão entre formatos de modelo** do TensorFlow/Keras (`.keras` ↔ SavedModel).
3. Um sistema **robusto de re-treinamento automático** (`auto_retrain_robusto`) com fallbacks.
4. Uma **interface mínima em Streamlit** para carregar modelos e gerar palpites com base no histórico de sorteios.

---

## 🔍 **O que o código faz?**

### 1. **Modelos de previsão de loteria**
- Usa **LSTM** (Redes Neurais Recorrentes) para prever quais dos 25 números têm maior probabilidade de aparecer no próximo sorteio.
- Existem **4 arquiteturas**:
  - `LS15`: baseado apenas na sequência de sorteios anteriores.
  - `LS14`: inclui informação de **repetição de números** do sorteio anterior.
  - `LS15++` e `LS14++`: versões **mais ricas**, com features adicionais:
    - Frequência de cada número nos últimos sorteios.
    - "Atraso" (quantos sorteios se passaram desde a última vez que um número saiu).
    - Estatísticas globais: soma total dos números, proporção de pares/ímpares.

### 2. **Dados de entrada**
- Os dados vêm de um **banco de dados SQL** (via `db.py`), tabela `resultados_oficiais`.
- Cada linha tem: `concurso, n1, n2, ..., n15`.
- Opcionalmente, usa uma tabela `repete` com quantos números se repetiram do sorteio anterior.

### 3. **Treinamento**
- Funções como `train_ls15()` e `train_ls14pp()`:
  - Constroem datasets com `build_dataset_*`.
  - Treinam modelos com callbacks (EarlyStopping, ModelCheckpoint, CSVLogger).
  - Salvam em **dois formatos**:
    - `.keras` (arquivo único, conveniente).
    - **SavedModel** (diretório, mais compatível entre versões do TensorFlow).

### 4. **Loss customizada serializável**
- `WeightedBCE`: Binary Cross-Entropy com peso maior para a classe positiva (números sorteados), útil porque apenas 15/25 = 60% dos números são 1 (desbalanceado).
- Usa `@register_keras_serializable` para garantir que o modelo possa ser salvo/carregado sem erros.

### 5. **Auto-retrain robusto**
- Tenta usar um módulo `pipeline_data.get_training_data()` (ideal para produção).
- Se falhar, **reconstrói os dados do zero** usando `fetch_history()` + builders.
- Faz **backup** do modelo antigo antes de re-treinar.
- Tem **fallback** para criar um modelo simples se tudo falhar.

### 6. **App Streamlit**
- Interface web simples para:
  - Escolher um modelo salvo (`.keras` ou SavedModel).
  - Carregar os últimos sorteios.
  - Gerar um **palpite com os 15 números mais prováveis**.

---

## ⚙️ **Parâmetros necessários para usar**

### ✅ **Pré-requisitos obrigatórios**
1. **Ambiente Python** com:
   - `tensorflow >= 2.12` (Keras 3+)
   - `streamlit` (só se usar a interface)
   - `sqlalchemy`, `numpy`
2. **Arquivo `db.py`** no mesmo diretório, com:
   - Uma variável `Session` (SQLAlchemy session factory).
   - Acesso a um banco com tabela `resultados_oficiais(concurso, n1..n15)`.
   - (Opcional) Tabela `repete(concurso_atual, qtd_repetidos)`.

> Se não tiver `db.py`, o código falhará em `fetch_history()`.

---

## 🧪 **Como usar (linha de comando)**

### 1. **Treinar um modelo a partir do banco**
```bash
# Treina LS15++ com os últimos 1000 sorteios
python train_ls_models_v1_and_auto_retrain_updates.py --auto-retrain
```

> Isso executa `main_auto_retrain()`, que tenta usar `pipeline_data` ou fallback para `fetch_history`.

### 2. **Converter um modelo .keras para SavedModel**
```bash
python train_ls_models_v1_and_auto_retrain_updates.py \
  --convert models/modelo_ls15pp.keras \
  --savedir models/modelo_ls15pp_saved
```

### 3. **Rodar a interface Streamlit**
```bash
streamlit run train_ls_models_v1_and_auto_retrain_updates.py -- --streamlit
```

> A interface permite carregar modelos e gerar palpites.

---

## 📁 **Estrutura de diretórios esperada**
```
projeto/
├── train_ls_models_v1_and_auto_retrain_updates.py  ← este arquivo
├── db.py                                           ← obrigatório
├── models/                                         ← criado automaticamente
│   ├── modelo_ls15pp.keras
│   ├── modelo_ls15pp_saved/                        ← SavedModel
│   └── ...
├── metrics/                                        ← logs de treino
└── backup_models/                                  ← backups automáticos
```

---

## ⚠️ **Avisos importantes**

- **Prefira SavedModel em produção**: o formato `.keras` pode falhar ao carregar em ambientes com versões diferentes do TensorFlow/Keras.
- **A loss `WeightedBCE` deve ser registrada** (`@register_keras_serializable`) — o código já faz isso corretamente.
- **O banco de dados é essencial**: sem `db.py` e a tabela `resultados_oficiais`, nada funciona.
- **Não é garantia de ganho**: modelos preditivos em loterias têm limitações estatísticas (os sorteios são aleatórios!).

---

## 📌 Resumo dos comandos úteis

| Ação | Comando |
|------|--------|
| Ver opções | `python arquivo.py --help` |
| Re-treinar modelos | `python arquivo.py --auto-retrain` |
| Converter modelo | `python arquivo.py --convert caminho/modelo.keras` |
| Rodar Streamlit | `streamlit run arquivo.py -- --streamlit` |

---

Se você tiver o `db.py` configurado corretamente, este script é **autocontido** e pode ser usado para treinar, converter, re-treinar e até fazer previsões via interface web.