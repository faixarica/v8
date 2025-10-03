# auto_retrain_robusto.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import traceback
import shutil

# ===============================
# CONFIGURAÇÕES
# ===============================
MODEL_DIR = "./models"
METRICS_DIR = "./metrics"
BACKUP_DIR = "./backup_models"

EPOCHS = 50
BATCH_SIZE = 32

MODELS = {
    "LS14++": "modelo_ls14pp.keras",
    "LS15++": "modelo_ls15pp.keras"
}

# ===============================
# FUNÇÕES AUXILIARES
# ===============================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def backup_model(model_file):
    """Cria backup do modelo atual antes de sobrescrever"""
    src_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(src_path):
        dst_path = os.path.join(BACKUP_DIR, model_file)
        shutil.copy2(src_path, dst_path)
        print(f"[INFO] Backup criado para {model_file}")

def build_model(input_dim):
    """Cria modelo denso genérico"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Ajuste se multiclass
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# FUNÇÃO DE TREINO
# ===============================
    
def retrain_model(model_name, model_file, X, y):
    """
    Treina ou retreina um modelo, suportando:
      - LS14++: multi-input (lista de arrays)
      - LS15++: single-input (array único)
    
    Parâmetros:
        model_name (str): Nome simbólico do modelo
        model_file (str): Nome do arquivo .h5 do modelo
        X (array ou lista de arrays): Dados de entrada
        y (array): Labels
    Retorna:
        model: Modelo treinado
        history: Histórico de treino (None se falhou)
    """
    model_path = os.path.join(MODEL_DIR, model_file)
    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.csv")
    
    # Backup do modelo existente
    backup_model(model_file)
    
    # Inicializa history para evitar UnboundLocalError
    history = None

    # Carrega ou cria o modelo
    if os.path.exists(model_path):
        try:
            print(f"[INFO] Carregando modelo existente (compile=False): {model_file}")
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print(f"[INFO] Modelo recompilado com sucesso.")
        except Exception as e:
            print(f"[WARN] Falha ao carregar/recompilar {model_file}: {e}")
            print("[INFO] Criando novo modelo a partir do build_model().")
            model = build_model(input_dim=X[0].shape[1] if isinstance(X, list) else X.shape[1])
    else:
        print(f"[INFO] Criando novo modelo: {model_file}")
        model = build_model(input_dim=X[0].shape[1] if isinstance(X, list) else X.shape[1])
    
    # Callbacks
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(metrics_file, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    try:
        # Detecta se é multi-input ou single-input
        if isinstance(X, list):
            print(f"[INFO] Treinando modelo multi-input ({len(X)} inputs).")
            history = model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                validation_split=0.2,
                callbacks=[checkpoint, csv_logger, early_stop]
            )
        else:
            print("[INFO] Treinando modelo single-input.")
            history = model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                validation_split=0.2,
                callbacks=[checkpoint, csv_logger, early_stop]
            )
        print(f"[SUCCESS] Modelo {model_name} treinado com sucesso!")
    except Exception as e:
        print(f"[ERROR] Erro no treino do modelo {model_name}: {e}")
    
    return model, history



# =============
# FUNÇÃO PRINCIPAL
# ===============================
def main():
    ensure_dirs()
    
    # ===============================
    # PEGAR DADOS DO PIPELINE ATUAL
    # ===============================
    try:
        from pipeline_data import get_training_data  # sua função que retorna X, y
        X, y = get_training_data()
    except Exception as e:
        print(f"[ERROR] Não foi possível obter dados do pipeline: {e}")
        traceback.print_exc()
        return
    
    # Treinar cada modelo
    for model_name, model_file in MODELS.items():
        retrain_model(model_name, model_file, X, y)

if __name__ == "__main__":
    main()
