# auto_retrain_robusto.py (VERSÃO FINAL – AJUSTADA)
import os
import shutil
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import train_ls_models_v1_and_auto_retrain_updates as train_ls_models

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativa oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduz logs

# Configurações adicionais para estabilidade
tf.config.run_functions_eagerly(True)  # Para debugging
# ===============================
# MODEL_DIR = "./models"
MODEL_DIR = "./tmp"
METRICS_DIR = "./metrics"
BACKUP_DIR = "./backup_models"

EPOCHS = 50
BATCH_SIZE = 16

MODELS = {
    "LS14++": "modelo_ls14pp.keras",
    "LS15++": "modelo_ls15pp.keras"
}

WINDOW = 50
LAST_N = 500

# ===============================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def backup_model(model_file):
    src_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(src_path):
        dst_path = os.path.join(BACKUP_DIR, model_file)
        shutil.copy2(src_path, dst_path)
        print(f"[INFO] Backup criado para {model_file}")

# ===============================
# Weighted BCE seguro contra overflow
def WeightedBCE(pos_weight=3.0):
    # Temporariamente usa BCE padrão para testar
    return 'binary_crossentropy'

# ===============================
def get_training_data(model_type="both", last_n=LAST_N, window=WINDOW):
    output = {}

    if model_type in ("ls14pp", "both"):
        rows_rep = train_ls_models.fetch_history(last_n=last_n, include_repeats=True)
        if isinstance(rows_rep, tuple) and len(rows_rep) == 2:
            rows, rep_map = rows_rep
        else:
            rows, rep_map = rows_rep, {}
        X_seq, X_hist, X_freq, X_atraso, X_global, y_ls14pp = train_ls_models.build_dataset_ls14pp(rows, rep_map, window)
        X_ls14pp = [arr.astype(np.float32) for arr in [X_seq, X_hist, X_freq, X_atraso, X_global]]
        y_ls14pp = np.clip(y_ls14pp.astype(np.float32), 0, 1)  # binariza
        output["LS14++"] = (X_ls14pp, y_ls14pp)

    if model_type in ("ls15pp", "both"):
        rows = train_ls_models.fetch_history(last_n=last_n)
        X_seq, X_freq, X_atraso, X_global, y_ls15pp = train_ls_models.build_dataset_ls15pp(rows, window)
        X_ls15pp = [arr.astype(np.float32) for arr in [X_seq, X_freq, X_atraso, X_global]]
        y_ls15pp = np.clip(y_ls15pp.astype(np.float32), 0, 1)
        output["LS15++"] = (X_ls15pp, y_ls15pp)

    return output

# ===============================
def preprocess_X_y(X, y):
    if isinstance(X, list):
        X = [arr.astype(np.float32) for arr in X]
    else:
        X = X.astype(np.float32)
    y = np.clip(y.astype(np.float32), 0, 1)
    
    # valida shapes
    n_samples = X[0].shape[0] if isinstance(X, list) else X.shape[0]
    for arr in (X if isinstance(X, list) else [X]):
        if arr.shape[0] != n_samples:
            raise ValueError("Arrays em X têm número diferente de samples")
    if y.shape[0] != n_samples:
        raise ValueError("X e y têm número diferente de samples")
    
    # ✅ Adiciona validação robusta
    X, y = validate_and_clean_data(X, y)
    
    return X, y

def validate_and_clean_data(X, y):
    """Valida e limpa dados para evitar overflow"""
    
    def check_array(arr, name):
        if isinstance(arr, list):
            for i, sub_arr in enumerate(arr):
                check_array(sub_arr, f"{name}[{i}]")
        else:
            # Verifica valores inválidos
            if np.any(np.isnan(arr)):
                print(f"[WARN] NaN encontrado em {name}, substituindo por 0")
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            
            if np.any(np.isinf(arr)):
                print(f"[WARN] Inf encontrado em {name}, substituindo por limites")
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Limita valores extremos
            arr = np.clip(arr, -1000, 1000)
            
            # Verifica se há valores muito grandes
            max_val = np.max(np.abs(arr))
            if max_val > 1e6:
                print(f"[WARN] Valores muito grandes em {name} (max: {max_val}), normalizando")
                arr = arr / max_val if max_val > 0 else arr
                
        return arr
    
    # Aplica validação
    if isinstance(X, list):
        X = [check_array(arr, f"X[{i}]") for i, arr in enumerate(X)]
    else:
        X = check_array(X, "X")
    
    y = check_array(y, "y")
    
    return X, y

# ===============================
# ===============================
def validate_and_clean_data(X, y):
    # ... código existente ...

# ===============================
# ✅ ADICIONE ESSA FUNÇÃO AQUI
def save_model_as_savedmodel(model, model_path):
    """Salva modelo no formato SavedModel para produção"""
    try:
        # Remove .keras e adiciona _saved
        saved_dir = model_path.replace(".keras", "") + "_saved"
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(saved_dir), exist_ok=True)
        
        # Salva no formato SavedModel
        model.save(saved_dir, save_format='tf')
        print(f"[INFO] SavedModel salvo em: {saved_dir}")
        
        # Lista arquivos gerados
        if os.path.exists(saved_dir):
            files = os.listdir(saved_dir)
            print(f"[INFO] Arquivos gerados: {files[:5]}...")  # Mostra primeiros 5
            
        return saved_dir
    except Exception as e:
        print(f"[ERROR] Falha ao salvar SavedModel: {e}")
        traceback.print_exc()
        return None

# ===============================
def retrain_model(model_name, model_file, X, y, lr=1e-3, pos_weight=3.0):
    # Debug dos dados
    debug_data_shapes_and_values(X, y, model_name)
    
    X, y = preprocess_X_y(X, y)
    model_path = os.path.join(MODEL_DIR, model_file)
    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.csv")

    backup_model(model_file)

    model = None
    if os.path.exists(model_path):
        print(f"[INFO] Carregando modelo existente: {model_path}")
        model = load_model(model_path, compile=False)
        if model_name == "LS14++":
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=WeightedBCE(pos_weight=pos_weight),
                          metrics=['accuracy'])
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    else:
        print(f"[INFO] Construindo novo modelo: {model_name}")
        seq_shape = (X[0].shape[1], X[0].shape[2])
        if model_name == "LS14++":
            model = train_ls_models.build_lstm_ls14pp_hybrid(seq_shape, lr=lr, pos_weight=pos_weight)
        elif model_name == "LS15++":
            model = train_ls_models.build_lstm_ls15pp(seq_shape=seq_shape, lr=lr)
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")

    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(metrics_file, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"[INFO] Treinando {model_name} ...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[checkpoint, csv_logger, early_stop],
        verbose=2
    )

    # TRECHO NOVO - SUBSTITUIR
    saved_model_path = save_model_as_savedmodel(model, model_path)

    return model

# ===============================
def main():
    ensure_dirs()

    try:
        data = get_training_data("both")
    except Exception as e:
        print(f"[ERROR] Não foi possível obter dados do pipeline: {e}")
        traceback.print_exc()
        return

    for model_name, model_file in MODELS.items():
        if model_name not in data:
            print(f"[WARN] Dados para {model_name} não encontrados — pulando.")
            continue
        X, y = data[model_name]
        retrain_model(model_name, model_file, X, y)


def debug_data_shapes_and_values(X, y, name=""):
    """Debug dos dados de entrada"""
    print(f"\n=== DEBUG {name} ===")
    if isinstance(X, list):
        for i, arr in enumerate(X):
            print(f"X[{i}]: shape={arr.shape}, dtype={arr.dtype}")
            print(f"  min={np.min(arr):.6f}, max={np.max(arr):.6f}, mean={np.mean(arr):.6f}")
            print(f"  has_nan={np.any(np.isnan(arr))}, has_inf={np.any(np.isinf(arr))}")
    else:
        print(f"X: shape={X.shape}, dtype={X.dtype}")
        print(f"  min={np.min(X):.6f}, max={np.max(X):.6f}, mean={np.mean(X):.6f}")
    
    print(f"y: shape={y.shape}, dtype={y.dtype}")
    print(f"  min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}")
    print("==================\n")
    
if __name__ == "__main__":
    main()
