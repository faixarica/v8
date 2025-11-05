"""
Arquivo único contendo:
 - Versão atualizada de train_ls_models_v1 (compatível com Streamlit / Py3.11)
 - Funções de conversão .keras -> SavedModel (tf)
 - Versão robusta de auto_retrain_robusto com fallback quando pipeline_data não existir
 - Exemplo mínimo de app Streamlit (streamlit_predict_app) para carregar modelos .keras / SavedModel e gerar uma predição a partir do histórico

INSTRUÇÕES RÁPIDAS:
 - Substitua/adicione este arquivo no mesmo diretório dos seus scripts atuais.
 - Rode `python train_ls_models_v1_and_auto_retrain_updates.py --help` para ver opções (treinar, converter).
 - Rode `python auto_retrain_robusto_updated.py` (há uma função main dentro do bloco auto-retrain) ou use o módulo retrain.
 - O Streamlit example está no final e pode ser usado como `streamlit run train_ls_models_v1_and_auto_retrain_updates.py --streamlit`.

Este arquivo contém código adaptado para:
 - salvar modelos tanto em formato .keras quanto em SavedModel (TF) — reduz problemas de incompatibilidade entre ambientes Keras/TensorFlow
 - garantir que a loss customizada (WeightedBCE) seja serializável (usando @register_keras_serializable)
 - fallback no auto-retrain para construir dados a partir de fetch_history quando pipeline_data não estiver disponível

IMPORTANTE: para produção, sempre prefira **usar o SavedModel (diretório)** como artefato a ser implantado. .keras pode, às vezes, falhar ao desserializar se houver incompatibilidade de versão do Keras.
"""

# -------------------- IMPORTS GERAIS --------------------
import argparse
import logging
import os
import shutil
import traceback
import numpy as np
import tensorflow as tf
# Substituir por esta versão (usa textwrap do stdlib, trata erros, fecha sessão)
import textwrap  # coloque no topo do arquivo se ainda não estiver importado
from db import Session
from sqlalchemy import text   # <-- adicione esta linha
from typing import Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# diretório padrão para salvar modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
# Se seu projeto usa um db.py local, mantemos a importação original

try:
    from db import Session
except Exception:
    Session = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ===================== UTILITÁRIOS / DATA =====================

def to_binary(jogo):
    b = np.zeros(25, dtype=np.uint8)
    for n in jogo:
        try:
            idx = int(n) - 1
            if 0 <= idx < 25:
                b[idx] = 1
            else:
                logging.warning(f"[to_binary] Valor fora do intervalo 1-25 detectado: {n}")
        except Exception as e:
            logging.error(f"[to_binary] Erro ao processar valor {n}: {e}")
    return b

# Substituir por esta versão (usa textwrap do stdlib, trata erros, fecha sessão)
import textwrap  # coloque no topo do arquivo se ainda não estiver importado

def fetch_history(last_n=None, include_repeats=False):
    """
    Retorna:
      - se include_repeats=False: rows (lista de tuples) com (concurso, n1..n15)
      - se include_repeats=True: (rows, rep_map) onde rep_map é {concurso: qtd_repetidos}
    Observações:
      - Usa o Session exportado por db.py (assume que Session está definido).
      - Garante fechamento da sessão em finally.
    """
    if Session is None:
        raise RuntimeError("Session (db) não encontrado — inclua seu db.py ou rode em um ambiente com DB")

    db = Session()
    try:
        if last_n:
            sql = textwrap.dedent("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso DESC
                LIMIT :lim
            """)
            rows = db.execute(text(sql), {"lim": last_n}).fetchall()
            rows = list(reversed(rows))
        else:
            sql = textwrap.dedent("""
                SELECT concurso, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                       n11,n12,n13,n14,n15
                FROM resultados_oficiais
                ORDER BY concurso ASC
            """)
            rows = db.execute(text(sql)).fetchall()

        # normalizar para lista de tuplas/ints
        cleaned = []
        for r in rows:
            try:
                cleaned.append(tuple(int(v) for v in r))
            except Exception:
                # fallback: tentar converter parcialmente
                cleaned.append(tuple([int(v) if isinstance(v, (int, str)) and str(v).isdigit() else v for v in r]))

        if include_repeats:
            # tenta obter tabela 'repete' se existir; senão devolve map vazio
            try:
                rep_rows = db.execute(text("SELECT concurso_atual, qtd_repetidos FROM repete")).fetchall()
                rep_map = {int(r[0]): int(r[1]) for r in rep_rows}
            except Exception:
                rep_map = {}
            return cleaned, rep_map

        return cleaned

    finally:
        try:
            db.close()
        except Exception:
            pass



# Dataset builders (copiados e mantidos) — adequados para o fallback do auto-retrain
# ... (mantive as no seu arquivo original: build_dataset_ls15, build_dataset_ls14, build_dataset_ls14pp, build_dataset_ls15pp)

# Para economizar espaço visual aqui, vou colocar as implementações diretamente (copiadas do seu arquivo original):
def build_dataset_ls15(rows, window):
    X, y = [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(rows[i + window][1:])
        X.append(seq)
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
def build_dataset_ls14(rows, rep_map, window):
    X_seq, X_hist, y = [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        seq = [to_binary(r[1:]) for r in seq_rows]
        target_row = rows[i + window]
        target = to_binary(target_row[1:])
        concurso_target = int(target_row[0])
        repeticoes = rep_map.get(concurso_target, 0) / 15.0
        X_seq.append(seq)
        X_hist.append([repeticoes])
        y.append(target)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_hist, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )

def build_dataset_ls14pp(rows, rep_map, window):
    X_seq, X_hist, X_freq, X_atraso, X_global, y = [], [], [], [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        target_row = rows[i + window]

        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(target_row[1:])
        concurso_target = int(target_row[0])

        repeticoes = rep_map.get(concurso_target, 0) / 15.0
        freq_vec = np.sum([to_binary(r[1:]) for r in seq_rows], axis=0) / float(window)

        atraso_vec = np.zeros(25, dtype=np.float32)
        for d in range(1, 26):
            atraso = 0
            for past in reversed(seq_rows):
                atraso += 1
                if d in past[1:]:
                    break
            atraso_vec[d-1] = atraso / float(window)

        soma = sum(target_row[1:]) / (25.0 * 15.0)
        pares = sum(1 for x in target_row[1:] if x % 2 == 0) / 15.0

        X_seq.append(seq)
        X_hist.append([repeticoes])
        X_freq.append(freq_vec)
        X_atraso.append(atraso_vec)
        X_global.append([soma, pares])
        y.append(target)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_hist, dtype=np.float32),
        np.array(X_freq, dtype=np.float32),
        np.array(X_atraso, dtype=np.float32),
        np.array(X_global, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )
def build_dataset_ls15pp(rows, window):
    X_seq, X_freq, X_atraso, X_global, y = [], [], [], [], []
    for i in range(len(rows) - window):
        seq_rows = rows[i:i + window]
        target_row = rows[i + window]

        seq = [to_binary(r[1:]) for r in seq_rows]
        target = to_binary(target_row[1:])

        freq_vec = np.sum([to_binary(r[1:]) for r in seq_rows], axis=0) / float(window)

        atraso_vec = np.zeros(25, dtype=np.float32)
        for d in range(1, 26):
            atraso = 0
            for past in reversed(seq_rows):
                atraso += 1
                if d in past[1:]:
                    break
            atraso_vec[d-1] = atraso / float(window)

        soma = sum(target_row[1:]) / (25.0 * 15.0)
        pares = sum(1 for x in target_row[1:] if x % 2 == 0) / 15.0

        X_seq.append(seq)
        X_freq.append(freq_vec)
        X_atraso.append(atraso_vec)
        X_global.append([soma, pares])
        y.append(target)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_freq, dtype=np.float32),
        np.array(X_atraso, dtype=np.float32),
        np.array(X_global, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )

# ===================== LOSS SERIALIZABLE =====================
@tf.keras.utils.register_keras_serializable(package="custom_losses")
class WeightedBCE(tf.keras.losses.Loss):
    """Binary cross-entropy with a positive-class weight — serializável.
    Usamos uma classe (não closure) para facilitar serialização entre versões.
    """
    def __init__(self, pos_weight=3.0, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss_pos = - self.pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weight": self.pos_weight})
        return cfg

# ===================== MODELS (construtores) =====================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

def build_lstm_ls15(seq_shape, lr=1e-3):
    seq_input = Input(shape=seq_shape, name="seq_input")
    x = LSTM(128, return_sequences=False)(seq_input)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(25, activation="sigmoid")(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

def build_lstm_ls14_hybrid(seq_shape, hist_shape, lr=1e-3, pos_weight=3.0):
    seq_input = Input(shape=seq_shape, name="seq_input")
    hist_input = Input(shape=hist_shape, name="hist_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    h = Dense(8, activation="relu")(hist_input)

    z = Concatenate()([x, h])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.2)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, hist_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=WeightedBCE(pos_weight=pos_weight))
    return model
def build_lstm_ls14pp_hybrid(seq_shape, lr=1e-3, pos_weight=3.0):
    seq_input = Input(shape=seq_shape, name="seq_input")
    hist_input = Input(shape=(1,), name="hist_input")
    freq_input = Input(shape=(25,), name="freq_input")
    atraso_input = Input(shape=(25,), name="atraso_input")
    global_input = Input(shape=(2,), name="global_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    h = Dense(8, activation="relu")(hist_input)
    f = Dense(32, activation="relu")(freq_input)
    a = Dense(32, activation="relu")(atraso_input)
    g = Dense(8, activation="relu")(global_input)

    z = Concatenate()([x, h, f, a, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, hist_input, freq_input, atraso_input, global_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=WeightedBCE(pos_weight=pos_weight))
    return model
def build_lstm_ls15pp(seq_shape, lr=1e-3):
    seq_input = Input(shape=seq_shape, name="seq_input")
    freq_input = Input(shape=(25,), name="freq_input")
    atraso_input = Input(shape=(25,), name="atraso_input")
    global_input = Input(shape=(2,), name="global_input")

    x = LSTM(128)(seq_input)
    x = Dropout(0.2)(x)
    f = Dense(32, activation="relu")(freq_input)
    a = Dense(32, activation="relu")(atraso_input)
    g = Dense(8, activation="relu")(global_input)

    z = Concatenate()([x, f, a, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    out = Dense(25, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, freq_input, atraso_input, global_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

# ===================== TREINO (salvando formatos compatíveis) =====================
def _save_both_formats(model: tf.keras.Model, base_path: str):
    """Salva um modelo em dois formatos:
       - Keras v3 single-file (.keras) quando base_path termina com .keras
       - TensorFlow SavedModel em base_path + '_saved'
    """
    # salva .keras (arquivo único) se foi solicitado
    if base_path.endswith('.keras') or base_path.endswith('.h5'):
        try:
            model.save(base_path)
            logging.info(f"Salvo Keras artifact: {base_path}")
        except Exception as e:
            logging.warning(f"Falha ao salvar .keras: {e}")
    else:
        try:
            model.save(base_path + '.keras')
            logging.info(f"Salvo Keras artifact: {base_path}.keras")
        except Exception as e:
            logging.warning(f"Falha ao salvar .keras: {e}")

    # salva formato SavedModel (mais resistente a desserialização entre versões)
    tf_dir = base_path.replace('.keras', '') + '_saved'
    try:
        model.save(tf_dir, save_format='tf')
        logging.info(f"Salvo TF SavedModel em: {tf_dir}")
    except Exception as e:
        logging.warning(f"Falha ao salvar SavedModel: {e}")

# Treinos (apenas wrappers que chamam os construtores e salvam corretamente)
# Treinos (wrappers que chamam os construtores, constroem datasets e salvam corretamente)

def convert_keras_to_savedmodel(keras_path, saved_dir=None):
    if not os.path.exists(keras_path):
        raise FileNotFoundError(keras_path)

    model = load_model(keras_path, compile=False)

    if saved_dir is None:
        saved_dir = keras_path.replace(".keras", "_saved")

    # Salvar como SavedModel (pasta)
    model.export(saved_dir)   # método novo no Keras 3
    print(f"Modelo salvo em {saved_dir}")



def train_ls15(rows, window=50, epochs=50, batch_size=32, out_path=os.path.join(MODEL_DIR, "modelo_ls15pp.keras")):
    """Treina o modelo LS15++ a partir de `rows` (lista de concursos) e salva em out_path (e também _saved/).
    Retorna (model, history).
    """
    # build dataset
    X_seq, X_freq, X_atraso, X_global, y = build_dataset_ls15pp(rows, window)
    seq_shape = (X_seq.shape[1], X_seq.shape[2])

    model = build_lstm_ls15pp(seq_shape, lr=1e-3)

    # callbacks
    checkpoint = ModelCheckpoint(out_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(out_path.replace('.keras', '') + '_training_log.csv', append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        [X_seq, X_freq, X_atraso, X_global], y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint, csv_logger, early_stop],
        verbose=2
    )

    # garantir salvamento em ambos formatos
    _save_both_formats(model, out_path)
    return model, history

def train_ls14pp(rows, rep_map, window=50, epochs=50, batch_size=32, out_path=os.path.join(MODEL_DIR, "modelo_ls14pp.keras")):
    """Treina o modelo LS14++ (hybrid) a partir de rows + rep_map e salva em out_path.
    Retorna (model, history).
    """
    X_seq, X_hist, X_freq, X_atraso, X_global, y = build_dataset_ls14pp(rows, rep_map, window)
    seq_shape = (X_seq.shape[1], X_seq.shape[2])

    model = build_lstm_ls14pp_hybrid(seq_shape, lr=1e-3, pos_weight=3.0)

    # callbacks
    checkpoint = ModelCheckpoint(out_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(out_path.replace('.keras', '') + '_training_log.csv', append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        [X_seq, X_hist, X_freq, X_atraso, X_global], y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint, csv_logger, early_stop],
        verbose=2
    )

    _save_both_formats(model, out_path)
    return model, history

# util adicional para treinar a partir do banco (conveniência)
def train_from_db(model_key: str = 'LS15++', last_n: int = 1000, window: int = 50, epochs: int = 50, batch_size: int = 32):
    """Convenience: busca history do DB (fetch_history) e chama o wrapper de treino adequado.
    model_key: 'LS14++' ou 'LS15++'
    """
    rows = None
    rep_map = None
    try:
        rows, rep_map = fetch_history(last_n=last_n, include_repeats=True)
    except Exception:
        # tentar sem repeats
        rows = fetch_history(last_n=last_n)
        rep_map = {}

    if model_key == 'LS15++':
        return train_ls15(rows, window=window, epochs=epochs, batch_size=batch_size)
    elif model_key == 'LS14++':
        return train_ls14pp(rows, rep_map, window=window, epochs=epochs, batch_size=batch_size)
    else:
        raise ValueError('model_key deve ser LS14++ ou LS15++')



# =====================================================================
# =================== AUTO RETRAIN ROBUSTO (versão atualizada) ===========
# =====================================================================

# Salvamos em arquivo separado ao executar como script — para facilitar deploy, deixei a lógica aqui mas também exportei como
# código independente (aqui o bloco principal chama o retrain automaticamente quando invocado como script).

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
METRICS_DIR = os.environ.get("METRICS_DIR", "./metrics")
BACKUP_DIR = os.environ.get("BACKUP_DIR", "./backup_models")
EPOCHS = int(os.environ.get("EPOCHS", 50))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

MODELS = {
    "LS14++": "modelo_ls14pp.keras",
    "LS15++": "modelo_ls15pp.keras",
}

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def backup_model(model_file):
    src_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(src_path):
        dst_path = os.path.join(BACKUP_DIR, model_file)
        shutil.copy2(src_path, dst_path)
        logging.info(f"Backup criado para {model_file}")

# retrain_model robusto
def retrain_model(model_name: str, model_file: str, X, y):
    model_path = os.path.join(MODEL_DIR, model_file)
    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.csv")

    backup_model(model_file)

    # callbacks
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(metrics_file, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model = None

    # Tente carregar modelo existente (compile=False para evitar problemas de deserialização de optimizer)
    if os.path.exists(model_path):
        try:
            logging.info(f"Tentando carregar modelo existente: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logging.info("Modelo carregado com sucesso (compile=False). Vou recompilar localmente.")
            # recomponha loss caso necessário
            if model_name == 'LS14++':
                model.compile(optimizer='adam', loss=WeightedBCE(pos_weight=3.0), metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception as e:
            logging.warning(f"Falha ao carregar modelo existente: {e}")
            model = None

    # Se falhou ao carregar, construa novo modelo a partir das funções construtoras
    if model is None:
        logging.info("Construindo novo modelo a partir do construtor adequado.")
        try:
            import train_ls_models_v1 as train_module
        except Exception:
            # caso o arquivo original não exista, tente importar deste mesmo arquivo (se o usuário copiou)
            try:
                import train_ls_models_v1 as train_module
            except Exception as e:
                logging.error("Não foi possível importar train_ls_models_v1 para construir modelo: %s", e)
                raise

        if model_name == 'LS14++':
            # espera X = [X_seq, X_hist, X_freq, X_atraso, X_global]
            seq_shape = (X[0].shape[1], X[0].shape[2])
            model = train_module.build_lstm_ls14pp_hybrid(seq_shape, lr=1e-3, pos_weight=3.0)
        elif model_name == 'LS15++':
            seq_shape = (X[0].shape[1], X[0].shape[2])
            model = train_module.build_lstm_ls15pp(seq_shape, lr=1e-3)
        else:
            # fallback genérico
            model = build_simple_dense(X[0])

    # Treino
    try:
        if isinstance(X, list):
            logging.info(f"Treinando {model_name} com {len(X)} inputs (multi-input).")
            history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_split=0.2, callbacks=[checkpoint, csv_logger, early_stop])
        else:
            logging.info(f"Treinando {model_name} com single-input.")
            history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_split=0.2, callbacks=[checkpoint, csv_logger, early_stop])

        # garantir também um SavedModel mais portátil
        try:
            tf_save_dir = os.path.join(MODEL_DIR, model_file.replace('.keras', '') + '_saved')
            model.save(tf_save_dir, save_format='tf')
            logging.info(f"Também salvei SavedModel em {tf_save_dir}")
        except Exception as e:
            logging.warning(f"Falha ao salvar SavedModel final: {e}")

        logging.info(f"Treino do {model_name} finalizado com sucesso.")
    except Exception as e:
        logging.error(f"Erro durante o treino de {model_name}: {e}")
        traceback.print_exc()

    return model

# Fallback: construção simples (caso tudo falhe)
def build_simple_dense(X_sample):
    # X_sample pode ser array (n_samples, features) ou list
    if isinstance(X_sample, list):
        input_dim = X_sample[0].shape[1]
    else:
        input_dim = X_sample.shape[1]

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ========== MAIN do auto-retrain (usa pipeline_data se disponível; senão cai no fetch_history) =============
def main_auto_retrain():
    ensure_dirs()

    # primeira tentativa: pipeline_data.get_training_data()
    data_map = {}
    try:
        from pipeline_data import get_training_data
        logging.info("Usando pipeline_data.get_training_data() para obter X/y por modelo.")
        data_map = get_training_data()
        # espera dicionário: {'LS14++': (X,y), 'LS15++': (X,y)} ou similar
    except Exception as e:
        logging.warning(f"Não foi possível usar pipeline_data.get_training_data(): {e}")
        # fallback: gere usando fetch_history + builders
        try:
            # aqui usamos funções definidas acima (build_dataset_*)
            rows, rep_map = None, None
            try:
                rows, rep_map = fetch_history(last_n=1000, include_repeats=True)
            except Exception:
                # se fetch_history não estiver disponível (Session None), não conseguimos construir dados
                logging.error("fetch_history falhou — verifique DB / db.py. Não há dados para treinar.")
                return

            # LS14++
            try:
                X_seq, X_hist, X_freq, X_atraso, X_global, y14 = build_dataset_ls14pp(rows, rep_map, window=50)
                data_map['LS14++'] = ([X_seq, X_hist, X_freq, X_atraso, X_global], y14)
            except Exception as e:
                logging.warning(f"Não foi possível construir dataset LS14++: {e}")

            # LS15++
            try:
                X_seq15, X_freq15, X_atraso15, X_global15, y15 = build_dataset_ls15pp(rows, window=50)
                data_map['LS15++'] = ([X_seq15, X_freq15, X_atraso15, X_global15], y15)
            except Exception as e:
                logging.warning(f"Não foi possível construir dataset LS15++: {e}")

        except Exception as e:
            logging.error(f"Erro no fallback de construção de dados: {e}")
            traceback.print_exc()
            return

    # agora iterar sobre modelos configurados
    for model_name, model_file in MODELS.items():
        if model_name not in data_map:
            logging.warning(f"Dados para {model_name} não encontrados — pulando.")
            continue
        X, y = data_map[model_name]
        retrain_model(model_name, model_file, X, y)

# ===================== STREAMLIT EXAMPLE (módulo pequeno) =====================
# Nota: o Streamlit é apenas exemplo. Ele usa fetch_history + os builders acima para montar o último sample

def streamlit_app():
    import streamlit as st
    st.title("Gerador de palpites — carregador de modelos (exemplo)")

    model_dir = st.text_input("Pasta de modelos", value=MODEL_DIR)

    # lista modelos disponíveis
    files = []
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.keras') or os.path.isdir(os.path.join(model_dir, f)):
                files.append(f)
    model_choice = st.selectbox("Escolha o arquivo .keras (ou dir saved)", options=files)

    window = st.number_input("Janela (window)", min_value=10, max_value=200, value=50)

    if st.button("Carregar e prever"):
        if not model_choice:
            st.error("Escolha um modelo")
        else:
            try:
                model_path = os.path.join(model_dir, model_choice)
                # primeiro tente carregar diretório SavedModel correspondente
                if os.path.isdir(model_path):
                    model = tf.keras.models.load_model(model_path, compile=False)
                else:
                    # se for arquivo .keras, tente carregar; caso falhe, procure versão saved
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                    except Exception as e:
                        st.warning(f"Falha ao carregar .keras diretamente: {e}")
                        alt_saved = model_path.replace('.keras', '') + '_saved'
                        if os.path.isdir(alt_saved):
                            model = tf.keras.models.load_model(alt_saved, compile=False)
                        else:
                            raise

                st.success("Modelo carregado — agora montando último sample")
                # montar último sample (usa fetch_history)
                try:
                    rows = fetch_history(last_n=window + 5)
                except Exception as e:
                    st.error(f"fetch_history falhou: {e}")
                    return

                # decide qual builder usar autom.
                # Se o modelo aceitar multi-input (baseado no nome), use os builders acima
                if 'ls15' in model_choice.lower():
                    X, y = build_dataset_ls15(rows, window)
                    sample = X[-1:]
                    probs = model.predict(sample)[0]
                elif 'ls14' in model_choice.lower():
                    # pode ser ls14pp -> tenta o builder pp
                    try:
                        rows_rep = fetch_history(last_n=window + 5, include_repeats=True)
                        if isinstance(rows_rep, tuple):
                            rows2, rep_map = rows_rep
                        else:
                            rows2, rep_map = rows, {}
                    except Exception:
                        rows2, rep_map = rows, {}

                    try:
                        Xs = build_dataset_ls14pp(rows2, rep_map, window)
                        sample = [arr[-1:] for arr in Xs[:-1]]
                        probs = model.predict(sample)[0]
                    except Exception:
                        X, y = build_dataset_ls14(rows, {}, window)
                        sample = [arr[-1:] for arr in X]
                        probs = model.predict(sample)[0]
                else:
                    # fallback
                    X, y = build_dataset_ls15(rows, window)
                    sample = X[-1:]
                    probs = model.predict(sample)[0]

                # converte probs para top-15 ou por threshold
                top_n = 15
                top_idxs = np.argsort(probs)[-top_n:][::-1] + 1
                st.write("Palpite (top {} números):".format(top_n))
                st.write(list(top_idxs))

            except Exception as e:
                st.error(f"Erro ao carregar/prever: {e}")
                st.exception(e)

# ===================== ENTRYPOINTS =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', type=str, help='Converte um arquivo .keras para SavedModel: --convert path/to/model.keras')
    parser.add_argument('--savedir', type=str, default=None, help='Destino para conversão (--convert). Se omitido, usa <model>_saved')
    parser.add_argument('--auto-retrain', action='store_true', help='Executa o auto-retrain robusto')
    parser.add_argument('--streamlit', action='store_true', help='Roda um exemplo Streamlit (use: streamlit run thisfile.py -- --streamlit)')
    args = parser.parse_args()

    if args.convert:
        src = args.convert
        dst = args.savedir or src.replace('.keras', '') + '_saved'
        convert_keras_to_savedmodel(src, dst)
        print('Conversão finalizada:', dst)
    elif args.auto_retrain:
        main_auto_retrain()
    elif args.streamlit:
        # streamlit executa este módulo; no entanto recomenda-se rodar via `streamlit run thisfile.py -- --streamlit`
        streamlit_app()
    else:
        print('Nenhuma ação solicitada. Use --help para ver opções.')
