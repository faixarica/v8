# ============================================================
# ensemble.py  (FaixaBet LS16-Platinum - versão inteligente c/ Telemetria)
# Compatível com TensorFlow 2.12–2.15 (produção segura)
# ============================================================

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv
from tensorflow.keras.models import load_model

# --------------------------------------------------------------
# 🔹 Configuração opcional de logs
# --------------------------------------------------------------
VERBOSE = "--verbose" in sys.argv

# 🔹 Permite rodar o script isoladamente (CLI)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

# 🔹 Importa o sampler de forma relativa (corrigido)
try:
    from .sampler import gerar_palpites
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from sampler import gerar_palpites

# --------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------
def load_model_safe(path):
    """
    Carrega um modelo Keras com tratamento de compatibilidade.
    Tenta .keras, depois .h5 e por fim SavedModel.
    """
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print(f"✅ Modelo carregado (.keras): {os.path.basename(path)}")
        return model
    except Exception as e1:
        print(f"[!] Falha ao carregar {path}: {e1}")

        # fallback .h5
        alt_h5 = path.replace(".keras", ".h5")
        if os.path.exists(alt_h5):
            try:
                print(f"⚙️ Tentando fallback .h5 → {alt_h5}")
                model = tf.keras.models.load_model(alt_h5, compile=False)
                print(f"✅ Modelo carregado (.h5): {os.path.basename(alt_h5)}")
                return model
            except Exception as e2:
                print(f"[!] Falha ao carregar fallback .h5: {e2}")

        # fallback SavedModel (pasta)
        alt_dir = path.replace(".keras", "")
        if os.path.isdir(alt_dir):
            try:
                print(f"⚙️ Tentando fallback SavedModel → {alt_dir}")
                model = tf.keras.models.load_model(alt_dir, compile=False)
                print(f"✅ Modelo carregado (SavedModel): {os.path.basename(alt_dir)}")
                return model
            except Exception as e3:
                print(f"[!] Falha ao carregar fallback SavedModel: {e3}")

        return None


def predict_from_models(models, X, weights=None):
    """Gera predições ponderadas de vários modelos."""
    preds = []
    for i, m in enumerate(models):
        if m is None:
            continue
        try:
            p = m.predict(X, verbose=0)[0]
            w = weights[i] if weights is not None else 1.0
            preds.append(p * w)
        except Exception as e:
            print(f"[!] Falha ao prever com modelo {i}: {e}")
    if not preds:
        return np.zeros(25)
    return np.mean(preds, axis=0)


# --------------------------------------------------------------
# Função principal
# --------------------------------------------------------------
def gerar_palpite_ensemble(base_dir=None, window=50):
    """
    Ensemble LS16-Platinum:
    Combina recent/mid/global (LS14++ + LS15++) com jitter, temperatura e pesos dinâmicos.
    Retorna 15 dezenas mais prováveis como lista de strings.
    Também grava logs locais e telemetria no Postgres.
    """

    # 🔹 Caminho base automático
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "models", "prod")

    print("=============================================================")
    print("🔮 Gerando palpite LS16 (ensemble híbrido LS14+LS15)")
    print("=============================================================")
    print("📂 Diretório base:", base_dir)

    # 🔹 Define semente aleatória dinâmica
    np.random.seed(None)
    seed_value = np.random.randint(0, 999999)
    np.random.seed(seed_value)
    print(f"🧬 Seed aleatória: {seed_value}")

    # ----------------------------------------------------------
    # 🔹 Carrega modelos LS14/LS15 (recent/mid/global)
    # ----------------------------------------------------------
    models = []
    for prefix in ["recent", "mid", "global"]:
        for tipo in ["ls14pp", "ls15pp"]:
            path = os.path.join(base_dir, f"{prefix}_{tipo}_final.keras")
            if os.path.exists(path):
                m = load_model_safe(path)
                if m:
                    models.append(m)

    if not models:
        print("❌ Nenhum modelo encontrado em", base_dir)
        return []

    # ----------------------------------------------------------
    # 🔹 Gera entrada contextual (últimos 50 sorteios + jitter)
    # ----------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), "dados", "rows.npy")

    if not os.path.exists(data_path):
        print("⚠️ rows.npy não encontrado. Tentando gerar com prepare_real_data_db_v3...")
        try:
            from prepare_real_data_db_v3 import main as gerar_dados
            gerar_dados()
        except Exception as e:
            print(f"[!] Falha ao gerar rows.npy automaticamente: {e}")

    if os.path.exists(data_path):
        ultimos = np.load(data_path, allow_pickle=True)
        X_base = ultimos[-window:]
        # 🧠 Se tiver 15 dezenas ao invés de 25 colunas, converte para matriz binária
        if X_base.shape[1] <= 16:
            matriz = np.zeros((X_base.shape[0], 25), dtype=np.float32)
            for i, linha in enumerate(X_base):
                for d in linha[-15:]:
                    if 1 <= int(d) <= 25:
                        matriz[i, int(d) - 1] = 1.0
            X_base = matriz
        elif X_base.shape[1] != 25:
            X_base = X_base[:, -25:]
        print(f"📊 Entrada real baseada nos últimos {X_base.shape[0]} concursos + jitter")
    else:
        X_base = np.random.rand(window, 25)
        print("⚠️ Dados reais não encontrados — usando ruído puro")

    jitter = np.random.normal(0, 0.05, size=X_base.shape)
    X = np.expand_dims(X_base + jitter, axis=0).astype(np.float32)

    # ----------------------------------------------------------
    # 🔹 Define pesos diferenciados e temperatura
    # ----------------------------------------------------------
    weights = np.linspace(0.6, 0.9, len(models))
    temperature = 0.9 + np.random.rand() * 0.2
    print(f"🌡️ Temperatura: {temperature:.3f}")
    print(f"⚖️ Pesos médios aplicados: {weights.round(2).tolist()}")

    # ----------------------------------------------------------
    # 🔹 Faz predição e aplica temperatura
    # ----------------------------------------------------------
    preds = predict_from_models(models, X, weights=weights)

    preds = np.maximum(preds, 1e-9)
    p_temp = preds ** (1.0 / float(temperature))
    p_temp = p_temp / p_temp.sum()

    # ----------------------------------------------------------
    # 🌀 Exploração controlada (Dirichlet + ruído leve em pesos)
    # ----------------------------------------------------------
    alpha = 0.35
    noise = np.random.dirichlet(np.ones(25) * alpha)
    lam = 0.18
    p_mix = (1.0 - lam) * p_temp + lam * noise
    p_mix = np.clip(p_mix / p_mix.sum(), 1e-9, 1.0)
    p_mix = p_mix / p_mix.sum()

    # ----------------------------------------------------------
    # 🎯 Amostragem múltipla com diversidade via sampler.py
    # ----------------------------------------------------------
    palpites = gerar_palpites(
        p_mix,
        n_palpites=13,
        temperatura=float(temperature),
        diversify_strength=0.88,
    )

    dezenas_principal = [f"{d:02d}" for d in palpites[0]]
    print("🎯 Palpite principal LS16:", dezenas_principal)
    print("📦 Total de palpites gerados:", len(palpites))

    # ----------------------------------------------------------
    # 💾 Salva todos os palpites no CSV
    # ----------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ls16_output.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, seed_value, round(temperature, 3)] + dezenas_principal)

    print(f"💾 Principal salvo em: {out_file}")
    print(f"🕒 Registro: {timestamp}")

    # ----------------------------------------------------------
    # 🗄️ Telemetria no Postgres
    # ----------------------------------------------------------
    try:
        import psycopg2

        pg_url = (
            os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("PG_URI")
        )

        if pg_url:
            conn = psycopg2.connect(pg_url)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO telemetria (modelo, plano, seed, temperatura, dezenas, origem, sucesso)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    "LS16",
                    "Platinum",
                    seed_value,
                    round(temperature, 3),
                    [int(d) for d in dezenas_principal],
                    "streamlit" if os.getenv("STREAMLIT_SERVER_PORT") else "cli",
                    True,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
            print("🗄️ Telemetria salva no banco (Postgres)")
        else:
            print("⚠️ Telemetria não gravada — DATABASE_URL ausente.")
    except Exception as e:
        print(f"[!] Falha ao gravar telemetria: {e}")

    print("=============================================================")
    return palpites


# 🔹 Execução direta (modo CLI)
if __name__ == "__main__":
    gerar_palpite_ensemble()
