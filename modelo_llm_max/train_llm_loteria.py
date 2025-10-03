# -------------------- [1] IMPORTS --------------------
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

# -------------------- [2] CALLBACK --------------------
class LoteriaMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_X, val_y, k=15, best_path=None):
        super().__init__()
        self.val_X = val_X
        self.val_y = val_y
        self.k = k
        self.best_path = best_path
        self.best_score = -1

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_X, verbose=0)
        topk = np.argsort(preds, axis=1)[:, -self.k:]
        acertos = [len(set(y).intersection(set(p))) for y, p in zip(self.val_y, topk)]
        mean_hits = np.mean(acertos)

        logging.info(f"Epoch {epoch+1}: média de acertos top-{self.k} = {mean_hits:.2f}")

        if mean_hits > self.best_score:
            self.best_score = mean_hits
            if self.best_path:
                self.model.save(self.best_path)
                logging.info(f"Novo melhor modelo salvo em {self.best_path}")

# -------------------- [3] DADOS FAKE (ajuste para seus históricos reais) --------------------
def gerar_dados_fake(n_amostras=1000, window=50, dezenas=25):
    X = np.random.randint(0, 2, size=(n_amostras, window, dezenas))
    y = np.random.randint(0, 2, size=(n_amostras, dezenas))
    return X.astype("float32"), y.astype("float32")

# -------------------- [4] MODELO --------------------
def build_model(window=50, dezenas=25):
    inp = keras.Input(shape=(window, dezenas))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    out = layers.Dense(dezenas, activation="sigmoid")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -------------------- [5] TREINAMENTO --------------------
def train_model(model_name, last_n, window, epochs, batch, out_dir):
    logging.info(f"Treinando {model_name} | last_n={last_n}, window={window}")

    X, y = gerar_dados_fake(n_amostras=last_n, window=window, dezenas=25)

    # Split treino/validação
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(window=window, dezenas=25)

    best_path = os.path.join(out_dir, f"{model_name}_best.keras")
    metrics_cb = LoteriaMetricsCallback(val_X, val_y, k=15, best_path=best_path)

    model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=epochs,
        batch_size=batch,
        callbacks=[metrics_cb],
        verbose=1
    )

    final_path = os.path.join(out_dir, f"{model_name}_final.keras")
    model.save(final_path)
    logging.info(f"Modelo final salvo em {final_path}")

# -------------------- [6] MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ls14", "ls15", "both"], required=True)
    parser.add_argument("--last_n", type=int, default=200)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.model in ["ls15", "both"]:
        train_model("ls15pp", args.last_n, args.window, args.epochs, args.batch, args.out)

    if args.model in ["ls14", "both"]:
        train_model("ls14pp", args.last_n, args.window, args.epochs, args.batch, args.out)
