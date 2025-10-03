# build_models.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class WeightedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weight=3.0, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss_pos = -self.pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self):
        return {"pos_weight": self.pos_weight}

def bce_with_topk_proxy(k=15, tau=0.05, alpha=1.0, pos_weight=1.0):
    def loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = tf.keras.losses.binary_crossentropy(y_true, p)
        logit = tf.math.log(p / (1.0 - p + eps) + eps)
        soft = tf.nn.softmax(logit / (tau + eps), axis=-1)
        exp_hits = tf.reduce_sum(soft * y_true, axis=-1) * tf.cast(k, tf.float32)
        reward = exp_hits / tf.cast(k, tf.float32)
        return bce - alpha * reward
    return loss

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