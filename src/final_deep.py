"""
Final-stage experiment E1 (deep learning part): 5 x 5 repeated stratified
cross-validation of the LSTM and GRU models from the interim report, plus a
feed-forward MLP as a tabular deep learning baseline.

The folds are identical to those used for the classical models
(final_classical.py), so fold-level results can be compared pairwise.

Output: results/final/cv_folds_deep.csv  (one row per model per fold)
Runtime: about 30-45 minutes on a 2-core CPU.
"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ckd_common import SEED, RES, load_clean, prep, folds, metrics

OUT = os.path.join(RES, "cv_folds_deep.csv")
EPOCHS, BATCH, PATIENCE, VAL_SPLIT = 150, 16, 15, 0.2


def make_model(kind, n_features):
    """Same architecture as the interim report: two recurrent layers (64, 32),
    dropout 0.3, Dense(16), sigmoid output, Adam 1e-3. The MLP replaces the
    recurrent layers with dense layers of the same width."""
    if kind == "MLP":
        body = [keras.Input(shape=(n_features,)),
                layers.Dense(64, activation="relu"), layers.Dropout(0.3),
                layers.Dense(32, activation="relu"), layers.Dropout(0.3)]
    else:
        R = layers.LSTM if kind == "LSTM" else layers.GRU
        body = [keras.Input(shape=(n_features, 1)),
                R(64, return_sequences=True), layers.Dropout(0.3),
                R(32), layers.Dropout(0.3)]
    m = keras.Sequential(body + [layers.Dense(16, activation="relu"),
                                 layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def fit_predict(kind, Xtr, ytr, Xte):
    tf.keras.utils.set_random_seed(SEED)
    shape = (lambda a: a.reshape(a.shape[0], a.shape[1], 1)) if kind != "MLP" else (lambda a: a)
    a_tr = shape(Xtr.values.astype("float32"))
    a_te = shape(Xte.values.astype("float32"))
    m = make_model(kind, Xtr.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)
    h = m.fit(a_tr, ytr.values.astype("float32"), validation_split=VAL_SPLIT,
              epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)
    return m.predict(a_te, verbose=0).ravel(), len(h.history["loss"])


def main():
    X, y = load_clean()
    fl = folds(X, y)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.model, prev.fold))
    kinds = sys.argv[1:] or ["MLP", "GRU", "LSTM"]
    for k, (tr, te) in enumerate(fl):
        Xtr, Xte = prep(X.iloc[tr], X.iloc[te])
        for kind in kinds:
            if (kind, k) in done:
                continue
            t0 = time.time()
            prob, ep = fit_predict(kind, Xtr, y.iloc[tr], Xte)
            row = {"model": kind, "fold": k, "epochs": ep, **metrics(y.iloc[te], prob),
                   "seconds": round(time.time() - t0, 1)}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"fold {k:02d} {kind:5s} acc={row['accuracy']:.3f} rec={row['recall']:.3f} "
                  f"f1={row['f1']:.3f} epochs={ep} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
